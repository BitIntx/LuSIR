from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.models import LRToLatentPredictor, LatentResidualRefiner
from sr_diffusion.utils import autocast_context
from sr_diffusion.utils import get_device, load_config, seed_everything
from tools.infer.infer_detail_branch import (
    load_detail_branch,
    load_detail_mask_predictor,
    resolve_checkpoint as resolve_detail_checkpoint,
    tiled_detail,
)
from tools.infer.infer_diffusion import (
    edge_pad_image,
    float_array_to_pil,
    inference_module_dtype,
    load_autoencoder,
    load_condition_encoder,
    pil_to_tensor,
    resolve_path,
    tile_blend_mask,
    tile_positions,
)
from tools.infer.infer_residual_refiner import load_refiner, resolve_checkpoint as resolve_refiner_checkpoint, tiled_refine


VARIANTS: dict[str, dict[str, Any]] = {
    "stage2_base": {
        "kind": "stage2",
        "config": Path("configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml"),
        "outputs": ("base",),
    },
    "stage2_guarded_detail_v2": {
        "kind": "stage2",
        "config": Path("configs/hf/latent_pretrain_photo130k_lsdir_dual_detail_guarded_v2.yaml"),
        "outputs": ("base",),
    },
    "detail_v1d": {
        "kind": "detail",
        "config": Path("configs/hf/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml"),
        "outputs": ("base", "detail"),
    },
    "detail_v2_masked": {
        "kind": "detail",
        "config": Path("configs/hf/detail_branch_v2_masked_photo130k_lsdir.yaml"),
        "outputs": ("base", "detail"),
    },
    "refiner_v2": {
        "kind": "refiner",
        "config": Path("configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml"),
        "outputs": ("condition", "refined"),
    },
}

TTA_OPTIONS = {
    "off": ("identity",),
    "hflip": ("identity", "hflip"),
    "x8": ("identity", "hflip", "vflip", "rot180", "rot90", "rot270", "transpose", "transverse"),
}


def build_stage2_encoder(model_config: dict[str, Any]) -> torch.nn.Module:
    model_type = str(model_config.get("type", "lr_to_latent_predictor"))
    if model_type == "lr_to_latent_predictor":
        return LRToLatentPredictor.from_config(model_config)
    if model_type == "latent_residual_refiner":
        return LatentResidualRefiner.from_config(model_config)
    raise ValueError(f"Unsupported Stage 2 model type: {model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic LuSIR path on standard full-image SR benchmarks.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--tta", choices=sorted(TTA_OPTIONS), default="off")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_manifest(path: Path, datasets: list[str], limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "id", "scale", "hr_path", "lr_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must include {sorted(required)}")
    if datasets:
        rows = [row for row in rows if row["dataset"] in datasets]
    return rows[:limit] if limit > 0 else rows


def resolve_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def apply_tta_transform(image: Image.Image, transform: str) -> Image.Image:
    if transform == "identity":
        return image.copy()
    if transform == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if transform == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if transform == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)
    if transform == "rot90":
        return image.transpose(Image.Transpose.ROTATE_90)
    if transform == "rot270":
        return image.transpose(Image.Transpose.ROTATE_270)
    if transform == "transpose":
        return image.transpose(Image.Transpose.TRANSPOSE)
    if transform == "transverse":
        return image.transpose(Image.Transpose.TRANSVERSE)
    raise ValueError(f"Unknown TTA transform: {transform}")


def invert_tta_transform(image: Image.Image, transform: str) -> Image.Image:
    if transform == "rot90":
        return image.transpose(Image.Transpose.ROTATE_270)
    if transform == "rot270":
        return image.transpose(Image.Transpose.ROTATE_90)
    return apply_tta_transform(image, transform)


def average_tta_images(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("TTA requires at least one image")
    arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) for image in images]
    reference_shape = arrays[0].shape
    if any(array.shape != reference_shape for array in arrays):
        shapes = [array.shape for array in arrays]
        raise ValueError(f"TTA outputs have mismatched shapes: {shapes}")
    merged = np.clip(np.round(np.stack(arrays, axis=0).mean(axis=0)), 0, 255).astype(np.uint8)
    return Image.fromarray(merged, mode="RGB")


def resolve_stage2_checkpoint(config: dict[str, Any], requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    inference_checkpoint = config.get("inference", {}).get("checkpoint") or config.get("checkpoint")
    if inference_checkpoint:
        candidates.append(Path(inference_checkpoint))
    project_output = config.get("project", {}).get("output_dir")
    if project_output:
        candidates.extend(
            [
                Path(project_output) / "checkpoints" / "best_eval_decoded.pt",
                Path(project_output) / "checkpoints" / "latest.pt",
            ]
        )
    for candidate in candidates:
        path = resolve_path(config, candidate.expanduser())
        if path.exists():
            return path.resolve()
    formatted = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find Stage 2 checkpoint. Checked:\n{formatted}")


def load_stage2_encoder(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    dtype_name: str | None,
) -> tuple[torch.nn.Module, int]:
    encoder = build_stage2_encoder(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(checkpoint["model"])
    dtype = inference_module_dtype(device, dtype_name)
    if dtype is None:
        encoder = encoder.to(device)
    else:
        encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    return encoder, int(checkpoint.get("step", 0))


@torch.no_grad()
def stage2_batch(
    vae: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    dtype_name: str,
) -> torch.Tensor:
    device = lr.device
    with autocast_context(device, dtype_name):
        condition = condition_encoder(normalize_image(lr), domain_id)
        decoded = denormalize(vae.decode(condition)).float()
    return decoded.float()


def tiled_stage2(
    vae: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    lr_image: Image.Image,
    domain_id_value: int,
    *,
    scale: int,
    tile_lr_size: int,
    overlap_lr: int,
    tile_batch_size: int,
    dtype_name: str,
    device: torch.device,
) -> Image.Image:
    if tile_batch_size <= 0:
        raise ValueError(f"tile_batch_size must be positive: {tile_batch_size}")
    if overlap_lr < 0 or overlap_lr >= tile_lr_size:
        raise ValueError(f"tile_overlap must be in [0, {tile_lr_size - 1}], got {overlap_lr}")

    original_width, original_height = lr_image.size
    padded = edge_pad_image(lr_image, tile_lr_size, tile_lr_size)
    padded_width, padded_height = padded.size
    x_positions = tile_positions(padded_width, tile_lr_size, overlap_lr)
    y_positions = tile_positions(padded_height, tile_lr_size, overlap_lr)
    tile_hr_size = tile_lr_size * scale
    overlap_hr = overlap_lr * scale
    canvas = np.zeros((padded_height * scale, padded_width * scale, 3), dtype=np.float32)
    weights = np.zeros((padded_height * scale, padded_width * scale, 1), dtype=np.float32)
    tiles = [(x, y) for y in y_positions for x in x_positions]
    num_batches = (len(tiles) + tile_batch_size - 1) // tile_batch_size
    print(
        f"tile_stage2 lr_size={original_width}x{original_height} padded={padded_width}x{padded_height} "
        f"tiles={len(tiles)} tile={tile_lr_size} overlap={overlap_lr} batch={tile_batch_size}",
        flush=True,
    )
    for batch_start in range(0, len(tiles), tile_batch_size):
        batch_index = batch_start // tile_batch_size + 1
        batch_coords = tiles[batch_start : batch_start + tile_batch_size]
        batch_images = [
            pil_to_tensor(padded.crop((x, y, x + tile_lr_size, y + tile_lr_size))) for x, y in batch_coords
        ]
        lr_tensor = torch.stack(batch_images, dim=0).to(device)
        domain_ids = torch.full((len(batch_coords),), domain_id_value, device=device, dtype=torch.long)
        decoded = stage2_batch(
            vae=vae,
            condition_encoder=condition_encoder,
            lr=lr_tensor,
            domain_id=domain_ids,
            dtype_name=dtype_name,
        )
        for tile, (x, y) in zip(decoded, batch_coords, strict=True):
            mask = tile_blend_mask(
                tile_hr_size,
                overlap_hr,
                left_edge=x == 0,
                right_edge=x + tile_lr_size >= padded_width,
                top_edge=y == 0,
                bottom_edge=y + tile_lr_size >= padded_height,
            )
            x0 = x * scale
            y0 = y * scale
            canvas[y0 : y0 + tile_hr_size, x0 : x0 + tile_hr_size] += (
                tile.detach().float().cpu().permute(1, 2, 0).numpy() * mask
            )
            weights[y0 : y0 + tile_hr_size, x0 : x0 + tile_hr_size] += mask
        print(
            f"tile_batch={batch_index}/{num_batches} "
            f"done={min(batch_start + len(batch_coords), len(tiles))}/{len(tiles)}",
            flush=True,
        )

    image = canvas / np.maximum(weights, 1e-6)
    image = image[: original_height * scale, : original_width * scale]
    return float_array_to_pil(image)


def run_variant_once(
    *,
    kind: str,
    vae: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    model: torch.nn.Module | None,
    detail_mask_predictor: torch.nn.Module | None,
    detail_mask_floor: float,
    lr_image: Image.Image,
    domain_id: int,
    scale: int,
    tile_lr_size: int,
    overlap_lr: int,
    tile_batch_size: int,
    dtype_name: str,
    device: torch.device,
    strength: float,
) -> tuple[Image.Image, ...]:
    if kind == "stage2":
        return (
            tiled_stage2(
                vae,
                condition_encoder,
                lr_image,
                domain_id,
                scale=scale,
                tile_lr_size=tile_lr_size,
                overlap_lr=overlap_lr,
                tile_batch_size=tile_batch_size,
                dtype_name=dtype_name,
                device=device,
            ),
        )
    if kind == "detail":
        if model is None:
            raise ValueError("Detail branch model is not loaded")
        return tiled_detail(
            vae,
            condition_encoder,
            model,
            lr_image,
            domain_id,
            scale=scale,
            tile_lr_size=tile_lr_size,
            overlap_lr=overlap_lr,
            tile_batch_size=tile_batch_size,
            dtype_name=dtype_name,
            device=device,
            detail_strength=strength,
            detail_mask_predictor=detail_mask_predictor,
            detail_mask_floor=detail_mask_floor,
        )
    if model is None:
        raise ValueError("Residual refiner model is not loaded")
    return tiled_refine(
        vae,
        condition_encoder,
        model,
        lr_image,
        domain_id,
        scale=scale,
        tile_lr_size=tile_lr_size,
        overlap_lr=overlap_lr,
        tile_batch_size=tile_batch_size,
        dtype_name=dtype_name,
        device=device,
        residual_strength=strength,
    )


def run_variant_with_tta(
    *,
    kind: str,
    output_names: tuple[str, ...],
    tta_mode: str,
    vae: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    model: torch.nn.Module | None,
    detail_mask_predictor: torch.nn.Module | None,
    detail_mask_floor: float,
    lr_image: Image.Image,
    domain_id: int,
    scale: int,
    tile_lr_size: int,
    overlap_lr: int,
    tile_batch_size: int,
    dtype_name: str,
    device: torch.device,
    strength: float,
) -> tuple[Image.Image, ...]:
    transforms = TTA_OPTIONS[tta_mode]
    if transforms == ("identity",):
        return run_variant_once(
            kind=kind,
            vae=vae,
            condition_encoder=condition_encoder,
            model=model,
            detail_mask_predictor=detail_mask_predictor,
            detail_mask_floor=detail_mask_floor,
            lr_image=lr_image,
            domain_id=domain_id,
            scale=scale,
            tile_lr_size=tile_lr_size,
            overlap_lr=overlap_lr,
            tile_batch_size=tile_batch_size,
            dtype_name=dtype_name,
            device=device,
            strength=strength,
        )

    restored_by_output: list[list[Image.Image]] = [[] for _ in output_names]
    for transform in transforms:
        transformed_lr = apply_tta_transform(lr_image, transform)
        transformed_outputs = run_variant_once(
            kind=kind,
            vae=vae,
            condition_encoder=condition_encoder,
            model=model,
            detail_mask_predictor=detail_mask_predictor,
            detail_mask_floor=detail_mask_floor,
            lr_image=transformed_lr,
            domain_id=domain_id,
            scale=scale,
            tile_lr_size=tile_lr_size,
            overlap_lr=overlap_lr,
            tile_batch_size=tile_batch_size,
            dtype_name=dtype_name,
            device=device,
            strength=strength,
        )
        if len(transformed_outputs) != len(output_names):
            raise ValueError(
                f"Variant returned {len(transformed_outputs)} outputs; expected {len(output_names)} for {output_names}"
            )
        for index, image in enumerate(transformed_outputs):
            restored_by_output[index].append(invert_tta_transform(image, transform))

    return tuple(average_tta_images(images) for images in restored_by_output)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    variant = VARIANTS[args.variant]
    config_path = args.config or variant["config"]
    config = load_config(config_path)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    scale = int(config["data"].get("scale", 4))
    tile_lr_size = int(config["data"]["hr_size"]) // scale
    domain_id = int(config["data"].get("domains", {"photo": 0})["photo"])
    vae = load_autoencoder(config, device, dtype_name)

    kind = variant["kind"]
    detail_mask_predictor = None
    detail_mask_step = 0
    detail_mask_floor = 0.0
    if kind == "stage2":
        checkpoint = resolve_stage2_checkpoint(config, args.checkpoint)
        condition_encoder, checkpoint_step = load_stage2_encoder(config, checkpoint, device, dtype_name)
        model = None
    elif kind == "detail":
        condition_encoder = load_condition_encoder(config, device, dtype_name)
        checkpoint = resolve_detail_checkpoint(config, args.checkpoint)
        model, checkpoint_step = load_detail_branch(config, checkpoint, device)
        detail_mask_predictor, detail_mask_step = load_detail_mask_predictor(config, device)
        detail_mask_floor = float(config.get("detail_mask", {}).get("floor", 0.0))
    else:
        condition_encoder = load_condition_encoder(config, device, dtype_name)
        checkpoint = resolve_refiner_checkpoint(config, args.checkpoint)
        model, checkpoint_step = load_refiner(config, checkpoint, device)

    rows = read_manifest(args.manifest, args.dataset, int(args.limit))
    timings: list[float] = []
    processed = 0
    for row in rows:
        sample_dir = args.output_dir / row["dataset"] / row["id"]
        output_names = variant["outputs"]
        output_paths = [sample_dir / f"{name}.png" for name in output_names]
        if args.skip_existing and all(path.exists() for path in output_paths):
            print(f"skip {row['dataset']}/{row['id']}", flush=True)
            continue
        lr_image = Image.open(resolve_manifest_path(args.manifest, row["lr_path"])).convert("RGB")
        started = time.perf_counter()
        results = run_variant_with_tta(
            kind=kind,
            output_names=output_names,
            tta_mode=str(args.tta),
            vae=vae,
            condition_encoder=condition_encoder,
            model=model,
            detail_mask_predictor=detail_mask_predictor,
            detail_mask_floor=detail_mask_floor,
            lr_image=lr_image,
            domain_id=domain_id,
            scale=scale,
            tile_lr_size=tile_lr_size,
            overlap_lr=int(args.tile_overlap),
            tile_batch_size=int(args.tile_batch_size),
            dtype_name=dtype_name,
            device=device,
            strength=float(args.strength),
        )
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        sample_dir.mkdir(parents=True, exist_ok=True)
        for image, output_path in zip(results, output_paths, strict=True):
            image.save(output_path)
        processed += 1
        print(f"processed {row['dataset']}/{row['id']} elapsed={elapsed:.2f}s", flush=True)

    summary = {
        "variant": args.variant,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "detail_mask_step": detail_mask_step if detail_mask_predictor is not None else None,
        "detail_mask_floor": detail_mask_floor if detail_mask_predictor is not None else None,
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "num_selected": len(rows),
        "num_processed": processed,
        "num_completed": sum(
            all((args.output_dir / row["dataset"] / row["id"] / f"{name}.png").exists() for name in variant["outputs"])
            for row in rows
        ),
        "datasets": args.dataset or sorted({row["dataset"] for row in rows}),
        "tile_overlap": int(args.tile_overlap),
        "tile_batch_size": int(args.tile_batch_size),
        "strength": float(args.strength),
        "tta": str(args.tta),
        "tta_transforms": list(TTA_OPTIONS[str(args.tta)]),
        "mean_seconds": sum(timings) / len(timings) if timings else None,
        "candidate_patterns": {
            name: str(args.output_dir / "{dataset}" / "{id}" / f"{name}.png") for name in variant["outputs"]
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.variant}_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
