from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from infer_diffusion import (
    edge_pad_image,
    float_array_to_pil,
    load_autoencoder,
    load_condition_encoder,
    pil_to_tensor,
    prepare_inputs,
    resolve_path,
    tensor_to_pil,
    tile_blend_mask,
    tile_positions,
)
from train_residual_refiner import BoundedResidualRefiner, apply_residual_strength, denormalize, normalize_image
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything


DEFAULT_HF_CHECKPOINT = Path("checkpoints/residual_refiner_stage2_xl_mild_best_eval_refined.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Stage 2 residual refiner inference.")
    parser.add_argument("--config", type=Path, default=Path("configs/residual_refiner_stage2_xl_mild_probe.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-lr", type=Path, help="Low-resolution RGB input image.")
    input_group.add_argument("--input-hr", type=Path, help="HR image to center-crop and degrade for controlled eval.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="photo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--degradation-preset", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resize-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile", action="store_true", help="Run tiled refiner inference for arbitrary-size LR input.")
    parser.add_argument("--tile-overlap", type=int, default=32, help="LR-pixel overlap between 128x128 tiles.")
    parser.add_argument("--tile-batch-size", type=int, default=4, help="Number of LR tiles to refine at once.")
    parser.add_argument(
        "--residual-strength",
        type=float,
        default=None,
        help="Scale the predicted residual correction. 1.0 is full correction; lower values are more conservative.",
    )
    return parser.parse_args()


def resolve_checkpoint(config: dict[str, Any], requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    inference_checkpoint = config.get("inference", {}).get("checkpoint") or config.get("checkpoint")
    if inference_checkpoint:
        candidates.append(Path(inference_checkpoint))
    candidates.append(DEFAULT_HF_CHECKPOINT)
    project_output = config.get("project", {}).get("output_dir")
    if project_output:
        candidates.append(Path(project_output) / "checkpoints" / "best_eval_refined.pt")

    for candidate in candidates:
        path = resolve_path(config, candidate.expanduser())
        if path.exists():
            return path.resolve()
    formatted = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find residual refiner checkpoint. Checked:\n{formatted}")


def load_refiner(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> tuple[BoundedResidualRefiner, int]:
    model = BoundedResidualRefiner.from_config(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, int(checkpoint.get("step", 0))


@torch.no_grad()
def refine_batch(
    vae: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    refiner: BoundedResidualRefiner,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    dtype_name: str,
    residual_strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = lr.device
    lr_input = normalize_image(lr)
    with autocast_context(device, dtype_name):
        condition = condition_encoder(lr_input, domain_id)
        _, residual, _ = refiner(condition, lr_input, domain_id)
        refined = apply_residual_strength(condition, residual, residual_strength)
        decoded_condition = denormalize(vae.decode(condition)).float()
        decoded_refined = denormalize(vae.decode(refined)).float()
    return decoded_condition, decoded_refined


def add_label(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 18
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill="black", font=font)
    return canvas


def make_grid(items: list[tuple[str, Image.Image]], output_path: Path, gap: int = 6) -> None:
    labeled = [add_label(image, label) for label, image in items]
    cell_width = max(image.width for image in labeled)
    cell_height = max(image.height for image in labeled)
    width = len(labeled) * cell_width + (len(labeled) + 1) * gap
    height = cell_height + 2 * gap
    sheet = Image.new("RGB", (width, height), "white")
    for index, image in enumerate(labeled):
        x = gap + index * (cell_width + gap)
        sheet.paste(image.convert("RGB"), (x, gap))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def tiled_refine(
    vae: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    refiner: BoundedResidualRefiner,
    lr_image: Image.Image,
    domain_id_value: int,
    *,
    scale: int,
    tile_lr_size: int,
    overlap_lr: int,
    tile_batch_size: int,
    dtype_name: str,
    device: torch.device,
    residual_strength: float,
) -> tuple[Image.Image, Image.Image]:
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
    condition_canvas = np.zeros((padded_height * scale, padded_width * scale, 3), dtype=np.float32)
    refined_canvas = np.zeros_like(condition_canvas)
    weights = np.zeros((padded_height * scale, padded_width * scale, 1), dtype=np.float32)
    tiles = [(x, y) for y in y_positions for x in x_positions]
    num_batches = (len(tiles) + tile_batch_size - 1) // tile_batch_size
    print(
        f"tile_refiner lr_size={original_width}x{original_height} padded={padded_width}x{padded_height} "
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
        decoded_condition, decoded_refined = refine_batch(
            vae=vae,
            condition_encoder=condition_encoder,
            refiner=refiner,
            lr=lr_tensor,
            domain_id=domain_ids,
            dtype_name=dtype_name,
            residual_strength=residual_strength,
        )
        for condition_tile, refined_tile, (x, y) in zip(decoded_condition, decoded_refined, batch_coords, strict=True):
            left_edge = x == 0
            top_edge = y == 0
            right_edge = x + tile_lr_size >= padded_width
            bottom_edge = y + tile_lr_size >= padded_height
            mask = tile_blend_mask(
                tile_hr_size,
                overlap_hr,
                left_edge=left_edge,
                right_edge=right_edge,
                top_edge=top_edge,
                bottom_edge=bottom_edge,
            )
            x0 = x * scale
            y0 = y * scale
            condition_canvas[y0 : y0 + tile_hr_size, x0 : x0 + tile_hr_size] += (
                condition_tile.detach().float().cpu().permute(1, 2, 0).numpy() * mask
            )
            refined_canvas[y0 : y0 + tile_hr_size, x0 : x0 + tile_hr_size] += (
                refined_tile.detach().float().cpu().permute(1, 2, 0).numpy() * mask
            )
            weights[y0 : y0 + tile_hr_size, x0 : x0 + tile_hr_size] += mask
        print(f"tile_batch={batch_index}/{num_batches} done={min(batch_start + len(batch_coords), len(tiles))}/{len(tiles)}")

    condition = condition_canvas / np.maximum(weights, 1e-6)
    refined = refined_canvas / np.maximum(weights, 1e-6)
    condition = condition[: original_height * scale, : original_width * scale]
    refined = refined[: original_height * scale, : original_width * scale]
    return float_array_to_pil(condition), float_array_to_pil(refined)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.degradation_preset is not None:
        config["data"]["degradation_preset"] = args.degradation_preset
    seed_everything(args.seed)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    residual_strength = float(
        args.residual_strength
        if args.residual_strength is not None
        else config.get("inference", {}).get("residual_strength", 1.0)
    )
    data_config = config["data"]
    domains = data_config.get("domains", {"photo": 0, "anime": 1})
    if args.domain not in domains:
        raise ValueError(f"Unknown domain '{args.domain}'. Available: {sorted(domains)}")
    if args.tile and args.input_hr:
        raise ValueError("--tile supports --input-lr only. Use --input-lr with an arbitrary-size LR image.")

    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_autoencoder(config, device, dtype_name)
    condition_encoder = load_condition_encoder(config, device, dtype_name)
    refiner, checkpoint_step = load_refiner(config, checkpoint_path, device)
    print(
        f"checkpoint={checkpoint_path} step={checkpoint_step} device={device} residual_strength={residual_strength:.2f}",
        flush=True,
    )

    scale = int(data_config.get("scale", 4))
    tile_lr_size = int(data_config["hr_size"]) // scale
    domain_id_value = int(domains[args.domain])

    if args.tile:
        lr_image = Image.open(args.input_lr).convert("RGB")
        lr_image.save(args.output_dir / "input_lr.png")
        condition_image, refined_image = tiled_refine(
            vae=vae,
            condition_encoder=condition_encoder,
            refiner=refiner,
            lr_image=lr_image,
            domain_id_value=domain_id_value,
            scale=scale,
            tile_lr_size=tile_lr_size,
            overlap_lr=int(args.tile_overlap),
            tile_batch_size=int(args.tile_batch_size),
            dtype_name=dtype_name,
            device=device,
            residual_strength=residual_strength,
        )
        bicubic_image = lr_image.resize(refined_image.size, Image.Resampling.BICUBIC)
        bicubic_image.save(args.output_dir / "bicubic.png")
        condition_image.save(args.output_dir / "condition.png")
        refined_image.save(args.output_dir / "refined.png")
        lr_nearest = lr_image.resize(refined_image.size, Image.Resampling.NEAREST)
        make_grid(
            [("LR", lr_nearest), ("bicubic", bicubic_image), ("condition", condition_image), ("refined", refined_image)],
            args.output_dir / "grid_lr_bicubic_condition_refined.png",
        )
        print(f"saved {args.output_dir}", flush=True)
        return

    lr_image, gt_image = prepare_inputs(args, config)
    lr_image.save(args.output_dir / "input_lr.png")
    if gt_image is not None:
        gt_image.save(args.output_dir / "gt_hr.png")
    lr_tensor = pil_to_tensor(lr_image).unsqueeze(0).to(device)
    domain_id = torch.full((1,), domain_id_value, device=device, dtype=torch.long)
    decoded_condition, decoded_refined = refine_batch(
        vae=vae,
        condition_encoder=condition_encoder,
        refiner=refiner,
        lr=lr_tensor,
        domain_id=domain_id,
        dtype_name=dtype_name,
        residual_strength=residual_strength,
    )
    condition_image = tensor_to_pil(decoded_condition[0])
    refined_image = tensor_to_pil(decoded_refined[0])
    bicubic = F.interpolate(lr_tensor.float(), size=decoded_refined.shape[-2:], mode="bicubic", align_corners=False).clamp(
        0.0,
        1.0,
    )
    bicubic_image = tensor_to_pil(bicubic[0])
    lr_nearest = lr_image.resize(refined_image.size, Image.Resampling.NEAREST)
    bicubic_image.save(args.output_dir / "bicubic.png")
    condition_image.save(args.output_dir / "condition.png")
    refined_image.save(args.output_dir / "refined.png")
    grid_items = [("LR", lr_nearest), ("bicubic", bicubic_image), ("condition", condition_image), ("refined", refined_image)]
    if gt_image is not None:
        grid_items.append(("GT", gt_image))
    make_grid(grid_items, args.output_dir / "grid_lr_bicubic_condition_refined_gt.png")
    print(f"saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
