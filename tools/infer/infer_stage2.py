from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.models import LRToLatentPredictor
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything
from tools.infer.infer_diffusion import (
    edge_pad_image,
    float_array_to_pil,
    inference_module_dtype,
    load_autoencoder,
    pil_to_tensor,
    prepare_inputs,
    resolve_path,
    tensor_to_pil,
    tile_blend_mask,
    tile_positions,
)
from tools.infer.infer_residual_refiner import make_grid


DEFAULT_HF_CHECKPOINT = Path("checkpoints/stage2_photo130k_lsdir_dual_detail_guarded_v2_best10000.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Stage 2 condition decoding.")
    parser.add_argument("--config", type=Path, default=Path("configs/hf/latent_pretrain_photo130k_lsdir_dual_detail_guarded_v2.yaml"))
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
    parser.add_argument("--tile", action="store_true", help="Run tiled Stage 2 inference for arbitrary-size LR input.")
    parser.add_argument("--tile-overlap", type=int, default=32, help="LR-pixel overlap between 128x128 tiles.")
    parser.add_argument("--tile-batch-size", type=int, default=1, help="Number of LR tiles to process at once.")
    return parser.parse_args()


def resolve_checkpoint(config: dict, requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    inference_checkpoint = config.get("inference", {}).get("checkpoint") or config.get("checkpoint")
    if inference_checkpoint:
        candidates.append(Path(inference_checkpoint))
    candidates.append(DEFAULT_HF_CHECKPOINT)
    for candidate in candidates:
        path = resolve_path(config, candidate.expanduser())
        if path.exists():
            return path.resolve()
    formatted = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find Stage 2 checkpoint. Checked:\n{formatted}")


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def load_stage2_encoder(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
    dtype_name: str | None,
) -> tuple[LRToLatentPredictor, int]:
    encoder = LRToLatentPredictor.from_config(config["model"])
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
    with autocast_context(lr.device, dtype_name):
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


def save_outputs(
    output_dir: Path,
    lr_image: Image.Image,
    condition_image: Image.Image,
    gt_image: Image.Image | None = None,
) -> None:
    bicubic = lr_image.resize(condition_image.size, Image.Resampling.BICUBIC)
    nearest = lr_image.resize(condition_image.size, Image.Resampling.NEAREST)
    lr_image.save(output_dir / "input_lr.png")
    bicubic.save(output_dir / "bicubic.png")
    condition_image.save(output_dir / "condition.png")
    condition_image.save(output_dir / "stage2.png")
    condition_image.save(output_dir / "sr.png")
    items = [("LR", nearest), ("bicubic", bicubic), ("stage2", condition_image)]
    if gt_image is not None:
        gt_image.save(output_dir / "gt_hr.png")
        items.append(("GT", gt_image))
    make_grid(items, output_dir / "grid_lr_bicubic_stage2_gt.png")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.degradation_preset is not None:
        config["data"]["degradation_preset"] = args.degradation_preset
    if args.tile and args.input_hr:
        raise ValueError("--tile supports --input-lr only. Use --input-lr with an arbitrary-size LR image.")
    seed_everything(args.seed)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    data_config = config["data"]
    domains = data_config.get("domains", {"photo": 0, "anime": 1})
    if args.domain not in domains:
        raise ValueError(f"Unknown domain '{args.domain}'. Available: {sorted(domains)}")

    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae = load_autoencoder(config, device, dtype_name)
    condition_encoder, checkpoint_step = load_stage2_encoder(config, checkpoint_path, device, dtype_name)
    print(f"checkpoint={checkpoint_path} step={checkpoint_step} device={device}", flush=True)

    scale = int(data_config.get("scale", 4))
    domain_id_value = int(domains[args.domain])
    if args.tile:
        lr_image = Image.open(args.input_lr).convert("RGB")
        condition_image = tiled_stage2(
            vae=vae,
            condition_encoder=condition_encoder,
            lr_image=lr_image,
            domain_id_value=domain_id_value,
            scale=scale,
            tile_lr_size=int(data_config["hr_size"]) // scale,
            overlap_lr=int(args.tile_overlap),
            tile_batch_size=int(args.tile_batch_size),
            dtype_name=dtype_name,
            device=device,
        )
        save_outputs(args.output_dir, lr_image, condition_image)
        print(f"saved {args.output_dir}", flush=True)
        return

    lr_image, gt_image = prepare_inputs(args, config)
    lr_tensor = pil_to_tensor(lr_image).unsqueeze(0).to(device)
    domain_id = torch.full((1,), domain_id_value, device=device, dtype=torch.long)
    decoded = stage2_batch(
        vae=vae,
        condition_encoder=condition_encoder,
        lr=lr_tensor,
        domain_id=domain_id,
        dtype_name=dtype_name,
    )
    condition_image = tensor_to_pil(decoded[0])
    bicubic = F.interpolate(lr_tensor.float(), size=decoded.shape[-2:], mode="bicubic", align_corners=False).clamp(
        0.0,
        1.0,
    )
    # Keep bicubic numerically identical to the tensor path for fixed-size tests.
    save_outputs(args.output_dir, lr_image, condition_image, gt_image)
    tensor_to_pil(bicubic[0]).save(args.output_dir / "bicubic.png")
    print(f"saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
