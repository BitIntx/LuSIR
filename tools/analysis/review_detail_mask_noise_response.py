from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.detail_mask import DetailMaskPredictor, detail_need_components, top_fraction_mask
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything
from tools.train.train_detail_branch import (
    load_autoencoder,
    load_condition_encoder,
    make_base_prediction,
    make_dataset,
    make_grid,
    tensor_to_pil,
)
from tools.train.train_detail_mask_predictor import heatmap, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review whether the learned detail mask reacts to injected noise.")
    parser.add_argument("--config", type=Path, default=Path("configs/detail_mask_predictor_v1_probe.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/detail_mask_predictor_v1_best3250.pt"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/scratch/sr-diffusion/runs/detail_mask_noise_response_review"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--noise-sigma", type=float, default=0.10)
    parser.add_argument("--noise-patch-size", type=int, default=96)
    parser.add_argument("--highpass-kernel", type=int, default=15)
    parser.add_argument("--patch-kernel", type=int, default=9)
    parser.add_argument("--score-quantile", type=float, default=0.95)
    return parser.parse_args()


def choose_low_texture_patch(target: torch.Tensor, patch_size: int, stride: int = 32) -> tuple[int, int]:
    _, height, width = target.shape
    patch_size = min(int(patch_size), height, width)
    gray = target.float().mean(dim=0, keepdim=True).unsqueeze(0)
    blur = F.avg_pool2d(F.pad(gray, (7, 7, 7, 7), mode="reflect"), kernel_size=15, stride=1)
    energy = (gray - blur).abs()
    best_score = float("inf")
    best_xy = (0, 0)
    for y in range(0, max(height - patch_size + 1, 1), stride):
        for x in range(0, max(width - patch_size + 1, 1), stride):
            score = float(energy[..., y : y + patch_size, x : x + patch_size].mean().cpu())
            if score < best_score:
                best_score = score
                best_xy = (x, y)
    return best_xy


def inject_noise(
    base: torch.Tensor,
    target: torch.Tensor,
    *,
    patch_size: int,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    noisy = base.clone()
    noise_mask = torch.zeros_like(base[:, :1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for item_idx in range(base.shape[0]):
        x, y = choose_low_texture_patch(target[item_idx], patch_size=patch_size)
        patch = noisy[item_idx : item_idx + 1, :, y : y + patch_size, x : x + patch_size]
        noise = torch.randn(patch.shape, generator=generator, device="cpu", dtype=torch.float32).to(patch.device)
        noisy[item_idx : item_idx + 1, :, y : y + patch_size, x : x + patch_size] = (
            patch.float() + float(sigma) * noise
        ).clamp(0.0, 1.0)
        noise_mask[item_idx : item_idx + 1, :, y : y + patch_size, x : x + patch_size] = 1.0
    return noisy, noise_mask


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value.float() * mask.float()).flatten(1).sum(dim=1) / mask.float().flatten(1).sum(dim=1).clamp_min(1.0)


def outside_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inverse = 1.0 - mask.float()
    return (value.float() * inverse).flatten(1).sum(dim=1) / inverse.flatten(1).sum(dim=1).clamp_min(1.0)


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.top_fraction) <= 1.0:
        raise ValueError(f"--top-fraction must be in (0, 1], got {args.top_fraction}")
    config = load_config(args.config)
    seed_everything(args.seed)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(config, split=args.split, seed=args.seed, deterministic=True)
    if 0 < args.limit < len(dataset):
        dataset = Subset(dataset, list(range(args.limit)))
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    predictor = DetailMaskPredictor.from_config(config["model"]).to(device)
    checkpoint_step = load_checkpoint(args.checkpoint, predictor, None, device)
    predictor.eval()

    totals: dict[str, float] = {
        "clean_top_noise_region_mean": 0.0,
        "noisy_top_noise_region_mean": 0.0,
        "noisy_prediction_noise_region_mean": 0.0,
        "noisy_prediction_outside_mean": 0.0,
        "noisy_target_noise_region_mean": 0.0,
        "noisy_excess_noise_region_mean": 0.0,
        "clean_noisy_top_iou": 0.0,
    }
    count = 0
    grid_rows: list[list[tuple[str, Any]]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            condition, base, bicubic = make_base_prediction(vae, condition_encoder, hr, lr, domain_id, device, dtype_name)
            noisy_base, noise_mask = inject_noise(
                base,
                hr,
                patch_size=int(args.noise_patch_size),
                sigma=float(args.noise_sigma),
                seed=int(args.seed) + batch_index,
            )
            with autocast_context(device, dtype_name):
                clean_prediction = predictor(base, bicubic, condition, domain_id).float()
                noisy_prediction = predictor(noisy_base, bicubic, condition, domain_id).float()
            clean_top = top_fraction_mask(clean_prediction, float(args.top_fraction))
            noisy_top = top_fraction_mask(noisy_prediction, float(args.top_fraction))
            noisy_components = detail_need_components(
                noisy_base,
                hr,
                highpass_kernel=int(args.highpass_kernel),
                patch_kernel=int(args.patch_kernel),
                score_quantile=float(args.score_quantile),
            )
            noisy_excess = noisy_components["excess"] / noisy_components["excess"].flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
            intersection = (clean_top * noisy_top).flatten(1).sum(dim=1)
            union = ((clean_top + noisy_top) > 0).float().flatten(1).sum(dim=1).clamp_min(1.0)

            batch_size = int(hr.shape[0])
            totals["clean_top_noise_region_mean"] += float(masked_mean(clean_top, noise_mask).sum().cpu())
            totals["noisy_top_noise_region_mean"] += float(masked_mean(noisy_top, noise_mask).sum().cpu())
            totals["noisy_prediction_noise_region_mean"] += float(masked_mean(noisy_prediction, noise_mask).sum().cpu())
            totals["noisy_prediction_outside_mean"] += float(outside_mean(noisy_prediction, noise_mask).sum().cpu())
            totals["noisy_target_noise_region_mean"] += float(masked_mean(noisy_components["score"], noise_mask).sum().cpu())
            totals["noisy_excess_noise_region_mean"] += float(masked_mean(noisy_excess, noise_mask).sum().cpu())
            totals["clean_noisy_top_iou"] += float((intersection / union).sum().cpu())
            count += batch_size

            lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest")
            for item_idx in range(batch_size):
                if len(grid_rows) >= int(args.sample_count):
                    break
                grid_rows.append(
                    [
                        ("LR", tensor_to_pil(lr_nearest[item_idx])),
                        ("base", tensor_to_pil(base[item_idx])),
                        ("noisy base", tensor_to_pil(noisy_base[item_idx])),
                        ("GT", tensor_to_pil(hr[item_idx])),
                        ("noise region", heatmap(noise_mask[item_idx])),
                        ("noisy target", heatmap(noisy_components["score"][item_idx])),
                        ("noisy excess", heatmap(noisy_excess[item_idx])),
                        ("pred clean", heatmap(clean_prediction[item_idx])),
                        (f"top {args.top_fraction:.0%} clean", heatmap(clean_top[item_idx])),
                        ("pred noisy", heatmap(noisy_prediction[item_idx])),
                        (f"top {args.top_fraction:.0%} noisy", heatmap(noisy_top[item_idx])),
                    ]
                )

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "config": str(args.config),
        "num_images": int(count),
        "noise_patch_size": int(args.noise_patch_size),
        "noise_sigma": float(args.noise_sigma),
        "top_fraction": float(args.top_fraction),
        "interpretation": {
            "clean_top_noise_region_mean": "top-mask coverage inside the future noise patch before noise injection; random baseline is top_fraction",
            "noisy_top_noise_region_mean": "top-mask coverage inside the injected noise patch after noise injection; much above top_fraction means the mask reacts to noise",
            "noisy_prediction_noise_region_mean": "soft prediction mean inside the injected noise patch",
            "noisy_prediction_outside_mean": "soft prediction mean outside the injected noise patch",
            "noisy_target_noise_region_mean": "GT-supervised missing-detail target inside the injected noise patch; should remain low if noise is excess detail",
            "noisy_excess_noise_region_mean": "normalized excess-detail energy inside the injected noise patch",
            "clean_noisy_top_iou": "IoU between clean and noisy top masks",
        },
    }
    summary["metrics"] = {name: value / max(1, count) for name, value in totals.items()}
    summary_path = args.output_dir / "summary.json"
    grid_path = args.output_dir / "detail_mask_noise_response_grid.png"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_grid(grid_rows, grid_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(f"wrote_grid={grid_path}")


if __name__ == "__main__":
    main()
