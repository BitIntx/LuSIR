from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything
from tools.train.train_detail_branch import (
    TeacherImageCache,
    apply_detail_mask_policy,
    load_autoencoder,
    load_condition_encoder,
    load_detail_mask_predictor,
    local_highpass_error,
    make_base_prediction,
    make_dataset,
    make_grid,
    teacher_improvement_mask,
    tensor_to_pil,
)
from tools.train.train_residual_refiner import laplacian_response, metric_highpass, ssim_per_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose RealESRGAN teacher patch quality against GT and LuSIR base.")
    parser.add_argument("--config", type=Path, default=Path("configs/detail_branch_v7_teacher_filtered_hinge_probe.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/scratch/sr-diffusion/runs/diagnose_teacher_patch_quality_v7_train256"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--highpass-kernel", type=int, default=None)
    parser.add_argument("--patch-kernel", type=int, default=None)
    parser.add_argument("--teacher-ratio", type=float, default=None)
    parser.add_argument("--teacher-margin", type=float, default=None)
    parser.add_argument("--teacher-min-base-error", type=float, default=None)
    parser.add_argument("--teacher-min-residual", type=float, default=None)
    return parser.parse_args()


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def mean_abs_per_image(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().abs().flatten(1).mean(dim=1)


def ratio_per_image(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator.float() / denominator.float().clamp_min(1e-12)


def weighted_mean_per_image(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-2:] != values.shape[-2:]:
        weight = F.interpolate(weight.float(), size=values.shape[-2:], mode="bilinear", align_corners=False)
    if values.shape[1] != 1:
        values = values.float().mean(dim=1, keepdim=True)
    numerator = (values.float() * weight.float()).flatten(1).sum(dim=1)
    denominator = weight.float().flatten(1).sum(dim=1).clamp_min(1e-8)
    return numerator / denominator


def heatmap(score: torch.Tensor) -> Image.Image:
    if score.ndim == 3:
        score = score[:1]
    array = score.detach().float().squeeze(0).cpu().clamp(0.0, 1.0).numpy()
    red = array
    green = np.sqrt(array) * 0.65
    blue = (1.0 - array) * 0.35
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def signed_tensor_to_pil(tensor: torch.Tensor, scale: float = 0.12) -> Image.Image:
    vis = tensor.detach().float().cpu() / max(float(scale), 1e-8) * 0.5 + 0.5
    return tensor_to_pil(vis.clamp(0.0, 1.0))


def collect(totals: dict[str, list[float]], name: str, values: torch.Tensor) -> None:
    totals.setdefault(name, []).extend(float(value) for value in values.detach().float().cpu())


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def add_candidate_metrics(
    totals: dict[str, list[float]],
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    highpass_kernel: int,
) -> None:
    prediction_high = metric_highpass(prediction, highpass_kernel)
    target_high = metric_highpass(target, highpass_kernel)
    prediction_lap = laplacian_response(prediction)
    target_lap = laplacian_response(target)
    collect(totals, f"{prefix}_psnr", psnr_per_image(prediction, target))
    collect(totals, f"{prefix}_ssim", ssim_per_image(prediction, target))
    collect(totals, f"{prefix}_highpass_l1", mean_abs_per_image(prediction_high - target_high))
    collect(totals, f"{prefix}_laplacian_l1", mean_abs_per_image(prediction_lap - target_lap))
    collect(
        totals,
        f"{prefix}_highpass_ratio",
        ratio_per_image(mean_abs_per_image(prediction_high), mean_abs_per_image(target_high)),
    )
    collect(
        totals,
        f"{prefix}_laplacian_ratio",
        ratio_per_image(mean_abs_per_image(prediction_lap), mean_abs_per_image(target_lap)),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    loss_cfg = config.get("loss", {})
    highpass_kernel = int(args.highpass_kernel or loss_cfg.get("highpass_kernel", 15))
    patch_kernel = int(args.patch_kernel or loss_cfg.get("teacher_filter_kernel", loss_cfg.get("lowpass_kernel", 31)))
    teacher_ratio = float(args.teacher_ratio if args.teacher_ratio is not None else loss_cfg.get("teacher_filter_ratio", 1.0))
    teacher_margin = float(
        args.teacher_margin if args.teacher_margin is not None else loss_cfg.get("teacher_filter_margin", 0.0)
    )
    min_base_error = float(
        args.teacher_min_base_error
        if args.teacher_min_base_error is not None
        else loss_cfg.get("teacher_filter_min_base_error", 0.0)
    )
    min_residual = float(
        args.teacher_min_residual
        if args.teacher_min_residual is not None
        else loss_cfg.get("teacher_filter_min_residual", 0.0)
    )
    detail_mask_cfg = config.get("detail_mask", {})
    detail_mask_floor = float(detail_mask_cfg.get("floor", 0.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(config, split=args.split, seed=args.seed, deterministic=True)
    if args.limit > 0 and args.limit < len(dataset):
        dataset = Subset(dataset, list(range(args.limit)))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    detail_mask_predictor = load_detail_mask_predictor(config, device)
    teacher_cache = TeacherImageCache(config)

    totals: dict[str, list[float]] = {}
    grid_rows: list[list[tuple[str, Image.Image]]] = []
    processed = 0

    with torch.no_grad():
        for batch in dataloader:
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            condition, base_sr, bicubic = make_base_prediction(
                vae=vae,
                condition_encoder=condition_encoder,
                hr=hr,
                lr=lr,
                domain_id=domain_id,
                device=device,
                dtype_name=dtype_name,
            )
            teacher_sr = teacher_cache.load(batch, hr=hr, device=device)
            if teacher_sr is None:
                raise RuntimeError("Teacher cache returned None; this diagnostic requires teacher images.")
            with autocast_context(device, dtype_name):
                detail_mask = (
                    detail_mask_predictor(base_sr, bicubic, condition, domain_id)
                    if detail_mask_predictor is not None
                    else None
                )
            detail_mask = apply_detail_mask_policy(detail_mask, detail_mask_cfg)
            if detail_mask is None:
                detail_mask = torch.ones_like(base_sr[:, :1])

            selected_mask, selected_stats = teacher_improvement_mask(
                teacher_sr=teacher_sr,
                base_sr=base_sr,
                hr=hr,
                detail_mask=None,
                highpass_kernel=highpass_kernel,
                patch_kernel=patch_kernel,
                ratio=teacher_ratio,
                margin=teacher_margin,
                min_base_error=min_base_error,
                min_teacher_residual=min_residual,
                mask_floor=0.0,
            )
            effective_mask, effective_stats = teacher_improvement_mask(
                teacher_sr=teacher_sr,
                base_sr=base_sr,
                hr=hr,
                detail_mask=detail_mask,
                highpass_kernel=highpass_kernel,
                patch_kernel=patch_kernel,
                ratio=teacher_ratio,
                margin=teacher_margin,
                min_base_error=min_base_error,
                min_teacher_residual=min_residual,
                mask_floor=detail_mask_floor,
            )
            base_error_map = local_highpass_error(
                base_sr,
                hr,
                highpass_kernel=highpass_kernel,
                patch_kernel=patch_kernel,
            )
            teacher_error_map = local_highpass_error(
                teacher_sr,
                hr,
                highpass_kernel=highpass_kernel,
                patch_kernel=patch_kernel,
            )
            improvement_map = base_error_map - teacher_error_map
            hard_overlap = selected_mask * detail_mask.float()
            selected_area = selected_mask.flatten(1).mean(dim=1)
            detail_area = detail_mask.float().flatten(1).mean(dim=1)
            hard_overlap_area = hard_overlap.flatten(1).mean(dim=1)
            selected_inside_detail = hard_overlap.flatten(1).sum(dim=1) / selected_mask.flatten(1).sum(dim=1).clamp_min(1e-8)
            detail_covered_by_selected = hard_overlap.flatten(1).sum(dim=1) / detail_mask.flatten(1).sum(dim=1).clamp_min(1e-8)

            teacher_delta = teacher_sr - base_sr
            teacher_high_delta = metric_highpass(teacher_delta, highpass_kernel)
            oracle_rgb = (base_sr + teacher_delta * effective_mask).clamp(0.0, 1.0)
            oracle_highpass = (base_sr + teacher_high_delta * effective_mask).clamp(0.0, 1.0)
            oracle_highpass_unmasked = (base_sr + teacher_high_delta * selected_mask).clamp(0.0, 1.0)

            add_candidate_metrics(totals, "base", base_sr, hr, highpass_kernel=highpass_kernel)
            add_candidate_metrics(totals, "teacher", teacher_sr, hr, highpass_kernel=highpass_kernel)
            add_candidate_metrics(totals, "oracle_rgb_effective", oracle_rgb, hr, highpass_kernel=highpass_kernel)
            add_candidate_metrics(totals, "oracle_highpass_effective", oracle_highpass, hr, highpass_kernel=highpass_kernel)
            add_candidate_metrics(totals, "oracle_highpass_unmasked", oracle_highpass_unmasked, hr, highpass_kernel=highpass_kernel)

            base_psnr = psnr_per_image(base_sr, hr)
            teacher_psnr = psnr_per_image(teacher_sr, hr)
            base_high_l1 = mean_abs_per_image(metric_highpass(base_sr, highpass_kernel) - metric_highpass(hr, highpass_kernel))
            teacher_high_l1 = mean_abs_per_image(
                metric_highpass(teacher_sr, highpass_kernel) - metric_highpass(hr, highpass_kernel)
            )
            collect(totals, "teacher_minus_base_psnr", teacher_psnr - base_psnr)
            collect(totals, "teacher_psnr_wins", (teacher_psnr > base_psnr).float())
            collect(totals, "teacher_highpass_l1_wins", (teacher_high_l1 < base_high_l1).float())
            collect(totals, "teacher_selected_area", selected_area)
            collect(totals, "detail_mask_area", detail_area)
            collect(totals, "teacher_detail_hard_overlap_area", hard_overlap_area)
            collect(totals, "teacher_selected_inside_detail", selected_inside_detail)
            collect(totals, "detail_covered_by_teacher_selected", detail_covered_by_selected)
            collect(totals, "teacher_effective_weight", effective_mask.flatten(1).mean(dim=1))
            collect(totals, "teacher_global_improvement", improvement_map.flatten(1).mean(dim=1))
            collect(totals, "teacher_selected_improvement", weighted_mean_per_image(improvement_map, selected_mask))
            collect(totals, "teacher_effective_improvement", weighted_mean_per_image(improvement_map, effective_mask))
            collect(totals, "teacher_stats_base_error", torch.full_like(base_psnr, float(selected_stats["base_error"].detach().cpu())))
            collect(totals, "teacher_stats_error", torch.full_like(base_psnr, float(selected_stats["teacher_error"].detach().cpu())))
            collect(totals, "teacher_stats_residual_energy", torch.full_like(base_psnr, float(selected_stats["teacher_residual_energy"].detach().cpu())))
            collect(totals, "teacher_stats_selected", torch.full_like(base_psnr, float(selected_stats["selected"].detach().cpu())))
            collect(totals, "teacher_stats_effective_weight", torch.full_like(base_psnr, float(effective_stats["weight"].detach().cpu())))

            lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)
            for item_idx in range(hr.shape[0]):
                if len(grid_rows) >= args.sample_count:
                    break
                row = [
                    ("LR", tensor_to_pil(lr_nearest[item_idx])),
                    ("base", tensor_to_pil(base_sr[item_idx])),
                    ("teacher", tensor_to_pil(teacher_sr[item_idx])),
                    ("oracle HP", tensor_to_pil(oracle_highpass[item_idx])),
                    ("oracle HP raw", tensor_to_pil(oracle_highpass_unmasked[item_idx])),
                    ("GT", tensor_to_pil(hr[item_idx])),
                    ("teacher win", heatmap(selected_mask[item_idx])),
                    ("detail mask", heatmap(detail_mask[item_idx])),
                    ("effective", heatmap(effective_mask[item_idx] / max(effective_mask[item_idx].max().item(), 1e-8))),
                    ("teacher HP delta", signed_tensor_to_pil(teacher_high_delta[item_idx])),
                ]
                grid_rows.append(row)

            processed += int(hr.shape[0])
            print(f"processed={processed}/{len(dataset)}", flush=True)

    summary: dict[str, Any] = {
        "config": str(args.config),
        "split": args.split,
        "limit": args.limit,
        "num_images": processed,
        "device": str(device),
        "dtype": dtype_name,
        "teacher_cache": str(config.get("teacher", {}).get("cache_dir")),
        "detail_mask": {
            "top_fraction": detail_mask_cfg.get("top_fraction"),
            "top_mode": detail_mask_cfg.get("top_mode"),
            "floor": detail_mask_floor,
        },
        "filter": {
            "highpass_kernel": highpass_kernel,
            "patch_kernel": patch_kernel,
            "ratio": teacher_ratio,
            "margin": teacher_margin,
            "min_base_error": min_base_error,
            "min_teacher_residual": min_residual,
        },
        "metrics": {name: summarize(values) for name, values in sorted(totals.items())},
        "interpretation": {
            "teacher_selected_area": "unmasked local area where teacher highpass is near/better than base against GT",
            "teacher_effective_weight": "teacher-selected area after multiplying by learned detail mask and mask floor",
            "oracle_highpass_effective": "base plus teacher highpass residual only under the v7 effective mask",
            "oracle_highpass_unmasked": "base plus teacher highpass residual under teacher-selected patches without learned-mask gating",
            "teacher_selected_improvement": "mean local highpass-error improvement inside teacher-selected patches; positive is useful",
        },
    }
    summary_path = args.output_dir / "summary.json"
    grid_path = args.output_dir / "teacher_patch_quality_grid.png"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if grid_rows:
        make_grid(grid_rows, grid_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(f"wrote_grid={grid_path}")


if __name__ == "__main__":
    main()
