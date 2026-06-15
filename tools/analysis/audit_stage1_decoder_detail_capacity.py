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

from sr_diffusion.detail_mask import detail_need_components, masked_highpass_oracle, top_fraction_mask
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything
from tools.train.train_detail_branch import (
    load_autoencoder,
    load_condition_encoder,
    make_base_prediction,
    make_dataset,
    make_grid,
    tensor_to_pil,
)
from tools.train.train_residual_refiner import (
    denormalize,
    laplacian_response,
    metric_highpass,
    normalize_image,
    ssim_per_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit how much high-frequency detail Stage 1 can preserve by itself.")
    parser.add_argument("--config", type=Path, default=Path("configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/scratch/sr-diffusion/runs/audit_stage1_decoder_detail_capacity_val100"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--degradation-preset", default=None)
    parser.add_argument("--highpass-kernel", type=int, default=15)
    parser.add_argument("--patch-kernel", type=int, default=9)
    parser.add_argument("--score-quantile", type=float, default=0.95)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--skip-stage2-base", action="store_true")
    parser.add_argument("--diff-scale", type=float, default=0.08)
    parser.add_argument("--highpass-vis-scale", type=float, default=0.12)
    return parser.parse_args()


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def mean_abs_per_image(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().abs().flatten(1).mean(dim=1)


def ratio_per_image(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator.float() / denominator.float().clamp_min(1e-12)


def selected_capture(energy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (energy.float() * mask.float()).flatten(1).sum(dim=1) / energy.float().flatten(1).sum(dim=1).clamp_min(1e-12)


def heatmap(score: torch.Tensor) -> Image.Image:
    if score.ndim == 3:
        score = score[:1]
    array = score.detach().float().squeeze(0).cpu().clamp(0.0, 1.0).numpy()
    red = array
    green = np.sqrt(array) * 0.65
    blue = (1.0 - array) * 0.35
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def diff_heatmap(prediction: torch.Tensor, target: torch.Tensor, scale: float) -> Image.Image:
    diff = (prediction.float() - target.float()).abs().mean(dim=0, keepdim=True) / max(float(scale), 1e-8)
    return heatmap(diff)


def signed_tensor_to_pil(tensor: torch.Tensor, scale: float) -> Image.Image:
    vis = tensor.detach().float().cpu() / max(float(scale), 1e-8) * 0.5 + 0.5
    return tensor_to_pil(vis.clamp(0.0, 1.0))


def collect_metric(totals: dict[str, list[float]], name: str, values: torch.Tensor) -> None:
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


def add_reconstruction_metrics(
    totals: dict[str, list[float]],
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    highpass_kernel: int,
    patch_kernel: int,
    score_quantile: float,
    fractions: list[float],
) -> dict[str, torch.Tensor]:
    high_prediction = metric_highpass(prediction, kernel_size=highpass_kernel)
    high_target = metric_highpass(target, kernel_size=highpass_kernel)
    lap_prediction = laplacian_response(prediction)
    lap_target = laplacian_response(target)
    prediction_high_energy = mean_abs_per_image(high_prediction)
    target_high_energy = mean_abs_per_image(high_target)
    prediction_lap_energy = mean_abs_per_image(lap_prediction)
    target_lap_energy = mean_abs_per_image(lap_target)

    collect_metric(totals, f"{prefix}_psnr", psnr_per_image(prediction, target))
    collect_metric(totals, f"{prefix}_ssim", ssim_per_image(prediction, target))
    collect_metric(totals, f"{prefix}_highpass_ratio", ratio_per_image(prediction_high_energy, target_high_energy))
    collect_metric(totals, f"{prefix}_laplacian_ratio", ratio_per_image(prediction_lap_energy, target_lap_energy))
    collect_metric(totals, f"{prefix}_highpass_l1", mean_abs_per_image(high_prediction - high_target))
    collect_metric(totals, f"{prefix}_laplacian_l1", mean_abs_per_image(lap_prediction - lap_target))

    components = detail_need_components(
        prediction,
        target,
        highpass_kernel=highpass_kernel,
        patch_kernel=patch_kernel,
        score_quantile=score_quantile,
    )
    collect_metric(totals, f"{prefix}_missing_energy", components["missing"].flatten(1).mean(dim=1))
    collect_metric(totals, f"{prefix}_excess_energy", components["excess"].flatten(1).mean(dim=1))
    collect_metric(totals, f"{prefix}_mismatch_energy", components["mismatch"].flatten(1).mean(dim=1))
    for fraction in fractions:
        mask = top_fraction_mask(components["score"], fraction)
        oracle = masked_highpass_oracle(prediction, target, mask, highpass_kernel=highpass_kernel)
        collect_metric(totals, f"{prefix}_top{fraction:.2f}_missing_capture", selected_capture(components["missing"], mask))
        collect_metric(totals, f"{prefix}_top{fraction:.2f}_excess_capture", selected_capture(components["excess"], mask))
        collect_metric(totals, f"{prefix}_top{fraction:.2f}_oracle_psnr_gain", psnr_per_image(oracle, target) - psnr_per_image(prediction, target))
    return components


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.degradation_preset is not None:
        config["data"]["degradation_preset"] = args.degradation_preset
    seed_everything(args.seed)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
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
    condition_encoder = None if args.skip_stage2_base else load_condition_encoder(config, device)

    totals: dict[str, list[float]] = {}
    grid_rows: list[list[tuple[str, Image.Image]]] = []
    processed = 0

    with torch.no_grad():
        for batch in dataloader:
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)

            with autocast_context(device, dtype_name):
                reconstruction = denormalize(vae(normalize_image(hr), sample_posterior=False).reconstruction).float()
            vae_components = add_reconstruction_metrics(
                totals,
                "vae_recon",
                reconstruction,
                hr,
                highpass_kernel=args.highpass_kernel,
                patch_kernel=args.patch_kernel,
                score_quantile=args.score_quantile,
                fractions=args.fractions,
            )

            base = None
            base_components = None
            bicubic = F.interpolate(lr.float(), size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
            if condition_encoder is not None:
                _, base, bicubic = make_base_prediction(vae, condition_encoder, hr, lr, domain_id, device, dtype_name)
                base_components = add_reconstruction_metrics(
                    totals,
                    "stage2_base",
                    base,
                    hr,
                    highpass_kernel=args.highpass_kernel,
                    patch_kernel=args.patch_kernel,
                    score_quantile=args.score_quantile,
                    fractions=args.fractions,
                )
                collect_metric(totals, "stage2_base_minus_vae_psnr", psnr_per_image(base, hr) - psnr_per_image(reconstruction, hr))
                collect_metric(
                    totals,
                    "stage2_base_minus_vae_highpass_ratio",
                    ratio_per_image(
                        mean_abs_per_image(metric_highpass(base, args.highpass_kernel)),
                        mean_abs_per_image(metric_highpass(hr, args.highpass_kernel)),
                    )
                    - ratio_per_image(
                        mean_abs_per_image(metric_highpass(reconstruction, args.highpass_kernel)),
                        mean_abs_per_image(metric_highpass(hr, args.highpass_kernel)),
                    ),
                )

            lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest")
            for item_idx in range(hr.shape[0]):
                if len(grid_rows) >= args.sample_count:
                    break
                row: list[tuple[str, Image.Image]] = [
                    ("LR", tensor_to_pil(lr_nearest[item_idx])),
                    ("bicubic", tensor_to_pil(bicubic[item_idx])),
                    ("HR", tensor_to_pil(hr[item_idx])),
                    ("VAE recon", tensor_to_pil(reconstruction[item_idx])),
                    ("VAE abs diff", diff_heatmap(reconstruction[item_idx], hr[item_idx], args.diff_scale)),
                    ("HR highpass", signed_tensor_to_pil(metric_highpass(hr[item_idx : item_idx + 1], args.highpass_kernel)[0], args.highpass_vis_scale)),
                    (
                        "VAE highpass",
                        signed_tensor_to_pil(
                            metric_highpass(reconstruction[item_idx : item_idx + 1], args.highpass_kernel)[0],
                            args.highpass_vis_scale,
                        ),
                    ),
                    ("VAE missing", heatmap(vae_components["score"][item_idx])),
                ]
                if base is not None and base_components is not None:
                    row.extend(
                        [
                            ("Stage2 base", tensor_to_pil(base[item_idx])),
                            ("Stage2 missing", heatmap(base_components["score"][item_idx])),
                        ]
                    )
                grid_rows.append(row)
            processed += int(hr.shape[0])
            print(f"processed={processed}/{len(dataset)}", flush=True)

    summary: dict[str, Any] = {
        "config": str(args.config),
        "degradation_preset": str(config["data"].get("degradation_preset")),
        "split": args.split,
        "num_images": processed,
        "device": str(device),
        "dtype": dtype_name,
        "highpass_kernel": args.highpass_kernel,
        "patch_kernel": args.patch_kernel,
        "score_quantile": args.score_quantile,
        "fractions": args.fractions,
        "stage2_base_included": condition_encoder is not None,
        "interpretation": {
            "vae_recon_*": "Stage 1 autoencoder reconstruction of the HR target; this is the decoder/detail preservation upper-bound audit.",
            "stage2_base_*": "Stage 2 condition encoder output decoded through the same Stage 1 decoder.",
            "highpass_ratio": "mean |highpass(pred)| divided by mean |highpass(HR)|; values far below 1 indicate smoothing.",
            "laplacian_ratio": "mean |laplacian(pred)| divided by mean |laplacian(HR)|; values far below 1 indicate edge/detail loss.",
            "top_missing_capture": "oracle concentration of missing-detail energy in the top-fraction target mask.",
            "oracle_psnr_gain": "PSNR gain from applying the true highpass correction only inside the selected mask.",
        },
        "metrics": {name: summarize(values) for name, values in sorted(totals.items())},
    }
    summary_path = args.output_dir / "summary.json"
    grid_path = args.output_dir / "stage1_decoder_detail_capacity_grid.png"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_grid(grid_rows, grid_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(f"wrote_grid={grid_path}")


if __name__ == "__main__":
    main()
