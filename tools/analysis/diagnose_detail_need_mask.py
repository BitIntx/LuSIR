from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.detail_mask import (
    detail_need_components,
    masked_highpass_oracle,
    observable_detail_proxies,
    top_fraction_mask,
)
from sr_diffusion.utils import get_device, load_config, seed_everything
from tools.train.train_detail_branch import (
    load_autoencoder,
    load_condition_encoder,
    make_base_prediction,
    make_dataset,
    make_grid,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a GT-supervised detail-need target and observable proxies.")
    parser.add_argument("--config", type=Path, default=Path("configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/scratch/sr-diffusion/runs/diagnose_detail_need_mask_val100"),
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
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def pearson_per_image(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    first_centered = first_flat - first_flat.mean(dim=1, keepdim=True)
    second_centered = second_flat - second_flat.mean(dim=1, keepdim=True)
    numerator = (first_centered * second_centered).mean(dim=1)
    denominator = first_centered.square().mean(dim=1).sqrt() * second_centered.square().mean(dim=1).sqrt()
    return numerator / denominator.clamp_min(1e-12)


def selected_capture(energy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (energy.float() * mask.float()).flatten(1).sum(dim=1) / energy.float().flatten(1).sum(dim=1).clamp_min(1e-12)


def selected_concentration(energy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected_mean = (energy.float() * mask.float()).flatten(1).sum(dim=1) / mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
    return selected_mean / energy.float().flatten(1).mean(dim=1).clamp_min(1e-12)


def heatmap(score: torch.Tensor) -> Image.Image:
    array = score.detach().float().squeeze(0).cpu().clamp(0.0, 1.0).numpy()
    red = array
    green = np.sqrt(array) * 0.65
    blue = (1.0 - array) * 0.35
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


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
    condition_encoder = load_condition_encoder(config, device)

    totals: dict[str, list[float]] = {}
    grid_rows: list[list[tuple[str, Image.Image]]] = []
    processed = 0

    def collect(name: str, values: torch.Tensor) -> None:
        totals.setdefault(name, []).extend(float(value) for value in values.detach().float().cpu())

    with torch.no_grad():
        for batch in dataloader:
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            _, base, bicubic = make_base_prediction(vae, condition_encoder, hr, lr, domain_id, device, dtype_name)
            components = detail_need_components(
                base,
                hr,
                highpass_kernel=args.highpass_kernel,
                patch_kernel=args.patch_kernel,
                score_quantile=args.score_quantile,
            )
            proxies = observable_detail_proxies(
                base,
                bicubic,
                highpass_kernel=args.highpass_kernel,
                patch_kernel=args.patch_kernel,
                score_quantile=args.score_quantile,
            )
            base_psnr = psnr_per_image(base, hr)
            collect("base_psnr", base_psnr)
            collect("missing_energy", components["missing"].flatten(1).mean(dim=1))
            collect("excess_energy", components["excess"].flatten(1).mean(dim=1))
            for name, proxy in proxies.items():
                collect(f"proxy_{name}_corr", pearson_per_image(proxy, components["score"]))
                for fraction in args.fractions:
                    proxy_mask = top_fraction_mask(proxy, fraction)
                    collect(f"proxy_{name}_top{fraction:.2f}_missing_capture", selected_capture(components["missing"], proxy_mask))
                    collect(f"proxy_{name}_top{fraction:.2f}_missing_concentration", selected_concentration(components["missing"], proxy_mask))
                    collect(f"proxy_{name}_top{fraction:.2f}_excess_capture", selected_capture(components["excess"], proxy_mask))

            fraction_masks: dict[float, torch.Tensor] = {}
            for fraction in args.fractions:
                mask = top_fraction_mask(components["score"], fraction)
                fraction_masks[fraction] = mask
                oracle = masked_highpass_oracle(base, hr, mask, highpass_kernel=args.highpass_kernel)
                collect(f"top{fraction:.2f}_missing_capture", selected_capture(components["missing"], mask))
                collect(f"top{fraction:.2f}_missing_concentration", selected_concentration(components["missing"], mask))
                collect(f"top{fraction:.2f}_excess_capture", selected_capture(components["excess"], mask))
                collect(f"top{fraction:.2f}_oracle_psnr_gain", psnr_per_image(oracle, hr) - base_psnr)

            lr_nearest = torch.nn.functional.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest")
            vis_fraction = min(args.fractions, key=lambda value: abs(value - 0.2))
            vis_mask = fraction_masks[vis_fraction]
            vis_oracle = masked_highpass_oracle(base, hr, vis_mask, highpass_kernel=args.highpass_kernel)
            for item_idx in range(hr.shape[0]):
                if len(grid_rows) >= args.sample_count:
                    break
                grid_rows.append(
                    [
                        ("LR", tensor_to_pil(lr_nearest[item_idx])),
                        ("bicubic", tensor_to_pil(bicubic[item_idx])),
                        ("base", tensor_to_pil(base[item_idx])),
                        ("GT", tensor_to_pil(hr[item_idx])),
                        ("detail-need target", heatmap(components["score"][item_idx])),
                        (f"top {vis_fraction:.0%} mask", heatmap(vis_mask[item_idx])),
                        ("observable gap proxy", heatmap(proxies["base_bicubic_gap"][item_idx])),
                        ("observable HP proxy", heatmap(proxies["highpass_disagreement"][item_idx])),
                        (f"oracle top {vis_fraction:.0%}", tensor_to_pil(vis_oracle[item_idx])),
                    ]
                )
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
        "interpretation": {
            "missing_capture": "fraction of all missing-detail energy contained in selected pixels; higher is better",
            "missing_concentration": "selected missing-detail density divided by image average; >1 means useful concentration",
            "excess_capture": "fraction of excessive base-detail energy selected; lower is better",
            "oracle_psnr_gain": "PSNR gain from applying the true highpass correction only inside the mask",
            "proxy_corr": "pixelwise Pearson correlation between an inference-time proxy and the GT-supervised target",
            "proxy_top_missing_capture": "missing-detail capture when the proxy, rather than the GT target, selects pixels",
        },
    }
    summary["metrics"] = {name: mean(values) for name, values in sorted(totals.items())}
    summary_path = args.output_dir / "summary.json"
    grid_path = args.output_dir / "detail_need_mask_grid.png"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_grid(grid_rows, grid_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(f"wrote_grid={grid_path}")


if __name__ == "__main__":
    main()
