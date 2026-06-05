from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from infer_diffusion import load_autoencoder, load_condition_encoder, tensor_to_pil
from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Stage 2 condition residuals against GT VAE latents.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/diffusion_photo100k_xl_stage4_condition_v3_gated_residual_mild_b8_probe.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/runs/diagnose_stage2_xl_residuals_mild_val100"),
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--degradation-preset", default=None)
    parser.add_argument("--lowpass-kernel", type=int, default=15)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    return 20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(max(mse, 1e-12)))


def make_dataset(config: dict[str, Any], split: str, seed: int, limit: int | None) -> ManifestImageDataset | Subset:
    data_config = config["data"]
    dataset = ManifestImageDataset(
        manifest_path=data_config["manifest"],
        split=split,
        hr_size=data_config.get("hr_size", 512),
        scale=data_config.get("scale", 4),
        domains=data_config.get("domains", {"photo": 0, "anime": 1}),
        degradation_preset=data_config.get("degradation_preset", "mild"),
        seed=seed,
        deterministic=True,
    )
    if limit is not None and limit > 0 and limit < len(dataset):
        return Subset(dataset, list(range(limit)))
    return dataset


def lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError(f"lowpass kernel must be odd, got {kernel_size}")
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def latent_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    x_abs = x.detach().float().abs()
    x_sq = x_abs.pow(2)
    flat = x_abs.flatten()
    return {
        f"{prefix}_mean_abs": float(x_abs.mean().cpu()),
        f"{prefix}_rms": float(torch.sqrt(x_sq.mean()).cpu()),
        f"{prefix}_p50_abs": float(torch.quantile(flat, 0.50).cpu()),
        f"{prefix}_p90_abs": float(torch.quantile(flat, 0.90).cpu()),
        f"{prefix}_p95_abs": float(torch.quantile(flat, 0.95).cpu()),
        f"{prefix}_p99_abs": float(torch.quantile(flat, 0.99).cpu()),
        f"{prefix}_max_abs": float(x_abs.max().cpu()),
    }


def mse_and_psnr(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    mse = float(F.mse_loss(prediction.detach().float(), target.detach().float()).cpu())
    return mse, psnr_from_mse(mse)


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def heatmap_from_tensor(x: torch.Tensor) -> Image.Image:
    score = x.detach().float().abs().mean(dim=0).cpu().numpy()
    hi = float(np.quantile(score, 0.99))
    if hi <= 1e-12:
        hi = float(score.max()) if float(score.max()) > 0 else 1.0
    score = np.clip(score / hi, 0.0, 1.0)
    red = score
    green = np.sqrt(score) * 0.45
    blue = 1.0 - score
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB").resize((512, 512), Image.Resampling.NEAREST)


def add_label(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 18
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill="black", font=font)
    return canvas


def make_grid(rows: list[list[tuple[str, Image.Image]]], output_path: Path, gap: int = 6) -> None:
    if not rows:
        return
    labeled_rows = [[add_label(image, label) for label, image in row] for row in rows]
    cell_width = max(image.width for row in labeled_rows for image in row)
    cell_height = max(image.height for row in labeled_rows for image in row)
    columns = max(len(row) for row in labeled_rows)
    width = columns * cell_width + (columns + 1) * gap
    height = len(labeled_rows) * cell_height + (len(labeled_rows) + 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    for row_index, row in enumerate(labeled_rows):
        y = gap + row_index * (cell_height + gap)
        for column_index, image in enumerate(row):
            x = gap + column_index * (cell_width + gap)
            sheet.paste(image.convert("RGB"), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if float(x_arr.std()) <= 1e-12 or float(y_arr.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.degradation_preset is not None:
        config["data"]["degradation_preset"] = args.degradation_preset
    split = str(args.split or config.get("eval", {}).get("split", "val"))
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    seed_everything(args.seed)
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(config, split=split, seed=args.seed, limit=args.limit)
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    vae = load_autoencoder(config, device, dtype_name)
    condition_encoder = load_condition_encoder(config, device, dtype_name)
    vae.eval()
    condition_encoder.eval()

    rows: list[dict[str, Any]] = []
    grid_rows: list[list[tuple[str, Image.Image]]] = []
    totals: dict[str, list[float]] = {}
    global_index = 0
    kernel = int(args.lowpass_kernel)

    def collect(key: str, value: float) -> None:
        totals.setdefault(key, []).append(float(value))

    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            target = normalize_image(hr)
            lr_input = normalize_image(lr)
            with autocast_context(device, dtype_name):
                target_latent, _ = vae.encode(target)
                condition = condition_encoder(lr_input, domain_id)

            residual = target_latent - condition
            residual_low = lowpass(residual.float(), kernel).to(dtype=residual.dtype)
            residual_high = residual - residual_low

            latent_variants = {
                "condition": condition,
                "oracle_full": target_latent,
                "oracle_residual_025": condition + residual * 0.25,
                "oracle_residual_050": condition + residual * 0.50,
                "oracle_residual_075": condition + residual * 0.75,
                "oracle_lowpass": condition + residual_low,
                "oracle_highpass": condition + residual_high,
                "oracle_highpass_050": condition + residual_high * 0.50,
            }

            decoded: dict[str, torch.Tensor] = {}
            with autocast_context(device, dtype_name):
                for label, latent in latent_variants.items():
                    decoded[label] = denormalize(vae.decode(latent)).float()
            bicubic = F.interpolate(lr.float(), size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
            lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)

            for item_idx in range(hr.shape[0]):
                sample_id = global_index + item_idx
                row: dict[str, Any] = {
                    "index": sample_id,
                    "path": batch["path"][item_idx],
                    "domain": batch["domain"][item_idx],
                }
                bicubic_mse, bicubic_psnr = mse_and_psnr(bicubic[item_idx], hr[item_idx])
                row["bicubic_mse"] = bicubic_mse
                row["bicubic_psnr"] = bicubic_psnr
                for label, image_batch in decoded.items():
                    mse, psnr = mse_and_psnr(image_batch[item_idx], hr[item_idx])
                    row[f"{label}_decoded_mse"] = mse
                    row[f"{label}_decoded_psnr"] = psnr
                    collect(f"{label}_decoded_psnr", psnr)
                condition_latent_mse = float(F.mse_loss(condition[item_idx].float(), target_latent[item_idx].float()).cpu())
                row["condition_latent_mse"] = condition_latent_mse
                row.update(latent_stats(residual[item_idx], "residual"))
                row.update(latent_stats(residual_low[item_idx], "residual_lowpass"))
                row.update(latent_stats(residual_high[item_idx], "residual_highpass"))
                residual_energy = float(residual[item_idx].float().pow(2).mean().cpu())
                high_energy = float(residual_high[item_idx].float().pow(2).mean().cpu())
                low_energy = float(residual_low[item_idx].float().pow(2).mean().cpu())
                row["residual_highpass_energy_ratio"] = high_energy / max(residual_energy, 1e-12)
                row["residual_lowpass_energy_ratio"] = low_energy / max(residual_energy, 1e-12)
                row["residual_abs_gt_125_frac"] = float((residual[item_idx].abs() > 1.25).float().mean().cpu())
                row["oracle_full_gain_psnr"] = row["oracle_full_decoded_psnr"] - row["condition_decoded_psnr"]
                row["oracle_highpass_gain_psnr"] = row["oracle_highpass_decoded_psnr"] - row["condition_decoded_psnr"]
                row["oracle_lowpass_gain_psnr"] = row["oracle_lowpass_decoded_psnr"] - row["condition_decoded_psnr"]
                rows.append(row)
                for key, value in row.items():
                    if isinstance(value, float) and (
                        key.startswith("residual_")
                        or key.endswith("_gain_psnr")
                        or key in {"condition_latent_mse", "bicubic_psnr"}
                    ):
                        collect(key, value)

                if len(grid_rows) < int(args.sample_count):
                    grid_rows.append(
                        [
                            ("LR", Image.fromarray(tensor_to_uint8(lr_nearest[item_idx]), mode="RGB")),
                            ("bicubic", Image.fromarray(tensor_to_uint8(bicubic[item_idx]), mode="RGB")),
                            ("condition", Image.fromarray(tensor_to_uint8(decoded["condition"][item_idx]), mode="RGB")),
                            ("oracle high", Image.fromarray(tensor_to_uint8(decoded["oracle_highpass"][item_idx]), mode="RGB")),
                            ("oracle full", Image.fromarray(tensor_to_uint8(decoded["oracle_full"][item_idx]), mode="RGB")),
                            ("GT", Image.fromarray(tensor_to_uint8(hr[item_idx]), mode="RGB")),
                            ("residual abs", heatmap_from_tensor(residual[item_idx])),
                            ("highpass abs", heatmap_from_tensor(residual_high[item_idx])),
                        ]
                    )
            global_index += int(hr.shape[0])
            print(f"processed {global_index}/{len(dataset)}", flush=True)

    if not rows:
        raise RuntimeError("No rows evaluated")

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "config": str(args.config),
        "condition_encoder_checkpoint": str(config["condition_encoder"]["checkpoint"]),
        "autoencoder_checkpoint": str(config["autoencoder"]["checkpoint"]),
        "degradation_preset": str(config["data"].get("degradation_preset", "mild")),
        "split": split,
        "limit": len(rows),
        "device": str(device),
        "dtype": dtype_name,
        "lowpass_kernel": kernel,
        "metrics_csv": str(metrics_path),
        "sample_grid": str(args.output_dir / "residual_diagnostic_grid.png"),
    }
    for key, values in sorted(totals.items()):
        summary[f"mean_{key}"] = mean(values)

    summary["corr_residual_rms_oracle_full_gain"] = safe_corr(
        [float(row["residual_rms"]) for row in rows],
        [float(row["oracle_full_gain_psnr"]) for row in rows],
    )
    summary["corr_highpass_ratio_highpass_gain"] = safe_corr(
        [float(row["residual_highpass_energy_ratio"]) for row in rows],
        [float(row["oracle_highpass_gain_psnr"]) for row in rows],
    )
    summary["condition_vs_bicubic_delta_psnr"] = (
        float(summary["mean_condition_decoded_psnr"]) - float(summary["mean_bicubic_psnr"])
    )
    summary["oracle_full_vs_condition_delta_psnr"] = (
        float(summary["mean_oracle_full_decoded_psnr"]) - float(summary["mean_condition_decoded_psnr"])
    )
    summary["oracle_highpass_vs_condition_delta_psnr"] = (
        float(summary["mean_oracle_highpass_decoded_psnr"]) - float(summary["mean_condition_decoded_psnr"])
    )
    summary["oracle_lowpass_vs_condition_delta_psnr"] = (
        float(summary["mean_oracle_lowpass_decoded_psnr"]) - float(summary["mean_condition_decoded_psnr"])
    )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_grid(grid_rows, args.output_dir / "residual_diagnostic_grid.png")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
