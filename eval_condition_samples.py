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
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from infer_diffusion import load_autoencoder, load_condition_encoder, tensor_to_pil
from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen LR-to-latent condition encoder directly.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-count", type=int, default=8)
    parser.add_argument("--degradation-preset", default=None)
    return parser.parse_args()


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    return 20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(max(mse, 1e-12)))


def tensor_to_numpy_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    return np.round(array * 255.0).astype(np.uint8)


def make_grid(rows: list[list[Image.Image]]) -> Image.Image:
    if not rows:
        raise ValueError("Cannot build an empty grid")
    widths = [max(row[col].width for row in rows) for col in range(len(rows[0]))]
    heights = [max(image.height for image in row) for row in rows]
    grid = Image.new("RGB", (sum(widths), sum(heights)), color=(0, 0, 0))
    top = 0
    for row, height in zip(rows, heights, strict=True):
        left = 0
        for image, width in zip(row, widths, strict=True):
            grid.paste(image, (left, top))
            left += width
        top += height
    return grid


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


def save_image(path: Path, image: torch.Tensor) -> None:
    tensor_to_pil(image).save(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.degradation_preset is not None:
        config["data"]["degradation_preset"] = args.degradation_preset
    seed_everything(args.seed)
    device = get_device(args.device)
    dtype_name = config["train"].get("dtype", "bf16")
    split = args.split or str(config.get("eval", {}).get("split", "val"))
    output_dir = args.output_dir
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(config, split=split, seed=args.seed, limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    vae.eval()
    condition_encoder.eval()

    rows: list[dict[str, Any]] = []
    grid_rows: list[list[Image.Image]] = []
    global_index = 0
    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            domain_id = batch["domain_id"].to(device)
            with autocast_context(device, dtype_name):
                condition = condition_encoder(normalize_image(lr), domain_id)
                prediction = denormalize(vae.decode(condition)).float()
            bicubic = F.interpolate(lr.float(), size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
            lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)

            for item_idx in range(prediction.shape[0]):
                sample_id = global_index + item_idx
                sample_prefix = f"{sample_id:04d}"
                sample_dir = samples_dir / sample_prefix
                sample_dir.mkdir(parents=True, exist_ok=True)

                save_image(sample_dir / "lr_nearest.png", lr_nearest[item_idx])
                save_image(sample_dir / "bicubic.png", bicubic[item_idx])
                save_image(sample_dir / "condition.png", prediction[item_idx])
                save_image(sample_dir / "gt.png", hr[item_idx])

                condition_mse = float(F.mse_loss(prediction[item_idx], hr[item_idx]).detach().cpu())
                bicubic_mse = float(F.mse_loss(bicubic[item_idx], hr[item_idx]).detach().cpu())
                row = {
                    "index": sample_id,
                    "path": batch["path"][item_idx],
                    "domain": batch["domain"][item_idx],
                    "condition_mse": condition_mse,
                    "condition_psnr": psnr_from_mse(condition_mse),
                    "bicubic_mse": bicubic_mse,
                    "bicubic_psnr": psnr_from_mse(bicubic_mse),
                    "psnr_delta": psnr_from_mse(condition_mse) - psnr_from_mse(bicubic_mse),
                }
                rows.append(row)

                if len(grid_rows) < args.grid_count:
                    grid_rows.append(
                        [
                            Image.fromarray(tensor_to_numpy_uint8(lr_nearest[item_idx]), mode="RGB"),
                            Image.fromarray(tensor_to_numpy_uint8(bicubic[item_idx]), mode="RGB"),
                            Image.fromarray(tensor_to_numpy_uint8(prediction[item_idx]), mode="RGB"),
                            Image.fromarray(tensor_to_numpy_uint8(hr[item_idx]), mode="RGB"),
                        ]
                    )
            global_index += int(prediction.shape[0])
            print(f"processed {global_index}/{len(dataset)}")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_condition_psnr = float(np.mean([row["condition_psnr"] for row in rows]))
    mean_bicubic_psnr = float(np.mean([row["bicubic_psnr"] for row in rows]))
    summary = {
        "condition_encoder_checkpoint": str(config["condition_encoder"]["checkpoint"]),
        "config": str(args.config),
        "degradation_preset": str(config["data"].get("degradation_preset", "mild")),
        "split": split,
        "limit": len(rows),
        "mean_condition_psnr": mean_condition_psnr,
        "mean_bicubic_psnr": mean_bicubic_psnr,
        "mean_psnr_delta": mean_condition_psnr - mean_bicubic_psnr,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if grid_rows:
        make_grid(grid_rows).save(output_dir / "grid_lr_bicubic_condition_gt.png")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
