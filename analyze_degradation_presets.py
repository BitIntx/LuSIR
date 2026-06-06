from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LR degradation preset severity on fixed samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--presets", nargs="+", default=["clean", "mild", "photo_v2", "photo_v3_noise_mix", "photo_detail_mix"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--grid-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def make_dataset(config: dict[str, Any], preset: str, split: str, seed: int) -> ManifestImageDataset:
    data = config["data"]
    return ManifestImageDataset(
        manifest_path=data["manifest"],
        split=split,
        hr_size=data.get("hr_size", 512),
        scale=data.get("scale", 4),
        domains=data.get("domains", {"photo": 0, "anime": 1}),
        degradation_preset=preset,
        seed=seed,
        deterministic=True,
    )


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(F.mse_loss(a.float(), b.float()))
    return -10.0 * math.log10(max(mse, 1e-12))


def total_variation(x: torch.Tensor) -> float:
    x = x.float()
    dx = (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
    dy = (x[:, 1:, :] - x[:, :-1, :]).abs().mean()
    return float(dx + dy)


def chroma_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    matrix = torch.tensor(
        [[-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
        dtype=torch.float32,
    )
    a_chroma = torch.einsum("kc,chw->khw", matrix, a.float())
    b_chroma = torch.einsum("kc,chw->khw", matrix, b.float())
    return float(torch.sqrt(F.mse_loss(a_chroma, b_chroma)))


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    array = x.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")


def labeled(image: Image.Image, label: str, label_height: int = 24) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    ImageDraw.Draw(canvas).text((5, 5), label, fill="black")
    return canvas


def make_grid(rows: list[list[Image.Image]]) -> Image.Image:
    widths = [max(row[col].width for row in rows) for col in range(len(rows[0]))]
    heights = [max(image.height for image in row) for row in rows]
    grid = Image.new("RGB", (sum(widths), sum(heights)), "white")
    top = 0
    for row, height in zip(rows, heights, strict=True):
        left = 0
        for image, width in zip(row, widths, strict=True):
            grid.paste(image, (left, top))
            left += width
        top += height
    return grid


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    presets = list(dict.fromkeys(args.presets))
    datasets = {preset: make_dataset(config, preset, args.split, args.seed) for preset in set(presets + ["clean"])}
    limit = min(args.limit, *(len(dataset) for dataset in datasets.values()))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    grid_rows: list[list[Image.Image]] = []
    for index in range(limit):
        clean_item = datasets["clean"][index]
        clean_lr = clean_item["lr"]
        hr = clean_item["hr"]
        clean_tv = total_variation(clean_lr)
        grid_row = [labeled(tensor_to_pil(hr), "GT")]
        for preset in presets:
            item = datasets[preset][index]
            lr = item["lr"]
            upsampled = F.interpolate(
                lr.unsqueeze(0),
                size=hr.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )[0].clamp(0.0, 1.0)
            rows.append(
                {
                    "index": index,
                    "path": item["path"],
                    "preset": preset,
                    "bicubic_psnr_vs_hr": psnr(upsampled, hr),
                    "lr_psnr_vs_clean": psnr(lr, clean_lr),
                    "lr_chroma_rms_vs_clean": chroma_rms(lr, clean_lr),
                    "lr_tv_ratio_vs_clean": total_variation(lr) / max(clean_tv, 1e-12),
                }
            )
            if index < args.grid_count:
                grid_row.append(labeled(tensor_to_pil(upsampled), preset))
        if index < args.grid_count:
            grid_rows.append(grid_row)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {"split": args.split, "limit": limit, "seed": args.seed, "presets": {}}
    for preset in presets:
        preset_rows = [row for row in rows if row["preset"] == preset]
        summary["presets"][preset] = {
            key: summarize([float(row[key]) for row in preset_rows])
            for key in ("bicubic_psnr_vs_hr", "lr_psnr_vs_clean", "lr_chroma_rms_vs_clean", "lr_tv_ratio_vs_clean")
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if grid_rows:
        make_grid(grid_rows).save(output_dir / "grid_gt_and_degradations.png")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
