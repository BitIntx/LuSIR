from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two sampled SR eval directories.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def psnr(row: dict[str, str], key: str = "sr_psnr") -> float:
    return float(row[key])


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def add_label(image: Image.Image, label: str, height: int = 28) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), color=(18, 18, 18))
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(235, 235, 235))
    return canvas


def make_row(
    baseline_dir: Path,
    candidate_dir: Path,
    row: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
) -> Image.Image:
    index = int(row["index"])
    sample = f"{index:04d}"
    baseline_sample = baseline_dir / "samples" / sample
    candidate_sample = candidate_dir / "samples" / sample
    pieces = [
        add_label(load_image(candidate_sample / "lr_nearest.png"), f"{sample} LR"),
        add_label(load_image(candidate_sample / "bicubic.png"), "bicubic"),
        add_label(load_image(baseline_sample / "sr.png"), f"{baseline_label} {row['baseline_psnr']:.2f}"),
        add_label(load_image(candidate_sample / "sr.png"), f"{candidate_label} {row['candidate_psnr']:.2f}"),
        add_label(load_image(candidate_sample / "gt.png"), f"GT d={row['delta']:+.3f}"),
    ]
    width = sum(piece.width for piece in pieces)
    height = max(piece.height for piece in pieces)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    left = 0
    for piece in pieces:
        canvas.paste(piece, (left, 0))
        left += piece.width
    return canvas


def make_sheet(
    baseline_dir: Path,
    candidate_dir: Path,
    rows: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
) -> Image.Image:
    image_rows = [
        make_row(baseline_dir, candidate_dir, row, baseline_label, candidate_label)
        for row in rows
    ]
    width = max(row.width for row in image_rows)
    height = sum(row.height for row in image_rows)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    top = 0
    for row in image_rows:
        canvas.paste(row, (0, top))
        top += row.height
    return canvas


def main() -> None:
    args = parse_args()
    baseline_metrics = args.baseline_dir / "metrics.csv"
    candidate_metrics = args.candidate_dir / "metrics.csv"
    if not baseline_metrics.exists():
        raise FileNotFoundError(baseline_metrics)
    if not candidate_metrics.exists():
        raise FileNotFoundError(candidate_metrics)

    baseline_rows = read_rows(baseline_metrics)
    candidate_rows = read_rows(candidate_metrics)
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError(f"Metric row count differs: {len(baseline_rows)} vs {len(candidate_rows)}")

    compared: list[dict[str, Any]] = []
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        if baseline["index"] != candidate["index"]:
            raise ValueError(f"Index mismatch: {baseline['index']} vs {candidate['index']}")
        baseline_psnr = psnr(baseline)
        candidate_psnr = psnr(candidate)
        compared.append(
            {
                "index": int(baseline["index"]),
                "path": baseline["path"],
                "domain": baseline["domain"],
                "baseline_psnr": baseline_psnr,
                "candidate_psnr": candidate_psnr,
                "bicubic_psnr": psnr(candidate, "bicubic_psnr"),
                "delta": candidate_psnr - baseline_psnr,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compared[0].keys()))
        writer.writeheader()
        writer.writerows(compared)

    deltas = [row["delta"] for row in compared]
    summary = {
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "count": len(compared),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "wins": int(sum(delta > 0.0 for delta in deltas)),
        "losses": int(sum(delta < 0.0 for delta in deltas)),
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top_k = max(1, min(int(args.top_k), len(compared)))
    wins = sorted(compared, key=lambda row: row["delta"], reverse=True)[:top_k]
    losses = sorted(compared, key=lambda row: row["delta"])[:top_k]
    make_sheet(args.baseline_dir, args.candidate_dir, wins, args.baseline_label, args.candidate_label).save(
        args.output_dir / "top_wins.png"
    )
    make_sheet(args.baseline_dir, args.candidate_dir, losses, args.baseline_label, args.candidate_label).save(
        args.output_dir / "top_losses.png"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
