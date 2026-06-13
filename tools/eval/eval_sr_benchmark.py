from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.eval.sr_benchmark_metrics import benchmark_metrics, matlab_bicubic_resize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standard full-image x4 SR benchmark outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="name=path pattern with {dataset} and {id}.")
    parser.add_argument("--include-bicubic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop-border", type=int, default=4)
    parser.add_argument("--sheet-count", type=int, default=12)
    parser.add_argument("--include-rgb-ssim", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def parse_candidates(values: list[str]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must use name=pattern: {value}")
        name, pattern = value.split("=", 1)
        candidates.append((name.strip(), pattern.strip()))
    return candidates


def read_manifest(path: Path, datasets: list[str], limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if datasets:
        rows = [row for row in rows if row["dataset"] in datasets]
    return rows[:limit] if limit > 0 else rows


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def evaluate_row(
    row: dict[str, str],
    manifest: Path,
    candidates: list[tuple[str, str]],
    include_bicubic: bool,
    crop_border_value: int,
    include_rgb_ssim: bool,
) -> list[dict[str, Any]]:
    target = load_rgb(resolve_path(manifest, row["hr_path"]))
    lr = load_rgb(resolve_path(manifest, row["lr_path"]))
    sample_candidates: list[tuple[str, np.ndarray, str]] = []
    if include_bicubic:
        bicubic = matlab_bicubic_resize(lr, float(row["scale"]))
        sample_candidates.append(("bicubic", bicubic, "generated:matlab_bicubic_resize"))
    for name, pattern in candidates:
        path = Path(pattern.format(dataset=row["dataset"], id=row["id"]))
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate {name} for {row['dataset']}/{row['id']}: {path}")
        sample_candidates.append((name, load_rgb(path), str(path)))
    records: list[dict[str, Any]] = []
    for name, prediction, path in sample_candidates:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {row['dataset']}/{row['id']} {name}: {prediction.shape} != {target.shape}"
            )
        records.append(
            {
                "dataset": row["dataset"],
                "id": row["id"],
                "candidate": name,
                "candidate_path": path,
                "hr_path": row["hr_path"],
                "lr_path": row["lr_path"],
                **benchmark_metrics(
                    prediction,
                    target,
                    crop_border_value,
                    include_rgb_ssim=include_rgb_ssim,
                ),
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_images": len({(row["dataset"], row["id"]) for row in records}), "candidates": {}}
    metric_names = [metric for metric in ("y_psnr", "y_ssim", "rgb_psnr", "rgb_ssim") if metric in records[0]]
    for candidate in sorted({row["candidate"] for row in records}):
        selected = [row for row in records if row["candidate"] == candidate]
        candidate_summary: dict[str, Any] = {"count": len(selected), "by_dataset": {}}
        for metric in metric_names:
            candidate_summary[f"mean_{metric}"] = float(np.mean([row[metric] for row in selected]))
        for dataset in sorted({row["dataset"] for row in selected}):
            dataset_rows = [row for row in selected if row["dataset"] == dataset]
            candidate_summary["by_dataset"][dataset] = {
                "count": len(dataset_rows),
                **{
                    f"mean_{metric}": float(np.mean([row[metric] for row in dataset_rows]))
                    for metric in metric_names
                },
            }
        summary["candidates"][candidate] = candidate_summary
    if "bicubic" in summary["candidates"]:
        indexed = {(row["dataset"], row["id"], row["candidate"]): row for row in records}
        for candidate, candidate_summary in summary["candidates"].items():
            if candidate == "bicubic":
                continue
            comparisons = []
            for row in records:
                if row["candidate"] != candidate:
                    continue
                baseline = indexed.get((row["dataset"], row["id"], "bicubic"))
                if baseline is not None:
                    comparisons.append((row, baseline))
            if not comparisons:
                continue
            for metric in metric_names:
                deltas = [row[metric] - baseline[metric] for row, baseline in comparisons]
                candidate_summary[f"mean_{metric}_delta_vs_bicubic"] = float(np.mean(deltas))
                candidate_summary[f"wins_{metric}_vs_bicubic"] = int(sum(delta > 0.0 for delta in deltas))
    return summary


def thumbnail(image: Image.Image, size: int = 240) -> Image.Image:
    result = image.copy().convert("RGB")
    result.thumbnail((size, size), Image.Resampling.LANCZOS)
    return result


def make_contact_sheet(
    rows: list[dict[str, str]],
    candidates: list[tuple[str, str]],
    manifest: Path,
    output_path: Path,
    count: int,
) -> None:
    rows = rows[:count]
    if not rows:
        return
    columns = 2 + len(candidates)
    cell = 256
    label = 20
    canvas = Image.new("RGB", (columns * cell, len(rows) * (cell + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        paths = [
            ("LR", resolve_path(manifest, row["lr_path"])),
            ("GT", resolve_path(manifest, row["hr_path"])),
            *[
                (name, Path(pattern.format(dataset=row["dataset"], id=row["id"])))
                for name, pattern in candidates
                if Path(pattern.format(dataset=row["dataset"], id=row["id"])).exists()
            ],
        ]
        for column, (name, path) in enumerate(paths):
            image = thumbnail(Image.open(path))
            x = column * cell + (cell - image.width) // 2
            y = row_index * (cell + label) + label + (cell - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((column * cell + 4, row_index * (cell + label) + 3), f"{row['dataset']} {row['id']} {name}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest, args.dataset, int(args.limit))
    candidates = parse_candidates(args.candidate)
    records: list[dict[str, Any]] = []
    worker_args = [
        (
            row,
            args.manifest,
            candidates,
            bool(args.include_bicubic),
            int(args.crop_border),
            bool(args.include_rgb_ssim),
        )
        for row in rows
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for index, row_records in enumerate(executor.map(lambda values: evaluate_row(*values), worker_args), start=1):
            records.extend(row_records)
            row = rows[index - 1]
            print(f"evaluated {index}/{len(rows)} {row['dataset']}/{row['id']}", flush=True)
    if not records:
        raise ValueError("No benchmark outputs were evaluated")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    summary.update(
        {
            "manifest": str(args.manifest),
            "crop_border": int(args.crop_border),
            "include_rgb_ssim": bool(args.include_rgb_ssim),
            "workers": int(args.workers),
            "protocol": "official x4 LR pairs; MATLAB BT.601 Y; scale-pixel shave; MATLAB-style SSIM",
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_contact_sheet(rows, candidates, args.manifest, args.output_dir / "contact_sheet.jpg", int(args.sheet_count))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
