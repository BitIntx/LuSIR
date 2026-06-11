from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets.manifest import crop_square, pil_to_tensor
from sr_diffusion.degradations import DegradationPipeline
from sr_diffusion.utils import load_config


@dataclass(frozen=True)
class SourceEntry:
    index: int
    path: Path
    domain: str
    split: str


@dataclass(frozen=True)
class ScoredEntry:
    entry: SourceEntry
    laplacian_energy: float
    edge_density: float
    colorfulness: float
    local_contrast: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen LR/HR review set for visual and perceptual SR checks.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config to read hr_size/scale/domains from.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=12, help="Number of source HR crops before preset expansion.")
    parser.add_argument("--candidate-pool-limit", type=int, default=0, help="0 means use every matching manifest row.")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--hr-size", type=int, default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--presets", nargs="+", default=["photo_detail_mix", "mild", "photo_v2", "photo_v3_noise_mix"])
    parser.add_argument("--domains", nargs="*", default=None, help="Optional domain allow-list, e.g. photo anime.")
    parser.add_argument("--copy-hr-only", action="store_true", help="Only save HR crops; skip LR degradation files.")
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_entries(manifest_path: Path, split: str, allowed_domains: set[str] | None) -> list[SourceEntry]:
    base_dir = manifest_path.parent
    entries: list[SourceEntry] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "domain", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must include columns: {sorted(required)}")
        for row_index, row in enumerate(reader):
            if row["split"] != split:
                continue
            if allowed_domains is not None and row["domain"] not in allowed_domains:
                continue
            entries.append(
                SourceEntry(
                    index=row_index,
                    path=resolve_path(base_dir, row["path"]),
                    domain=row["domain"],
                    split=row["split"],
                )
            )
    if not entries:
        raise ValueError(f"No manifest rows for split={split!r} domains={sorted(allowed_domains or [])}")
    return entries


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha1("|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def tensor_laplacian(gray: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    return F.conv2d(gray, kernel, padding=1)


def tensor_sobel(gray: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def score_image(image: Image.Image) -> dict[str, float]:
    tensor = pil_to_tensor(image).unsqueeze(0).float()
    gray = (0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]).contiguous()
    laplacian = tensor_laplacian(gray).abs()
    sobel = tensor_sobel(gray)
    color_std = tensor.flatten(2).std(dim=2).mean()
    local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    local_contrast = (gray - local_mean).abs().mean()
    return {
        "laplacian_energy": float(laplacian.mean().item()),
        "edge_density": float((sobel > sobel.mean()).float().mean().item()),
        "colorfulness": float(color_std.item()),
        "local_contrast": float(local_contrast.item()),
    }


def crop_entry(entry: SourceEntry, hr_size: int, seed: int) -> Image.Image:
    rng = random.Random(stable_seed(seed, entry.index, entry.path))
    with Image.open(entry.path) as image:
        return crop_square(image.convert("RGB"), hr_size, rng=rng, random_crop=False)


def score_entries(entries: list[SourceEntry], hr_size: int, seed: int) -> list[ScoredEntry]:
    scored: list[ScoredEntry] = []
    for entry in entries:
        try:
            crop = crop_entry(entry, hr_size=hr_size, seed=seed)
            metrics = score_image(crop)
        except Exception as exc:
            print(f"skip unreadable source index={entry.index} path={entry.path}: {exc}", flush=True)
            continue
        scored.append(ScoredEntry(entry=entry, **metrics))
    if not scored:
        raise ValueError("No readable images remained after scoring.")
    return scored


def select_bucket(
    scored: list[ScoredEntry],
    selected: set[int],
    key: str,
    take: int,
    *,
    reverse: bool = True,
) -> list[tuple[str, ScoredEntry]]:
    ordered = sorted(scored, key=lambda item: getattr(item, key), reverse=reverse)
    rows: list[tuple[str, ScoredEntry]] = []
    for item in ordered:
        if item.entry.index in selected:
            continue
        selected.add(item.entry.index)
        rows.append((key if reverse else f"low_{key}", item))
        if len(rows) >= take:
            break
    return rows


def select_review_sources(scored: list[ScoredEntry], count: int, seed: int) -> list[tuple[str, ScoredEntry]]:
    if count <= 0:
        raise ValueError(f"count must be positive: {count}")
    selected: set[int] = set()
    per_bucket = max(1, count // 5)
    rows: list[tuple[str, ScoredEntry]] = []
    rows.extend(select_bucket(scored, selected, "laplacian_energy", per_bucket, reverse=True))
    rows.extend(select_bucket(scored, selected, "edge_density", per_bucket, reverse=True))
    rows.extend(select_bucket(scored, selected, "colorfulness", per_bucket, reverse=True))
    rows.extend(select_bucket(scored, selected, "local_contrast", per_bucket, reverse=True))
    rows.extend(select_bucket(scored, selected, "laplacian_energy", per_bucket, reverse=False))

    remaining = [item for item in scored if item.entry.index not in selected]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    for item in remaining:
        if len(rows) >= count:
            break
        selected.add(item.entry.index)
        rows.append(("random_fill", item))
    return rows[:count]


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_config(args.config) if args.config else {}
    data_config = config.get("data", {})
    hr_size = int(args.hr_size or data_config.get("hr_size", 512))
    scale = int(args.scale or data_config.get("scale", 4))
    if hr_size % scale != 0:
        raise ValueError(f"hr_size must be divisible by scale: {hr_size}, {scale}")
    lr_size = hr_size // scale

    allowed_domains = set(args.domains) if args.domains else None
    entries = load_entries(args.manifest, split=args.split, allowed_domains=allowed_domains)
    if args.candidate_pool_limit > 0 and args.candidate_pool_limit < len(entries):
        rng = random.Random(args.seed)
        entries = rng.sample(entries, int(args.candidate_pool_limit))

    scored = score_entries(entries, hr_size=hr_size, seed=int(args.seed))
    selected = select_review_sources(scored, count=int(args.count), seed=int(args.seed))

    output_dir = args.output_dir
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    pipelines = {preset: DegradationPipeline.from_preset(preset, scale=scale) for preset in args.presets}
    for source_rank, (bucket, item) in enumerate(selected):
        crop = crop_entry(item.entry, hr_size=hr_size, seed=int(args.seed))
        source_id = f"{source_rank:04d}_{bucket}"
        source_dir = samples_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        hr_path = source_dir / "gt.png"
        crop.save(hr_path)
        source_rows.append(
            {
                "source_id": source_id,
                "source_index": item.entry.index,
                "source_path": str(item.entry.path),
                "domain": item.entry.domain,
                "split": item.entry.split,
                "bucket": bucket,
                "laplacian_energy": item.laplacian_energy,
                "edge_density": item.edge_density,
                "colorfulness": item.colorfulness,
                "local_contrast": item.local_contrast,
                "hr_path": relative_to_root(hr_path, output_dir),
            }
        )
        if args.copy_hr_only:
            continue
        for preset in args.presets:
            sample_id = f"{source_id}_{preset}"
            sample_dir = source_dir / preset
            sample_dir.mkdir(parents=True, exist_ok=True)
            preset_seed = stable_seed(args.seed, source_rank, item.entry.index, preset)
            lr = pipelines[preset].apply(crop, rng=random.Random(preset_seed), out_size=lr_size)
            bicubic = lr.resize((hr_size, hr_size), Image.Resampling.BICUBIC)
            lr_path = sample_dir / "lr.png"
            bicubic_path = sample_dir / "bicubic.png"
            lr.save(lr_path)
            bicubic.save(bicubic_path)
            rows.append(
                {
                    "id": sample_id,
                    "source_id": source_id,
                    "source_index": item.entry.index,
                    "source_path": str(item.entry.path),
                    "domain": item.entry.domain,
                    "split": item.entry.split,
                    "bucket": bucket,
                    "preset": preset,
                    "seed": preset_seed,
                    "hr_path": relative_to_root(hr_path, output_dir),
                    "lr_path": relative_to_root(lr_path, output_dir),
                    "bicubic_path": relative_to_root(bicubic_path, output_dir),
                    "notes": "",
                    "laplacian_energy": item.laplacian_energy,
                    "edge_density": item.edge_density,
                    "colorfulness": item.colorfulness,
                    "local_contrast": item.local_contrast,
                }
            )

    source_manifest = output_dir / "sources.csv"
    with source_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)

    review_manifest = output_dir / "review_manifest.csv"
    if rows:
        with review_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "manifest": str(args.manifest),
        "split": args.split,
        "source_count": len(source_rows),
        "sample_count": len(rows),
        "presets": list(args.presets),
        "hr_size": hr_size,
        "scale": scale,
        "seed": int(args.seed),
        "review_manifest": str(review_manifest),
        "source_manifest": str(source_manifest),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
