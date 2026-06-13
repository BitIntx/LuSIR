from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.utils import get_device, load_config, seed_everything
from tools.infer.infer_detail_branch import load_detail_branch, resolve_checkpoint as resolve_detail_checkpoint, tiled_detail
from tools.infer.infer_diffusion import load_autoencoder, load_condition_encoder
from tools.infer.infer_residual_refiner import load_refiner, resolve_checkpoint as resolve_refiner_checkpoint, tiled_refine


VARIANTS: dict[str, dict[str, Any]] = {
    "detail_v1d": {
        "config": Path("configs/hf/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml"),
        "outputs": ("base", "detail"),
    },
    "refiner_v2": {
        "config": Path("configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml"),
        "outputs": ("condition", "refined"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic LuSIR path on standard full-image SR benchmarks.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_manifest(path: Path, datasets: list[str], limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset", "id", "scale", "hr_path", "lr_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must include {sorted(required)}")
    if datasets:
        rows = [row for row in rows if row["dataset"] in datasets]
    return rows[:limit] if limit > 0 else rows


def resolve_manifest_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    variant = VARIANTS[args.variant]
    config_path = args.config or variant["config"]
    config = load_config(config_path)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    scale = int(config["data"].get("scale", 4))
    tile_lr_size = int(config["data"]["hr_size"]) // scale
    domain_id = int(config["data"].get("domains", {"photo": 0})["photo"])
    vae = load_autoencoder(config, device, dtype_name)
    condition_encoder = load_condition_encoder(config, device, dtype_name)

    if args.variant == "detail_v1d":
        checkpoint = resolve_detail_checkpoint(config, args.checkpoint)
        model, checkpoint_step = load_detail_branch(config, checkpoint, device)
    else:
        checkpoint = resolve_refiner_checkpoint(config, args.checkpoint)
        model, checkpoint_step = load_refiner(config, checkpoint, device)

    rows = read_manifest(args.manifest, args.dataset, int(args.limit))
    timings: list[float] = []
    processed = 0
    for row in rows:
        sample_dir = args.output_dir / row["dataset"] / row["id"]
        output_names = variant["outputs"]
        output_paths = [sample_dir / f"{name}.png" for name in output_names]
        if args.skip_existing and all(path.exists() for path in output_paths):
            print(f"skip {row['dataset']}/{row['id']}", flush=True)
            continue
        lr_image = Image.open(resolve_manifest_path(args.manifest, row["lr_path"])).convert("RGB")
        started = time.perf_counter()
        if args.variant == "detail_v1d":
            first, second = tiled_detail(
                vae,
                condition_encoder,
                model,
                lr_image,
                domain_id,
                scale=scale,
                tile_lr_size=tile_lr_size,
                overlap_lr=int(args.tile_overlap),
                tile_batch_size=int(args.tile_batch_size),
                dtype_name=dtype_name,
                device=device,
                detail_strength=float(args.strength),
            )
        else:
            first, second = tiled_refine(
                vae,
                condition_encoder,
                model,
                lr_image,
                domain_id,
                scale=scale,
                tile_lr_size=tile_lr_size,
                overlap_lr=int(args.tile_overlap),
                tile_batch_size=int(args.tile_batch_size),
                dtype_name=dtype_name,
                device=device,
                residual_strength=float(args.strength),
            )
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        sample_dir.mkdir(parents=True, exist_ok=True)
        first.save(output_paths[0])
        second.save(output_paths[1])
        processed += 1
        print(f"processed {row['dataset']}/{row['id']} elapsed={elapsed:.2f}s", flush=True)

    summary = {
        "variant": args.variant,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "num_selected": len(rows),
        "num_processed": processed,
        "num_completed": sum(
            all((args.output_dir / row["dataset"] / row["id"] / f"{name}.png").exists() for name in variant["outputs"])
            for row in rows
        ),
        "datasets": args.dataset or sorted({row["dataset"] for row in rows}),
        "tile_overlap": int(args.tile_overlap),
        "tile_batch_size": int(args.tile_batch_size),
        "strength": float(args.strength),
        "mean_seconds": sum(timings) / len(timings) if timings else None,
        "candidate_patterns": {
            name: str(args.output_dir / "{dataset}" / "{id}" / f"{name}.png") for name in variant["outputs"]
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.variant}_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
