from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.infer.infer_diffusion import pil_to_tensor, tensor_to_pil
from tools.infer.infer_residual_refiner import make_grid
from tools.train.train_detail_branch import (
    GatedHighFrequencyDetailBranch,
    load_autoencoder,
    load_checkpoint,
    load_condition_encoder,
    make_base_prediction,
)
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an image-space detail branch on a frozen review manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_manifest(path: Path, limit: int = 0) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "domain", "hr_path", "lr_path"}
    if not rows:
        raise ValueError(f"Empty review manifest: {path}")
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Review manifest must include columns: {sorted(required)}")
    return rows[:limit] if limit > 0 else rows


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    config = load_config(args.config)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    domains = config.get("data", {}).get("domains", {"photo": 0, "anime": 1})

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    model = GatedHighFrequencyDetailBranch.from_config(config["model"]).to(device)
    checkpoint_step = load_checkpoint(args.checkpoint, model, optimizer=None, device=device)
    model.eval()

    rows = read_manifest(args.review_manifest, limit=int(args.limit))
    output_dir = args.output_dir
    sample_root = output_dir / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    processed = 0

    for batch_start in range(0, len(rows), int(args.batch_size)):
        batch = rows[batch_start : batch_start + int(args.batch_size)]
        lr_tensors = []
        hr_tensors = []
        domain_ids = []
        for row in batch:
            lr_path = resolve_manifest_path(args.review_manifest, row["lr_path"])
            hr_path = resolve_manifest_path(args.review_manifest, row["hr_path"])
            lr_tensors.append(pil_to_tensor(Image.open(lr_path).convert("RGB")))
            hr_tensors.append(pil_to_tensor(Image.open(hr_path).convert("RGB")))
            if row["domain"] not in domains:
                raise ValueError(f"Unknown domain {row['domain']!r}. Available: {sorted(domains)}")
            domain_ids.append(int(domains[row["domain"]]))
        lr = torch.stack(lr_tensors, dim=0).to(device)
        hr = torch.stack(hr_tensors, dim=0).to(device)
        domain_id = torch.tensor(domain_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            condition, base_sr, bicubic = make_base_prediction(
                vae=vae,
                condition_encoder=condition_encoder,
                hr=hr,
                lr=lr,
                domain_id=domain_id,
                device=device,
                dtype_name=dtype_name,
            )
            with autocast_context(device, dtype_name):
                detail_sr, residual, _, _ = model(base_sr, bicubic, condition, domain_id)
        for item_index, row in enumerate(batch):
            sample_dir = sample_root / row["id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            lr_path = resolve_manifest_path(args.review_manifest, row["lr_path"])
            gt_path = resolve_manifest_path(args.review_manifest, row["hr_path"])
            shutil.copyfile(lr_path, sample_dir / "input_lr.png")
            shutil.copyfile(gt_path, sample_dir / "gt.png")
            tensor_to_pil(bicubic[item_index]).save(sample_dir / "bicubic.png")
            tensor_to_pil(base_sr[item_index]).save(sample_dir / "base.png")
            tensor_to_pil(detail_sr[item_index]).save(sample_dir / "detail.png")
            residual_vis = (residual[item_index].float() / float(config["model"].get("residual_scale", 0.18)) * 0.5 + 0.5).clamp(
                0.0,
                1.0,
            )
            tensor_to_pil(residual_vis).save(sample_dir / "residual.png")
            output_size = (int(detail_sr.shape[-1]), int(detail_sr.shape[-2]))
            lr_nearest = Image.open(lr_path).convert("RGB").resize(output_size, Image.Resampling.NEAREST)
            make_grid(
                [
                    ("LR", lr_nearest),
                    ("bicubic", Image.open(sample_dir / "bicubic.png").convert("RGB")),
                    ("base", Image.open(sample_dir / "base.png").convert("RGB")),
                    ("detail", Image.open(sample_dir / "detail.png").convert("RGB")),
                    ("residual", Image.open(sample_dir / "residual.png").convert("RGB")),
                    ("GT", Image.open(sample_dir / "gt.png").convert("RGB")),
                ],
                sample_dir / "grid_lr_bicubic_base_detail_residual_gt.png",
            )
        processed += len(batch)
        print(f"processed {processed}/{len(rows)}", flush=True)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "review_manifest": str(args.review_manifest),
        "output_dir": str(output_dir),
        "sample_root": str(sample_root),
        "num_samples": len(rows),
        "dtype": dtype_name,
        "device": str(device),
        "candidate_patterns": {
            "base": str(sample_root / "{id}" / "base.png"),
            "detail": str(sample_root / "{id}" / "detail.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
