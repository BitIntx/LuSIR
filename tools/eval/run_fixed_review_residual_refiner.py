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

from tools.infer.infer_diffusion import pil_to_tensor, resolve_path, tensor_to_pil
from tools.infer.infer_residual_refiner import load_refiner, make_grid, refine_batch
from tools.train.train_residual_refiner import BoundedResidualRefiner
from sr_diffusion.utils import get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic residual refiner on a frozen review manifest.")
    parser.add_argument("--config", type=Path, default=Path("configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--residual-strength", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_manifest(path: Path, limit: int = 0) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "domain", "hr_path", "lr_path", "bicubic_path"}
    if not rows:
        raise ValueError(f"Empty review manifest: {path}")
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Review manifest must include columns: {sorted(required)}")
    return rows[:limit] if limit > 0 else rows


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def resolve_checkpoint(config: dict[str, Any], requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    inference_checkpoint = config.get("inference", {}).get("checkpoint") or config.get("checkpoint")
    if inference_checkpoint:
        candidates.append(Path(inference_checkpoint))
    project_output = config.get("project", {}).get("output_dir")
    if project_output:
        candidates.append(Path(project_output) / "checkpoints" / "best_eval_refined.pt")
    for candidate in candidates:
        path = resolve_path(config, candidate.expanduser())
        if path.exists():
            return path.resolve()
    formatted = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find residual refiner checkpoint. Checked:\n{formatted}")


def load_models(config: dict[str, Any], checkpoint_path: Path, device: torch.device, dtype_name: str) -> tuple[Any, Any, BoundedResidualRefiner, int]:
    from tools.infer.infer_diffusion import load_autoencoder, load_condition_encoder

    vae = load_autoencoder(config, device, dtype_name)
    condition_encoder = load_condition_encoder(config, device, dtype_name)
    refiner, checkpoint_step = load_refiner(config, checkpoint_path, device)
    return vae, condition_encoder, refiner, checkpoint_step


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    config = load_config(args.config)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    data_config = config.get("data", {})
    domains = data_config.get("domains", {"photo": 0, "anime": 1})
    residual_strength = float(
        args.residual_strength
        if args.residual_strength is not None
        else config.get("inference", {}).get("residual_strength", 1.0)
    )

    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    vae, condition_encoder, refiner, checkpoint_step = load_models(config, checkpoint_path, device, dtype_name)
    rows = read_manifest(args.review_manifest, limit=int(args.limit))
    output_dir = args.output_dir
    sample_root = output_dir / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    for batch_start in range(0, len(rows), int(args.batch_size)):
        batch = rows[batch_start : batch_start + int(args.batch_size)]
        lr_tensors = []
        domain_ids = []
        for row in batch:
            lr_path = resolve_manifest_path(args.review_manifest, row["lr_path"])
            lr_tensors.append(pil_to_tensor(Image.open(lr_path).convert("RGB")))
            if row["domain"] not in domains:
                raise ValueError(f"Unknown domain {row['domain']!r}. Available: {sorted(domains)}")
            domain_ids.append(int(domains[row["domain"]]))
        lr = torch.stack(lr_tensors, dim=0).to(device)
        domain_id = torch.tensor(domain_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            decoded_condition, decoded_refined = refine_batch(
                vae=vae,
                condition_encoder=condition_encoder,
                refiner=refiner,
                lr=lr,
                domain_id=domain_id,
                dtype_name=dtype_name,
                residual_strength=residual_strength,
            )
        bicubic = F.interpolate(lr.float(), size=decoded_refined.shape[-2:], mode="bicubic", align_corners=False).clamp(
            0.0,
            1.0,
        )
        for item_index, row in enumerate(batch):
            sample_dir = sample_root / row["id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            lr_path = resolve_manifest_path(args.review_manifest, row["lr_path"])
            gt_path = resolve_manifest_path(args.review_manifest, row["hr_path"])
            shutil.copyfile(lr_path, sample_dir / "input_lr.png")
            shutil.copyfile(gt_path, sample_dir / "gt.png")
            tensor_to_pil(bicubic[item_index]).save(sample_dir / "bicubic.png")
            tensor_to_pil(decoded_condition[item_index]).save(sample_dir / "condition.png")
            tensor_to_pil(decoded_refined[item_index]).save(sample_dir / "refined.png")
            output_size = (int(decoded_refined.shape[-1]), int(decoded_refined.shape[-2]))
            lr_nearest = Image.open(lr_path).convert("RGB").resize(output_size, Image.Resampling.NEAREST)
            make_grid(
                [
                    ("LR", lr_nearest),
                    ("bicubic", Image.open(sample_dir / "bicubic.png").convert("RGB")),
                    ("condition", Image.open(sample_dir / "condition.png").convert("RGB")),
                    ("refined", Image.open(sample_dir / "refined.png").convert("RGB")),
                    ("GT", Image.open(sample_dir / "gt.png").convert("RGB")),
                ],
                sample_dir / "grid_lr_bicubic_condition_refined_gt.png",
            )
        processed += len(batch)
        print(f"processed {processed}/{len(rows)}", flush=True)

    summary = {
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "review_manifest": str(args.review_manifest),
        "output_dir": str(output_dir),
        "sample_root": str(sample_root),
        "num_samples": len(rows),
        "residual_strength": residual_strength,
        "dtype": dtype_name,
        "device": str(device),
        "candidate_patterns": {
            "condition": str(sample_root / "{id}" / "condition.png"),
            "refined": str(sample_root / "{id}" / "refined.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
