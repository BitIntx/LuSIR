from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train_residual_refiner import BoundedResidualRefiner, evaluate, load_autoencoder, load_condition_encoder, make_eval_loader
from sr_diffusion.utils import get_device, load_config, seed_everything


DEFAULT_HF_CHECKPOINT = Path("checkpoints/residual_refiner_stage2_xl_mild_best_eval_refined.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the deterministic residual refiner on a validation split.")
    parser.add_argument("--config", type=Path, default=Path("configs/residual_refiner_stage2_xl_mild_probe.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--degradation-preset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    return parser.parse_args()


def resolve_checkpoint(config: dict[str, Any], requested: Path | None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    inference_checkpoint = config.get("inference", {}).get("checkpoint") or config.get("checkpoint")
    if inference_checkpoint:
        candidates.append(Path(inference_checkpoint))
    candidates.append(DEFAULT_HF_CHECKPOINT)
    project_output = config.get("project", {}).get("output_dir")
    if project_output:
        candidates.append(Path(project_output) / "checkpoints" / "best_eval_refined.pt")

    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
    formatted = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find residual refiner checkpoint. Checked:\n{formatted}")


def build_output_dir(config: dict[str, Any], preset: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    scratch_root = Path(config.get("project", {}).get("output_dir", "runs/residual_refiner_stage2_xl_mild_probe")).parent
    return scratch_root / f"eval_residual_refiner_stage2_xl_{preset}_val100"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    preset = str(args.degradation_preset or config.get("data", {}).get("degradation_preset", "mild"))
    config["data"]["degradation_preset"] = preset
    config["eval"] = {
        **config.get("eval", {}),
        "split": args.split,
        "limit": int(args.limit),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "sample_count": int(args.sample_count),
    }
    if args.device is not None:
        config.setdefault("train", {})["device"] = args.device
    if args.dtype is not None:
        config.setdefault("train", {})["dtype"] = args.dtype

    seed = int(args.seed if args.seed is not None else config.get("seed", 1337))
    seed_everything(seed)
    device = get_device(str(config.get("train", {}).get("device", "auto")))
    dtype_name = str(config.get("train", {}).get("dtype", "bf16"))
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    output_dir = build_output_dir(config, preset, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    model = BoundedResidualRefiner.from_config(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    checkpoint_step = int(checkpoint.get("step", 0))

    dataloader = make_eval_loader(config, seed=seed, device=device)
    metrics = evaluate(
        model=model,
        vae=vae,
        condition_encoder=condition_encoder,
        dataloader=dataloader,
        device=device,
        dtype_name=dtype_name,
        output_dir=output_dir,
        sample_count=int(args.sample_count),
    )
    summary = {
        "config": str(args.config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "degradation_preset": preset,
        "split": args.split,
        "limit": int(args.limit),
        "seed": seed,
        "output_dir": str(output_dir),
        "grid": str(output_dir / "eval_grid_lr_bicubic_condition_refined_oracle_gt.png"),
        "metrics": metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"preset={preset} refined_mean_psnr={metrics['eval/refined_mean_psnr']:.4f} "
        f"condition_mean_psnr={metrics['eval/condition_mean_psnr']:.4f} "
        f"mean_delta={metrics['eval/refined_vs_condition_mean_psnr']:+.4f} "
        f"wins={metrics['eval/wins_vs_condition']:.0f}/{metrics['eval/num_images']:.0f} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
