from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.eval import evaluate_autoencoder, make_eval_loader, save_eval_metrics
from sr_diffusion.models import AutoencoderKL
from sr_diffusion.utils import get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Stage 1 autoencoder checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    train_cfg = config["train"]
    eval_cfg = config.get("eval", {})
    split = args.split or str(eval_cfg.get("split", "val"))
    limit = args.limit if args.limit is not None else int(eval_cfg.get("limit", 0))
    batch_size = args.batch_size or int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
    dtype_name = train_cfg.get("dtype", "bf16")
    device = get_device(train_cfg.get("device", "auto"))

    model = AutoencoderKL.from_config(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    step = int(checkpoint.get("step", 0))

    dataloader = make_eval_loader(
        config,
        split=split,
        seed=seed,
        batch_size=batch_size,
        limit=limit,
        num_workers=int(eval_cfg.get("num_workers", config["data"].get("num_workers", 0))),
    )
    metrics = evaluate_autoencoder(model, dataloader, device, dtype_name, config["loss"])

    output_dir = args.output_dir or Path(config["project"]["output_dir"]) / "eval"
    metrics_path = output_dir / f"step_{step:07d}_{split}_metrics.json"
    save_eval_metrics(metrics_path, step=step, metrics=metrics)

    print(f"checkpoint={args.checkpoint}")
    print(f"split={split} step={step}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
