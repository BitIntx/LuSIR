from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.models import AutoencoderKL, ConditionalUNet, LRToLatentPredictor
from sr_diffusion.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count LuSIR model parameters from configs.")
    parser.add_argument("--autoencoder-config", type=Path, default=None)
    parser.add_argument("--condition-config", type=Path, default=None)
    parser.add_argument("--diffusion-config", type=Path, default=None)
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def build_on_meta(model_cls: type[torch.nn.Module], config_path: Path) -> torch.nn.Module:
    config = load_config(config_path)
    with torch.device("meta"):
        return model_cls.from_config(config["model"])


def main() -> None:
    args = parse_args()
    rows: list[tuple[str, int]] = []
    if args.autoencoder_config is not None:
        rows.append(("autoencoder", count_params(build_on_meta(AutoencoderKL, args.autoencoder_config))))
    if args.condition_config is not None:
        rows.append(("condition_encoder", count_params(build_on_meta(LRToLatentPredictor, args.condition_config))))
    if args.diffusion_config is not None:
        rows.append(("diffusion_unet", count_params(build_on_meta(ConditionalUNet, args.diffusion_config))))

    total = 0
    for name, params in rows:
        total += params
        print(f"{name}\t{params}\t{params / 1_000_000:.3f}M")
    print(f"total\t{total}\t{total / 1_000_000:.3f}M")


if __name__ == "__main__":
    main()
