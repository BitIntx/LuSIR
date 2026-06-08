from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets.manifest import crop_square, pil_to_tensor
from sr_diffusion.models import AutoencoderKL
from sr_diffusion.utils import autocast_context, get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoencoder reconstruction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = ((tensor.detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    dtype_name = config["train"].get("dtype", "bf16")

    model = AutoencoderKL.from_config(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    hr_size = int(config["data"]["hr_size"])
    image = Image.open(args.input).convert("RGB")
    image = crop_square(image, hr_size, rng=random.Random(0), random_crop=False)
    x = pil_to_tensor(image).unsqueeze(0).to(device)
    x = x.mul(2.0).sub(1.0)

    with torch.no_grad(), autocast_context(device, dtype_name):
        output = model(x, sample_posterior=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "input.png")
    tensor_to_pil(output.reconstruction[0].float()).save(args.output_dir / "reconstruction.png")
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
