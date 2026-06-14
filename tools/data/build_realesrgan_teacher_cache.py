from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.utils import load_config, seed_everything


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "RealESRGAN_x4plus": {
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRNet_x4plus": {
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Real-ESRGAN teacher images for detail-branch training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=sorted(MODEL_SPECS), default="RealESRGAN_x4plus")
    parser.add_argument("--weights-dir", type=Path, default=Path("/home/ubuntu/scratch/sr-diffusion/baselines/weights"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--tile-pad", type=int, default=16)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def install_torchvision_compat() -> None:
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    from torchvision.transforms import functional

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module.__name__] = module


def load_upsampler(model_name: str, weights_dir: Path, tile: int, tile_pad: int, fp32: bool, gpu_id: int) -> tuple[Any, Path]:
    install_torchvision_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer

    spec = MODEL_SPECS[model_name]
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_path = weights_dir / f"{model_name}.pth"
    if not model_path.exists():
        downloaded = load_file_from_url(url=spec["url"], model_dir=str(weights_dir), progress=True, file_name=None)
        model_path = Path(downloaded)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=int(spec["num_block"]), num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=int(tile),
        tile_pad=int(tile_pad),
        pre_pad=0,
        half=not fp32,
        gpu_id=int(gpu_id),
    )
    return upsampler, model_path


def make_dataset(config: dict[str, Any], split: str, seed: int) -> ManifestImageDataset:
    data_config = config["data"]
    return ManifestImageDataset(
        manifest_path=data_config["manifest"],
        split=split,
        hr_size=data_config.get("hr_size", 512),
        scale=data_config.get("scale", 4),
        domains=data_config.get("domains", {"photo": 0, "anime": 1}),
        degradation_preset=data_config.get("degradation_preset", "mild"),
        seed=seed,
        deterministic=True,
        hflip_prob=data_config.get("hflip_prob", 0.0),
        texture_crop_retries=data_config.get("texture_crop_retries", 1),
        texture_crop_downsample=data_config.get("texture_crop_downsample", 128),
        hr_color_jitter_prob=data_config.get("hr_color_jitter_prob", 0.0),
        hr_color_jitter=data_config.get("hr_color_jitter", (0.97, 1.03)),
    )


def tensor_to_uint8_chw(tensor: Any) -> np.ndarray:
    array = tensor.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def save_contact_sheet(rows: list[tuple[Image.Image, Image.Image, Image.Image]], path: Path) -> None:
    if not rows:
        return
    width, height = rows[0][0].size
    sheet = Image.new("RGB", (width * 3, height * len(rows)), color=(255, 255, 255))
    for row_idx, (lr, teacher, hr) in enumerate(rows):
        y = row_idx * height
        sheet.paste(lr.resize((width, height), Image.Resampling.NEAREST), (0, y))
        sheet.paste(teacher, (width, y))
        sheet.paste(hr, (width * 2, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)


def main() -> None:
    args = parse_args()
    import cv2

    config = load_config(args.config)
    seed = int(config.get("seed", 1337))
    seed_everything(seed)
    dataset = make_dataset(config, split=str(args.split), seed=seed)
    total = len(dataset)
    start_index = max(0, int(args.start_index))
    end_index = total if int(args.limit) <= 0 else min(total, start_index + int(args.limit))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upsampler, model_path = load_upsampler(
        args.model_name,
        weights_dir=args.weights_dir,
        tile=int(args.tile),
        tile_pad=int(args.tile_pad),
        fp32=bool(args.fp32),
        gpu_id=int(args.gpu_id),
    )
    sample_rows: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    processed = 0
    skipped = 0
    started = time.perf_counter()
    for index in range(start_index, end_index):
        output_path = args.output_dir / f"{index:08d}.png"
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        sample = dataset[index]
        lr_rgb = tensor_to_uint8_chw(sample["lr"])
        lr_bgr = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR)
        teacher_bgr, _ = upsampler.enhance(lr_bgr, outscale=int(config["data"].get("scale", 4)))
        teacher_rgb = cv2.cvtColor(teacher_bgr, cv2.COLOR_BGR2RGB)
        teacher = Image.fromarray(teacher_rgb, mode="RGB")
        teacher.save(output_path)
        processed += 1
        if len(sample_rows) < int(args.sample_count):
            lr_image = Image.fromarray(lr_rgb, mode="RGB")
            hr_image = Image.fromarray(tensor_to_uint8_chw(sample["hr"]), mode="RGB")
            sample_rows.append((lr_image, teacher, hr_image))
        if processed == 1 or processed % 100 == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f"processed={processed} skipped={skipped} index={index} "
                f"rate={processed / elapsed:.2f}/s output={output_path}",
                flush=True,
            )
    sample_sheet = args.output_dir / "samples_lr_teacher_hr.jpg"
    save_contact_sheet(sample_rows, sample_sheet)
    elapsed = time.perf_counter() - started
    summary = {
        "config": str(args.config),
        "split": str(args.split),
        "model_name": args.model_name,
        "model_path": str(model_path),
        "output_dir": str(args.output_dir),
        "start_index": start_index,
        "end_index": end_index,
        "num_selected": end_index - start_index,
        "num_processed": processed,
        "num_skipped": skipped,
        "seconds": elapsed,
        "images_per_second": processed / elapsed if elapsed > 0 and processed > 0 else None,
        "sample_sheet": str(sample_sheet),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
