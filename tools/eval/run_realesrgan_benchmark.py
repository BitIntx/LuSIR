from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path
from typing import Any


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
    parser = argparse.ArgumentParser(description="Run official Real-ESRGAN x4 baselines on an SR benchmark manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=sorted(MODEL_SPECS), default="RealESRGAN_x4plus")
    parser.add_argument("--weights-dir", type=Path, default=Path("/home/ubuntu/scratch/sr-diffusion/baselines/weights"))
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--tile-pad", type=int, default=16)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def install_torchvision_compat() -> None:
    """Provide the legacy torchvision module expected by BasicSR 1.4.x."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    from torchvision.transforms import functional

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module.__name__] = module


def load_upsampler(model_name: str, weights_dir: Path, tile: int, tile_pad: int, fp32: bool, gpu_id: int) -> Any:
    install_torchvision_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer

    spec = MODEL_SPECS[model_name]
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=int(spec["num_block"]), num_grow_ch=32, scale=4)
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_path = weights_dir / f"{model_name}.pth"
    if not model_path.exists():
        downloaded = load_file_from_url(url=spec["url"], model_dir=str(weights_dir), progress=True, file_name=None)
        model_path = Path(downloaded)
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


def read_manifest(path: Path, datasets: list[str], limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if datasets:
        rows = [row for row in rows if row["dataset"] in datasets]
    return rows[:limit] if limit > 0 else rows


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the benchmark-baselines optional dependencies before running Real-ESRGAN.") from exc
    upsampler, model_path = load_upsampler(
        args.model_name,
        args.weights_dir,
        int(args.tile),
        int(args.tile_pad),
        bool(args.fp32),
        int(args.gpu_id),
    )
    rows = read_manifest(args.manifest, args.dataset, int(args.limit))
    timings: list[float] = []
    processed = 0
    for row in rows:
        output_path = args.output_dir / row["dataset"] / row["id"] / f"{args.model_name}.png"
        if args.skip_existing and output_path.exists():
            print(f"skip {row['dataset']}/{row['id']}", flush=True)
            continue
        image = cv2.imread(str(resolve_path(args.manifest, row["lr_path"])), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row["lr_path"])
        started = time.perf_counter()
        output, _ = upsampler.enhance(image, outscale=4)
        elapsed = time.perf_counter() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(f"Failed to write {output_path}")
        timings.append(elapsed)
        processed += 1
        print(f"processed {row['dataset']}/{row['id']} elapsed={elapsed:.2f}s", flush=True)
    summary = {
        "model_name": args.model_name,
        "model_path": str(model_path),
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "num_selected": len(rows),
        "num_processed": processed,
        "num_completed": sum(
            (args.output_dir / row["dataset"] / row["id"] / f"{args.model_name}.png").exists() for row in rows
        ),
        "datasets": args.dataset or sorted({row["dataset"] for row in rows}),
        "tile": int(args.tile),
        "tile_pad": int(args.tile_pad),
        "fp32": bool(args.fp32),
        "mean_seconds": sum(timings) / len(timings) if timings else None,
        "candidate_pattern": str(args.output_dir / "{dataset}" / "{id}" / f"{args.model_name}.png"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model_name}_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
