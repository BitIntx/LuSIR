from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.models import AutoencoderKL, LRToLatentPredictor
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything


DEFAULT_CANDIDATES = [
    (
        "best_eval_latent",
        "/home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/best_eval_latent.pt",
    ),
    (
        "step_0072000",
        "/home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/step_0072000.pt",
    ),
    (
        "latest",
        "/home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/latest.pt",
    ),
]


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 2 LR-to-latent candidate checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/latent_pretrain_photo100k_v3_noise_xl.yaml"))
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        default=None,
        help="Candidate checkpoint as label=/path/to/checkpoint.pt. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jwheojjang/scratch/sr-diffusion/runs/compare_stage2_xl_candidates"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--gt-last", action="store_true", help="Place GT after candidate outputs in the contact sheet.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    return parser.parse_args()


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def psnr_from_mse(mse: float, peak: float = 2.0) -> float:
    return 20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(max(mse, 1e-12)))


def charbonnier(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + 1e-6).mean()


def make_dataset(config: dict[str, Any], split: str) -> ManifestImageDataset:
    data_config = config["data"]
    return ManifestImageDataset(
        manifest_path=data_config["manifest"],
        split=split,
        hr_size=data_config.get("hr_size", 512),
        scale=data_config.get("scale", 4),
        domains=data_config.get("domains", {"photo": 0, "anime": 1}),
        degradation_preset=data_config.get("degradation_preset", "mild"),
        seed=int(config.get("seed", 0)),
        deterministic=True,
    )


def load_autoencoder(config: dict[str, Any], device: torch.device) -> AutoencoderKL:
    auto_cfg = config["autoencoder"]
    vae_config = load_config(auto_cfg["config"])
    vae = AutoencoderKL.from_config(vae_config["model"]).to(device)
    checkpoint = torch.load(auto_cfg["checkpoint"], map_location=device)
    vae.load_state_dict(checkpoint["model"])
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def load_candidate(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> tuple[LRToLatentPredictor, int]:
    model = LRToLatentPredictor.from_config(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, int(checkpoint.get("step", 0))


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    array = np.round(array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def add_label(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 18
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill="black", font=font)
    return canvas


def make_contact_sheet(rows: list[list[tuple[str, Image.Image]]], output_path: Path, gap: int = 6) -> None:
    if not rows:
        raise ValueError("No rows for contact sheet")
    labeled_rows = [[add_label(image, label) for label, image in row] for row in rows]
    cell_width = max(image.width for row in labeled_rows for image in row)
    cell_height = max(image.height for row in labeled_rows for image in row)
    columns = max(len(row) for row in labeled_rows)
    width = columns * cell_width + (columns + 1) * gap
    height = len(labeled_rows) * cell_height + (len(labeled_rows) + 1) * gap
    sheet = Image.new("RGB", (width, height), "white")
    for row_index, row in enumerate(labeled_rows):
        y = gap + row_index * (cell_height + gap)
        for column_index, image in enumerate(row):
            x = gap + column_index * (cell_width + gap)
            sheet.paste(image.convert("RGB"), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


@torch.no_grad()
def evaluate_candidate(
    model: LRToLatentPredictor,
    vae: AutoencoderKL,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
) -> dict[str, float]:
    totals = {"latent_loss": 0.0, "latent_mse": 0.0, "decoded_mse": 0.0}
    count = 0
    for batch in dataloader:
        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        target = normalize_image(hr)
        lr_input = normalize_image(lr)
        batch_size = int(hr.shape[0])
        with autocast_context(device, dtype_name):
            target_latent, _ = vae.encode(target)
            prediction = model(lr_input, domain_id)
            decoded = vae.decode(prediction)
            latent_loss = charbonnier(prediction, target_latent)
            latent_mse = F.mse_loss(prediction, target_latent)
            decoded_mse = F.mse_loss(decoded, target)
        totals["latent_loss"] += float(latent_loss.detach().cpu()) * batch_size
        totals["latent_mse"] += float(latent_mse.detach().cpu()) * batch_size
        totals["decoded_mse"] += float(decoded_mse.detach().cpu()) * batch_size
        count += batch_size
    count = max(1, count)
    decoded_mse_value = totals["decoded_mse"] / count
    return {
        "eval/latent_loss": totals["latent_loss"] / count,
        "eval/latent_mse": totals["latent_mse"] / count,
        "eval/decoded_mse": decoded_mse_value,
        "eval/decoded_psnr": psnr_from_mse(decoded_mse_value),
        "eval/num_images": float(count),
    }


@torch.no_grad()
def render_samples(
    candidates: list[tuple[str, LRToLatentPredictor]],
    vae: AutoencoderKL,
    dataset: ManifestImageDataset,
    indices: list[int],
    device: torch.device,
    dtype_name: str,
    gt_last: bool = False,
) -> list[list[tuple[str, Image.Image]]]:
    rows = []
    for index in indices:
        item = dataset[index % len(dataset)]
        hr = item["hr"].unsqueeze(0).to(device)
        lr = item["lr"].unsqueeze(0).to(device)
        domain_id = item["domain_id"].unsqueeze(0).to(device)
        lr_display = F.interpolate(item["lr"].unsqueeze(0), size=item["hr"].shape[-2:], mode="nearest").squeeze(0)
        row: list[tuple[str, Image.Image]] = [(f"idx {index} LR", tensor_to_pil(lr_display))]
        if not gt_last:
            row.append(("GT", tensor_to_pil(item["hr"])))
        lr_input = normalize_image(lr)
        with autocast_context(device, dtype_name):
            for label, model in candidates:
                prediction = model(lr_input, domain_id)
                decoded = denormalize(vae.decode(prediction)).squeeze(0)
                row.append((label, tensor_to_pil(decoded)))
        if gt_last:
            row.append(("GT", tensor_to_pil(item["hr"])))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)))
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates_arg = args.candidate or [(label, Path(path)) for label, path in DEFAULT_CANDIDATES]
    dataset = make_dataset(config, split=args.split)
    eval_indices = list(range(min(int(args.limit), len(dataset)))) if int(args.limit) > 0 else list(range(len(dataset)))
    eval_loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    vae = load_autoencoder(config, device)
    loaded_candidates: list[tuple[str, LRToLatentPredictor]] = []
    results: dict[str, Any] = {
        "config": str(args.config),
        "split": args.split,
        "limit": len(eval_indices),
        "indices": args.indices,
        "candidates": {},
    }

    for label, checkpoint_path in candidates_arg:
        model, step = load_candidate(config, checkpoint_path, device)
        metrics = evaluate_candidate(model, vae, eval_loader, device, dtype_name)
        print(
            f"{label}: step={step} "
            f"latent_loss={metrics['eval/latent_loss']:.5f} "
            f"latent_mse={metrics['eval/latent_mse']:.5f} "
            f"decoded_psnr={metrics['eval/decoded_psnr']:.2f}"
        )
        results["candidates"][label] = {
            "checkpoint": str(checkpoint_path),
            "step": step,
            "metrics": metrics,
        }
        loaded_candidates.append((label, model))

    rows = render_samples(loaded_candidates, vae, dataset, args.indices, device, dtype_name, gt_last=args.gt_last)
    contact_sheet_path = args.output_dir / "stage2_xl_candidate_contact_sheet.png"
    make_contact_sheet(rows, contact_sheet_path)
    results["contact_sheet"] = str(contact_sheet_path)
    metrics_path = args.output_dir / "stage2_xl_candidate_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
    print(f"wrote {contact_sheet_path}")


if __name__ == "__main__":
    main()
