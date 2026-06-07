from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.models import AutoencoderKL, LRToLatentPredictor
from sr_diffusion.utils import autocast_context, get_device, load_config, save_config, seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a deterministic bounded residual refiner on top of Stage 2.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    return parser.parse_args()


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    return 20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(max(mse, 1e-12)))


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt((prediction.float() - target.float()).pow(2) + float(eps) ** 2).mean()


def apply_residual_strength(
    condition: torch.Tensor,
    residual: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    strength = float(strength)
    if strength < 0.0:
        raise ValueError(f"residual_strength must be non-negative, got {strength}")
    return condition + strength * residual


def lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError(f"highpass kernel must be odd, got {kernel_size}")
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def highpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return x - lowpass(x, kernel_size)


def metric_highpass(x: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError(f"metric highpass kernel must be odd, got {kernel_size}")
    padding = kernel_size // 2
    padded = F.pad(x.float(), (padding, padding, padding, padding), mode="reflect")
    return x.float() - F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def laplacian_response(x: torch.Tensor) -> torch.Tensor:
    kernel = x.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    ) / 4.0
    channels = int(x.shape[1])
    weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    padded = F.pad(x.float(), (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, weight, groups=channels)


def ssim_per_image(prediction: torch.Tensor, target: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    padding = kernel_size // 2

    def local_mean(x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)

    mu_prediction = local_mean(prediction)
    mu_target = local_mean(target)
    prediction_variance = local_mean(prediction * prediction) - mu_prediction.pow(2)
    target_variance = local_mean(target * target) - mu_target.pow(2)
    covariance = local_mean(prediction * target) - mu_prediction * mu_target
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_prediction * mu_target + c1) * (2.0 * covariance + c2)
    denominator = (mu_prediction.pow(2) + mu_target.pow(2) + c1) * (
        prediction_variance + target_variance + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).flatten(1).mean(dim=1)


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def init_wandb(config: dict[str, Any], output_dir: Path, model: nn.Module) -> Any | None:
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging is enabled, but wandb is not installed") from exc
    wandb_dir = Path(wandb_cfg.get("dir", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = str(wandb_cfg.get("mode", "offline"))
    os.environ["WANDB_MODE"] = mode
    run = wandb.init(
        project=wandb_cfg.get("project", "sr-diffusion"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", config.get("project", {}).get("name")),
        dir=str(wandb_dir),
        mode=mode,
        tags=wandb_cfg.get("tags"),
        config=clean_config(config),
    )
    if bool(wandb_cfg.get("watch", False)):
        wandb.watch(model, log="gradients", log_freq=int(wandb_cfg.get("watch_log_freq", 100)))
    print(f"wandb_run={run.url}", flush=True)
    return run


def wandb_log(run: Any | None, data: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def make_dataset(config: dict[str, Any], split: str, seed: int, deterministic: bool | None = None) -> ManifestImageDataset:
    data_config = config["data"]
    return ManifestImageDataset(
        manifest_path=data_config["manifest"],
        split=split,
        hr_size=data_config.get("hr_size", 512),
        scale=data_config.get("scale", 4),
        domains=data_config.get("domains", {"photo": 0, "anime": 1}),
        degradation_preset=data_config.get("degradation_preset", "mild"),
        seed=seed,
        deterministic=deterministic,
    )


def make_eval_loader(config: dict[str, Any], seed: int, device: torch.device) -> DataLoader:
    eval_config = config.get("eval", {})
    split = str(eval_config.get("split", "val"))
    dataset = make_dataset(config, split=split, seed=seed, deterministic=True)
    limit = int(eval_config.get("limit", 100))
    if limit > 0 and limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(
        dataset,
        batch_size=int(eval_config.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(eval_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def load_autoencoder(config: dict[str, Any], device: torch.device) -> AutoencoderKL:
    auto_cfg = config["autoencoder"]
    vae_config = load_config(auto_cfg["config"])
    vae = AutoencoderKL.from_config(vae_config["model"])
    checkpoint = torch.load(auto_cfg["checkpoint"], map_location="cpu")
    vae.load_state_dict(checkpoint["model"])
    vae.to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"loaded_autoencoder={auto_cfg['checkpoint']} step={checkpoint.get('step', 'unknown')}")
    return vae


def load_condition_encoder(config: dict[str, Any], device: torch.device) -> LRToLatentPredictor:
    cond_cfg = config["condition_encoder"]
    cond_config = load_config(cond_cfg["config"])
    encoder = LRToLatentPredictor.from_config(cond_config["model"])
    checkpoint = torch.load(cond_cfg["checkpoint"], map_location="cpu")
    encoder.load_state_dict(checkpoint["model"])
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    print(f"loaded_condition_encoder={cond_cfg['checkpoint']} step={checkpoint.get('step', 'unknown')}")
    return encoder


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, math.gcd(channels, groups)), num_channels=channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        self.norm1 = _norm(channels, groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _norm(channels, groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class BoundedResidualRefiner(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        lr_channels: int = 3,
        hidden_channels: int = 128,
        num_blocks: int = 8,
        norm_groups: int = 32,
        num_domains: int = 2,
        residual_scale: float = 1.25,
        gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.residual_scale = float(residual_scale)
        self.gate_bias = float(gate_bias)
        self.input = nn.Conv2d(self.latent_channels + int(lr_channels), hidden_channels, kernel_size=3, padding=1)
        self.domain_embedding = nn.Embedding(num_domains, hidden_channels)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels, groups=norm_groups) for _ in range(num_blocks)])
        self.output_norm = _norm(hidden_channels, norm_groups)
        self.output = nn.Conv2d(hidden_channels, self.latent_channels * 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BoundedResidualRefiner":
        return cls(
            latent_channels=config.get("latent_channels", 16),
            lr_channels=config.get("lr_channels", 3),
            hidden_channels=config.get("hidden_channels", 128),
            num_blocks=config.get("num_blocks", 8),
            norm_groups=config.get("norm_groups", 32),
            num_domains=config.get("num_domains", 2),
            residual_scale=config.get("residual_scale", 1.25),
            gate_bias=config.get("gate_bias", 0.0),
        )

    def forward(
        self,
        condition: torch.Tensor,
        lr_input: torch.Tensor,
        domain_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([condition, lr_input.to(dtype=condition.dtype)], dim=1)
        x = self.input(x)
        if domain_id is not None:
            x = x + self.domain_embedding(domain_id).unsqueeze(-1).unsqueeze(-1)
        x = self.blocks(x)
        output = self.output(F.silu(self.output_norm(x)))
        residual_logits, gate_logits = torch.chunk(output, 2, dim=1)
        gate = torch.sigmoid(gate_logits + self.gate_bias)
        residual = self.residual_scale * torch.tanh(residual_logits) * gate
        refined = condition + residual
        return refined, residual, gate


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": clean_config(config),
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")


def add_label(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 18
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill="black", font=font)
    return canvas


def make_grid(rows: list[list[tuple[str, Image.Image]]], output_path: Path, gap: int = 6) -> None:
    if not rows:
        return
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
def evaluate(
    model: BoundedResidualRefiner,
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    output_dir: Path | None = None,
    sample_count: int = 0,
    residual_strength: float = 1.0,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {
        "condition_decoded_mse": 0.0,
        "refined_decoded_mse": 0.0,
        "bicubic_mse": 0.0,
        "oracle_full_decoded_mse": 0.0,
        "latent_mse": 0.0,
        "residual_l1": 0.0,
        "gate_mean": 0.0,
        "wins_vs_condition": 0.0,
        "condition_psnr": 0.0,
        "refined_psnr": 0.0,
        "bicubic_psnr": 0.0,
        "oracle_full_psnr": 0.0,
        "condition_ssim": 0.0,
        "refined_ssim": 0.0,
        "condition_highpass_mae": 0.0,
        "refined_highpass_mae": 0.0,
        "condition_laplacian_mae": 0.0,
        "refined_laplacian_mae": 0.0,
        "condition_laplacian_energy_ratio": 0.0,
        "refined_laplacian_energy_ratio": 0.0,
        "detail_wins_vs_condition": 0.0,
    }
    count = 0
    grid_rows: list[list[tuple[str, Image.Image]]] = []
    for batch in dataloader:
        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        target = normalize_image(hr)
        lr_input = normalize_image(lr)
        batch_size = int(hr.shape[0])
        with autocast_context(device, dtype_name):
            target_latent, _ = vae.encode(target)
            condition = condition_encoder(lr_input, domain_id)
            _, residual, gate = model(condition, lr_input, domain_id)
            applied_residual = residual * float(residual_strength)
            refined = apply_residual_strength(condition, residual, residual_strength)
            decoded_condition = denormalize(vae.decode(condition)).float()
            decoded_refined = denormalize(vae.decode(refined)).float()
            decoded_oracle = denormalize(vae.decode(target_latent)).float()
        bicubic = F.interpolate(lr.float(), size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        condition_mse_per = (decoded_condition - hr).float().pow(2).flatten(1).mean(dim=1)
        refined_mse_per = (decoded_refined - hr).float().pow(2).flatten(1).mean(dim=1)
        bicubic_mse_per = (bicubic - hr).float().pow(2).flatten(1).mean(dim=1)
        oracle_mse_per = (decoded_oracle - hr).float().pow(2).flatten(1).mean(dim=1)
        condition_ssim_per = ssim_per_image(decoded_condition, hr)
        refined_ssim_per = ssim_per_image(decoded_refined, hr)
        target_highpass = metric_highpass(hr)
        condition_highpass_mae_per = (metric_highpass(decoded_condition) - target_highpass).abs().flatten(1).mean(dim=1)
        refined_highpass_mae_per = (metric_highpass(decoded_refined) - target_highpass).abs().flatten(1).mean(dim=1)
        target_laplacian = laplacian_response(hr)
        condition_laplacian = laplacian_response(decoded_condition)
        refined_laplacian = laplacian_response(decoded_refined)
        condition_laplacian_mae_per = (condition_laplacian - target_laplacian).abs().flatten(1).mean(dim=1)
        refined_laplacian_mae_per = (refined_laplacian - target_laplacian).abs().flatten(1).mean(dim=1)
        target_laplacian_energy_per = target_laplacian.abs().flatten(1).mean(dim=1).clamp_min(1e-12)
        condition_laplacian_energy_ratio_per = (
            condition_laplacian.abs().flatten(1).mean(dim=1) / target_laplacian_energy_per
        )
        refined_laplacian_energy_ratio_per = (
            refined_laplacian.abs().flatten(1).mean(dim=1) / target_laplacian_energy_per
        )
        totals["condition_decoded_mse"] += float(condition_mse_per.sum().cpu())
        totals["refined_decoded_mse"] += float(refined_mse_per.sum().cpu())
        totals["bicubic_mse"] += float(bicubic_mse_per.sum().cpu())
        totals["oracle_full_decoded_mse"] += float(oracle_mse_per.sum().cpu())
        totals["condition_psnr"] += float((-10.0 * torch.log10(condition_mse_per.clamp_min(1e-12))).sum().cpu())
        totals["refined_psnr"] += float((-10.0 * torch.log10(refined_mse_per.clamp_min(1e-12))).sum().cpu())
        totals["bicubic_psnr"] += float((-10.0 * torch.log10(bicubic_mse_per.clamp_min(1e-12))).sum().cpu())
        totals["oracle_full_psnr"] += float((-10.0 * torch.log10(oracle_mse_per.clamp_min(1e-12))).sum().cpu())
        totals["condition_ssim"] += float(condition_ssim_per.sum().cpu())
        totals["refined_ssim"] += float(refined_ssim_per.sum().cpu())
        totals["condition_highpass_mae"] += float(condition_highpass_mae_per.sum().cpu())
        totals["refined_highpass_mae"] += float(refined_highpass_mae_per.sum().cpu())
        totals["condition_laplacian_mae"] += float(condition_laplacian_mae_per.sum().cpu())
        totals["refined_laplacian_mae"] += float(refined_laplacian_mae_per.sum().cpu())
        totals["condition_laplacian_energy_ratio"] += float(condition_laplacian_energy_ratio_per.sum().cpu())
        totals["refined_laplacian_energy_ratio"] += float(refined_laplacian_energy_ratio_per.sum().cpu())
        totals["detail_wins_vs_condition"] += float(
            (refined_laplacian_mae_per < condition_laplacian_mae_per).float().sum().cpu()
        )
        totals["latent_mse"] += float(F.mse_loss(refined.float(), target_latent.float(), reduction="sum").cpu()) / float(
            refined.shape[1] * refined.shape[2] * refined.shape[3]
        )
        totals["residual_l1"] += float(applied_residual.detach().float().abs().mean().cpu()) * batch_size
        totals["gate_mean"] += float(gate.detach().float().mean().cpu()) * batch_size
        totals["wins_vs_condition"] += float((refined_mse_per < condition_mse_per).float().sum().cpu())
        if output_dir is not None and len(grid_rows) < sample_count:
            lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)
            for item_idx in range(batch_size):
                if len(grid_rows) >= sample_count:
                    break
                grid_rows.append(
                    [
                        ("LR", tensor_to_pil(lr_nearest[item_idx])),
                        ("bicubic", tensor_to_pil(bicubic[item_idx])),
                        ("condition", tensor_to_pil(decoded_condition[item_idx])),
                        ("refined", tensor_to_pil(decoded_refined[item_idx])),
                        ("oracle", tensor_to_pil(decoded_oracle[item_idx])),
                        ("GT", tensor_to_pil(hr[item_idx])),
                    ]
                )
        count += batch_size
    count = max(1, count)
    metrics = {
        "eval/condition_decoded_mse": totals["condition_decoded_mse"] / count,
        "eval/refined_decoded_mse": totals["refined_decoded_mse"] / count,
        "eval/bicubic_mse": totals["bicubic_mse"] / count,
        "eval/oracle_full_decoded_mse": totals["oracle_full_decoded_mse"] / count,
        "eval/latent_mse": totals["latent_mse"] / count,
        "eval/residual_l1": totals["residual_l1"] / count,
        "eval/gate_mean": totals["gate_mean"] / count,
        "eval/wins_vs_condition": totals["wins_vs_condition"],
        "eval/num_images": float(count),
        "eval/condition_mean_psnr": totals["condition_psnr"] / count,
        "eval/refined_mean_psnr": totals["refined_psnr"] / count,
        "eval/bicubic_mean_psnr": totals["bicubic_psnr"] / count,
        "eval/oracle_full_mean_psnr": totals["oracle_full_psnr"] / count,
        "eval/condition_ssim": totals["condition_ssim"] / count,
        "eval/refined_ssim": totals["refined_ssim"] / count,
        "eval/condition_highpass_mae": totals["condition_highpass_mae"] / count,
        "eval/refined_highpass_mae": totals["refined_highpass_mae"] / count,
        "eval/condition_laplacian_mae": totals["condition_laplacian_mae"] / count,
        "eval/refined_laplacian_mae": totals["refined_laplacian_mae"] / count,
        "eval/condition_laplacian_energy_ratio": totals["condition_laplacian_energy_ratio"] / count,
        "eval/refined_laplacian_energy_ratio": totals["refined_laplacian_energy_ratio"] / count,
        "eval/detail_wins_vs_condition": totals["detail_wins_vs_condition"],
        "eval/residual_strength": float(residual_strength),
    }
    metrics["eval/condition_decoded_psnr"] = psnr_from_mse(metrics["eval/condition_decoded_mse"])
    metrics["eval/refined_decoded_psnr"] = psnr_from_mse(metrics["eval/refined_decoded_mse"])
    metrics["eval/bicubic_psnr"] = psnr_from_mse(metrics["eval/bicubic_mse"])
    metrics["eval/oracle_full_decoded_psnr"] = psnr_from_mse(metrics["eval/oracle_full_decoded_mse"])
    metrics["eval/refined_vs_condition_psnr"] = (
        metrics["eval/refined_decoded_psnr"] - metrics["eval/condition_decoded_psnr"]
    )
    metrics["eval/refined_vs_condition_mean_psnr"] = (
        metrics["eval/refined_mean_psnr"] - metrics["eval/condition_mean_psnr"]
    )
    metrics["eval/refined_vs_condition_ssim"] = metrics["eval/refined_ssim"] - metrics["eval/condition_ssim"]
    metrics["eval/refined_vs_condition_highpass_mae"] = (
        metrics["eval/condition_highpass_mae"] - metrics["eval/refined_highpass_mae"]
    )
    metrics["eval/refined_vs_condition_laplacian_mae"] = (
        metrics["eval/condition_laplacian_mae"] - metrics["eval/refined_laplacian_mae"]
    )
    if output_dir is not None and grid_rows:
        make_grid(grid_rows, output_dir / "eval_grid_lr_bicubic_condition_refined_oracle_gt.png")
    if was_training:
        model.train()
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 1337))
    seed_everything(seed)
    device = get_device(str(config.get("train", {}).get("device", "auto")))
    dtype_name = str(config.get("train", {}).get("dtype", "bf16"))
    output_dir = Path(config["project"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_config(clean_config(config), output_dir / "config.yaml")

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    model = BoundedResidualRefiner.from_config(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed={args.resume} step={start_step}")
        resume_lr = config.get("train", {}).get("resume_lr")
        if resume_lr is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = float(resume_lr)
            print(f"resume_lr={float(resume_lr):.8f}")
    run = init_wandb(config, output_dir, model)

    train_dataset = make_dataset(config, split=str(config["data"].get("split", "train")), seed=seed, deterministic=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 6)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    eval_loader = make_eval_loader(config, seed=seed, device=device)

    loss_cfg = config.get("loss", {})
    train_cfg = config.get("train", {})
    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 2000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every", 25))
    save_every = int(train_cfg.get("save_every", 500))
    eval_cfg = config.get("eval", {})
    eval_every = int(eval_cfg.get("every", 250))
    sample_count = int(eval_cfg.get("sample_count", 8))
    eps = float(loss_cfg.get("charbonnier_eps", 1e-3))
    highpass_kernel = int(loss_cfg.get("highpass_kernel", 15))
    decoded_weight = float(loss_cfg.get("decoded_weight", 0.0))
    decoded_highpass_weight = float(loss_cfg.get("decoded_highpass_weight", 0.0))
    decoded_highpass_kernel = int(loss_cfg.get("decoded_highpass_kernel", highpass_kernel))

    best_metric_name = str(eval_cfg.get("best_metric", "eval/refined_decoded_psnr"))
    best_mode = str(eval_cfg.get("best_mode", "max"))
    if best_mode not in {"min", "max"}:
        raise ValueError(f"eval.best_mode must be 'min' or 'max', got {best_mode!r}")
    best_metric = float("-inf") if best_mode == "max" else float("inf")
    best_metrics: dict[str, float] | None = None
    summary_path = output_dir / "summary.json"
    metrics_log_path = output_dir / "metrics.jsonl"

    def run_eval(step: int) -> dict[str, float]:
        eval_dir = output_dir / f"eval_step_{step:06d}"
        metrics = evaluate(
            model=model,
            vae=vae,
            condition_encoder=condition_encoder,
            dataloader=eval_loader,
            device=device,
            dtype_name=dtype_name,
            output_dir=eval_dir,
            sample_count=sample_count,
        )
        print(
            f"eval step={step} refined_psnr={metrics['eval/refined_decoded_psnr']:.4f} "
            f"condition_psnr={metrics['eval/condition_decoded_psnr']:.4f} "
            f"delta={metrics['eval/refined_vs_condition_psnr']:+.4f} "
            f"mean_delta={metrics['eval/refined_vs_condition_mean_psnr']:+.4f} "
            f"ssim_delta={metrics['eval/refined_vs_condition_ssim']:+.5f} "
            f"lap_delta={metrics['eval/refined_vs_condition_laplacian_mae']:+.6f} "
            f"detail_energy={metrics['eval/refined_laplacian_energy_ratio']:.4f} "
            f"wins={metrics['eval/wins_vs_condition']:.0f}/{metrics['eval/num_images']:.0f}"
        )
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
        wandb_data: dict[str, Any] = dict(metrics)
        grid_path = eval_dir / "eval_grid_lr_bicubic_condition_refined_oracle_gt.png"
        if run is not None and grid_path.exists():
            import wandb

            wandb_data["samples/eval_grid"] = wandb.Image(str(grid_path), caption=f"eval step {step}")
        wandb_log(run, wandb_data, step=step)
        return metrics

    if args.eval_only_checkpoint is not None:
        checkpoint_step = load_checkpoint(args.eval_only_checkpoint, model, optimizer, device)
        metrics = run_eval(checkpoint_step)
        summary = {
            "config": str(args.config),
            "checkpoint": str(args.eval_only_checkpoint),
            "checkpoint_step": checkpoint_step,
            "metrics": metrics,
        }
        (output_dir / f"eval_only_step_{checkpoint_step:06d}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if run is not None:
            run.finish()
        return

    existing_best_path = checkpoints_dir / "best_eval_refined.pt"
    if start_step > 0 and existing_best_path.exists():
        existing_best = torch.load(existing_best_path, map_location="cpu")
        existing_best_metrics = existing_best.get("metrics", {})
        if best_metric_name in existing_best_metrics:
            best_metric = float(existing_best_metrics[best_metric_name])
            best_metrics = existing_best_metrics
            print(f"preserved_best={best_metric_name} value={best_metric:.8f} step={existing_best.get('step', 'unknown')}")

    if bool(eval_cfg.get("run_at_start", True)) and start_step == 0:
        metrics = run_eval(0)
        best_metric = float(metrics[best_metric_name])
        best_metrics = metrics
        save_checkpoint(checkpoints_dir / "best_eval_refined.pt", model, optimizer, 0, config, metrics)

    step = start_step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    last_log_time = time.time()
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        target = normalize_image(hr)
        lr_input = normalize_image(lr)
        with torch.no_grad(), autocast_context(device, dtype_name):
            target_latent, _ = vae.encode(target)
            condition = condition_encoder(lr_input, domain_id)

        with autocast_context(device, dtype_name):
            refined, predicted_residual, gate = model(condition.detach(), lr_input, domain_id)
            target_residual = target_latent.detach() - condition.detach()
            latent_loss = charbonnier(refined, target_latent.detach(), eps)
            residual_loss = charbonnier(predicted_residual, target_residual, eps)
            highpass_loss = charbonnier(
                highpass(predicted_residual, highpass_kernel),
                highpass(target_residual, highpass_kernel),
                eps,
            )
            gate_loss = gate.float().abs().mean()
            decoded_loss = refined.new_zeros(())
            decoded_highpass_loss = refined.new_zeros(())
            if decoded_weight > 0.0 or decoded_highpass_weight > 0.0:
                decoded_refined = vae.decode(refined)
                if decoded_weight > 0.0:
                    decoded_loss = charbonnier(decoded_refined, target, eps)
                if decoded_highpass_weight > 0.0:
                    decoded_highpass_loss = charbonnier(
                        highpass(decoded_refined, decoded_highpass_kernel),
                        highpass(target, decoded_highpass_kernel),
                        eps,
                    )
            loss = (
                float(loss_cfg.get("latent_weight", 1.0)) * latent_loss
                + float(loss_cfg.get("residual_weight", 0.5)) * residual_loss
                + float(loss_cfg.get("highpass_weight", 1.0)) * highpass_loss
                + decoded_weight * decoded_loss
                + decoded_highpass_weight * decoded_highpass_loss
                + float(loss_cfg.get("gate_l1_weight", 0.001)) * gate_loss
            )
        (loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0 or step == 1:
            elapsed = max(time.time() - last_log_time, 1e-6)
            last_log_time = time.time()
            steps_per_s = log_every / elapsed
            train_metrics = {
                "train/loss": float(loss.detach().cpu()),
                "train/latent": float(latent_loss.detach().cpu()),
                "train/residual": float(residual_loss.detach().cpu()),
                "train/highpass": float(highpass_loss.detach().cpu()),
                "train/decoded": float(decoded_loss.detach().cpu()),
                "train/decoded_highpass": float(decoded_highpass_loss.detach().cpu()),
                "train/gate": float(gate_loss.detach().cpu()),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "system/steps_per_s": steps_per_s,
            }
            print(
                f"step={step} loss={train_metrics['train/loss']:.5f} "
                f"latent={train_metrics['train/latent']:.5f} "
                f"residual={train_metrics['train/residual']:.5f} "
                f"highpass={train_metrics['train/highpass']:.5f} "
                f"decoded={train_metrics['train/decoded']:.5f} "
                f"decoded_highpass={train_metrics['train/decoded_highpass']:.5f} "
                f"gate={train_metrics['train/gate']:.5f} "
                f"steps_per_s={steps_per_s:.3f}",
                flush=True,
            )
            wandb_log(run, train_metrics, step=step)

        if step % save_every == 0:
            save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, optimizer, step, config)
            save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config)

        if eval_every > 0 and step % eval_every == 0:
            metrics = run_eval(step)
            metric_value = float(metrics[best_metric_name])
            improved = metric_value > best_metric if best_mode == "max" else metric_value < best_metric
            if improved:
                best_metric = metric_value
                best_metrics = metrics
                save_checkpoint(checkpoints_dir / "best_eval_refined.pt", model, optimizer, step, config, metrics)
            model.train()

    final_metrics = run_eval(step)
    save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config, final_metrics)
    final_metric_value = float(final_metrics[best_metric_name])
    final_improved = final_metric_value > best_metric if best_mode == "max" else final_metric_value < best_metric
    if final_improved:
        best_metric = final_metric_value
        best_metrics = final_metrics
        save_checkpoint(checkpoints_dir / "best_eval_refined.pt", model, optimizer, step, config, final_metrics)

    summary = {
        "config": str(args.config),
        "output_dir": str(output_dir),
        "finished_step": step,
        "best_refined_decoded_psnr": best_metric,
        "best_metric_name": best_metric_name,
        "best_metric_mode": best_mode,
        "best_metric_value": best_metric,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "checkpoint_latest": str(checkpoints_dir / "latest.pt"),
        "checkpoint_best": str(checkpoints_dir / "best_eval_refined.pt"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
