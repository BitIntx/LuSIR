from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.models import AutoencoderKL, LRToLatentPredictor
from sr_diffusion.utils import autocast_context, get_device, load_config, save_config, seed_everything, seed_worker
from tools.train.train_residual_refiner import (
    charbonnier,
    clean_config,
    denormalize,
    laplacian_response,
    lowpass,
    make_grid,
    metric_highpass,
    normalize_image,
    psnr_from_mse,
    ssim_per_image,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image-space gated high-frequency detail branch.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, math.gcd(channels, groups)), num_channels=channels)


class ImageResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_groups: int) -> None:
        super().__init__()
        self.norm1 = _norm(channels, norm_groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _norm(channels, norm_groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class GatedHighFrequencyDetailBranch(nn.Module):
    """Predicts a bounded high-frequency RGB residual on top of a decoded base SR image."""

    def __init__(
        self,
        image_channels: int = 3,
        latent_channels: int = 16,
        hidden_channels: int = 96,
        num_blocks: int = 8,
        norm_groups: int = 32,
        num_domains: int = 2,
        residual_scale: float = 0.18,
        gate_bias: float = -2.0,
        highpass_kernel: int = 15,
        use_condition_latent: bool = False,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.latent_channels = int(latent_channels)
        self.residual_scale = float(residual_scale)
        self.gate_bias = float(gate_bias)
        self.highpass_kernel = int(highpass_kernel)
        self.use_condition_latent = bool(use_condition_latent)
        input_channels = self.image_channels * 2
        if self.use_condition_latent:
            input_channels += self.latent_channels
        self.input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.domain_embedding = nn.Embedding(num_domains, hidden_channels)
        self.blocks = nn.Sequential(
            *[ImageResidualBlock(hidden_channels, norm_groups=norm_groups) for _ in range(int(num_blocks))]
        )
        self.output_norm = _norm(hidden_channels, norm_groups)
        self.output = nn.Conv2d(hidden_channels, self.image_channels + 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GatedHighFrequencyDetailBranch":
        return cls(
            image_channels=config.get("image_channels", 3),
            latent_channels=config.get("latent_channels", 16),
            hidden_channels=config.get("hidden_channels", 96),
            num_blocks=config.get("num_blocks", 8),
            norm_groups=config.get("norm_groups", 32),
            num_domains=config.get("num_domains", 2),
            residual_scale=config.get("residual_scale", 0.18),
            gate_bias=config.get("gate_bias", -2.0),
            highpass_kernel=config.get("highpass_kernel", 15),
            use_condition_latent=config.get("use_condition_latent", False),
        )

    def forward(
        self,
        base_sr: torch.Tensor,
        bicubic: torch.Tensor,
        condition_latent: torch.Tensor | None = None,
        domain_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = [base_sr, bicubic.to(dtype=base_sr.dtype)]
        if self.use_condition_latent:
            if condition_latent is None:
                raise ValueError("condition_latent is required when use_condition_latent=True")
            condition_up = F.interpolate(
                condition_latent.float(),
                size=base_sr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=base_sr.dtype)
            inputs.append(condition_up)
        x = torch.cat(inputs, dim=1)
        x = self.input(x)
        if domain_id is not None:
            x = x + self.domain_embedding(domain_id).unsqueeze(-1).unsqueeze(-1)
        x = self.blocks(x)
        output = self.output(F.silu(self.output_norm(x)))
        residual_logits = output[:, : self.image_channels]
        gate_logits = output[:, self.image_channels :]
        raw_residual = self.residual_scale * torch.tanh(residual_logits)
        high_frequency_residual = metric_highpass(raw_residual, kernel_size=self.highpass_kernel)
        gate = torch.sigmoid(gate_logits + self.gate_bias)
        residual = high_frequency_residual * gate
        refined = (base_sr + residual).clamp(0.0, 1.0)
        return refined, residual, gate, raw_residual


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
        hflip_prob=data_config.get("hflip_prob", 0.0),
        texture_crop_retries=data_config.get("texture_crop_retries", 1),
        texture_crop_downsample=data_config.get("texture_crop_downsample", 128),
        hr_color_jitter_prob=data_config.get("hr_color_jitter_prob", 0.0),
        hr_color_jitter=data_config.get("hr_color_jitter", (0.97, 1.03)),
    )


def make_eval_loader(config: dict[str, Any], seed: int, device: torch.device) -> DataLoader:
    eval_config = config.get("eval", {})
    dataset = make_dataset(config, split=str(eval_config.get("split", "val")), seed=seed, deterministic=True)
    limit = int(eval_config.get("limit", 100))
    if limit > 0 and limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(
        dataset,
        batch_size=int(eval_config.get("batch_size", 4)),
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
    print(f"loaded_autoencoder={auto_cfg['checkpoint']} step={checkpoint.get('step', 'unknown')}", flush=True)
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
    print(f"loaded_condition_encoder={cond_cfg['checkpoint']} step={checkpoint.get('step', 'unknown')}", flush=True)
    return encoder


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
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": clean_config(config),
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def init_model_from_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    source_state = checkpoint["model"]
    target_state = model.state_dict()
    exact_tensors = 0
    partial_tensors = 0
    skipped_tensors = 0
    exact_params = 0
    partial_params = 0

    for key, source in source_state.items():
        target = target_state.get(key)
        if target is None:
            skipped_tensors += 1
            continue
        if tuple(source.shape) == tuple(target.shape):
            target_state[key] = source.to(device=device, dtype=target.dtype)
            exact_tensors += 1
            exact_params += int(source.numel())
            continue
        if key == "input.weight" and source.ndim == target.ndim == 4 and source.shape[0] == target.shape[0]:
            initialized = torch.zeros_like(target)
            out_channels = min(source.shape[0], target.shape[0])
            in_channels = min(source.shape[1], target.shape[1])
            kernel_h = min(source.shape[2], target.shape[2])
            kernel_w = min(source.shape[3], target.shape[3])
            initialized[:out_channels, :in_channels, :kernel_h, :kernel_w] = source[
                :out_channels, :in_channels, :kernel_h, :kernel_w
            ].to(device=device, dtype=target.dtype)
            target_state[key] = initialized
            partial_tensors += 1
            partial_params += int(out_channels * in_channels * kernel_h * kernel_w)
            continue
        skipped_tensors += 1

    model.load_state_dict(target_state)
    return {
        "checkpoint": str(path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "exact_tensors": exact_tensors,
        "partial_tensors": partial_tensors,
        "skipped_tensors": skipped_tensors,
        "exact_params": exact_params,
        "partial_params": partial_params,
    }


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
    os.environ["WANDB_MODE"] = str(wandb_cfg.get("mode", "online"))
    tags = list(wandb_cfg.get("tags") or [])
    if "detail-branch" not in tags:
        tags.append("detail-branch")
    run = wandb.init(
        project=wandb_cfg.get("project", "LuSIR"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", config.get("project", {}).get("name")),
        dir=str(wandb_dir),
        mode=wandb_cfg.get("mode", "online"),
        tags=tags,
        group=wandb_cfg.get("group", "stage-detail"),
        job_type=wandb_cfg.get("job_type", "detail-branch"),
        config=clean_config(config),
    )
    if bool(wandb_cfg.get("watch", False)):
        wandb.watch(model, log="gradients", log_freq=int(wandb_cfg.get("watch_log_freq", 200)))
    print(f"wandb_run={run.url}", flush=True)
    return run


def wandb_log(run: Any | None, data: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(data, step=step)


@torch.no_grad()
def make_base_prediction(
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    hr: torch.Tensor,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    device: torch.device,
    dtype_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lr_input = normalize_image(lr)
    with autocast_context(device, dtype_name):
        condition = condition_encoder(lr_input, domain_id)
        base_sr = denormalize(vae.decode(condition)).float()
    bicubic = F.interpolate(lr.float(), size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    return condition, base_sr, bicubic


@torch.no_grad()
def evaluate(
    model: GatedHighFrequencyDetailBranch,
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    output_dir: Path | None = None,
    sample_count: int = 0,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {
        "bicubic_mse": 0.0,
        "base_mse": 0.0,
        "sr_mse": 0.0,
        "bicubic_psnr": 0.0,
        "base_psnr": 0.0,
        "sr_psnr": 0.0,
        "base_ssim": 0.0,
        "sr_ssim": 0.0,
        "base_highpass_l1": 0.0,
        "sr_highpass_l1": 0.0,
        "base_laplacian_l1": 0.0,
        "sr_laplacian_l1": 0.0,
        "base_laplacian_ratio": 0.0,
        "sr_laplacian_ratio": 0.0,
        "base_highpass_ratio": 0.0,
        "sr_highpass_ratio": 0.0,
        "residual_l1": 0.0,
        "gate_mean": 0.0,
        "wins_vs_base": 0.0,
        "detail_wins_vs_base": 0.0,
    }
    count = 0
    grid_rows: list[list[tuple[str, Image.Image]]] = []
    for batch in dataloader:
        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
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
            sr, residual, gate, _ = model(base_sr, bicubic, condition, domain_id)
        sr = sr.float()
        residual = residual.float()
        gate = gate.float()
        batch_size = int(hr.shape[0])
        bicubic_mse_per = (bicubic - hr).square().flatten(1).mean(dim=1)
        base_mse_per = (base_sr - hr).square().flatten(1).mean(dim=1)
        sr_mse_per = (sr - hr).square().flatten(1).mean(dim=1)
        base_ssim_per = ssim_per_image(base_sr, hr)
        sr_ssim_per = ssim_per_image(sr, hr)
        target_high = metric_highpass(hr)
        base_high = metric_highpass(base_sr)
        sr_high = metric_highpass(sr)
        target_lap = laplacian_response(hr)
        base_lap = laplacian_response(base_sr)
        sr_lap = laplacian_response(sr)
        target_high_energy = target_high.abs().flatten(1).mean(dim=1).clamp_min(1e-12)
        target_lap_energy = target_lap.abs().flatten(1).mean(dim=1).clamp_min(1e-12)
        base_high_l1_per = (base_high - target_high).abs().flatten(1).mean(dim=1)
        sr_high_l1_per = (sr_high - target_high).abs().flatten(1).mean(dim=1)
        base_lap_l1_per = (base_lap - target_lap).abs().flatten(1).mean(dim=1)
        sr_lap_l1_per = (sr_lap - target_lap).abs().flatten(1).mean(dim=1)
        totals["bicubic_mse"] += float(bicubic_mse_per.sum().cpu())
        totals["base_mse"] += float(base_mse_per.sum().cpu())
        totals["sr_mse"] += float(sr_mse_per.sum().cpu())
        totals["bicubic_psnr"] += float((-10.0 * torch.log10(bicubic_mse_per.clamp_min(1e-12))).sum().cpu())
        totals["base_psnr"] += float((-10.0 * torch.log10(base_mse_per.clamp_min(1e-12))).sum().cpu())
        totals["sr_psnr"] += float((-10.0 * torch.log10(sr_mse_per.clamp_min(1e-12))).sum().cpu())
        totals["base_ssim"] += float(base_ssim_per.sum().cpu())
        totals["sr_ssim"] += float(sr_ssim_per.sum().cpu())
        totals["base_highpass_l1"] += float(base_high_l1_per.sum().cpu())
        totals["sr_highpass_l1"] += float(sr_high_l1_per.sum().cpu())
        totals["base_laplacian_l1"] += float(base_lap_l1_per.sum().cpu())
        totals["sr_laplacian_l1"] += float(sr_lap_l1_per.sum().cpu())
        totals["base_laplacian_ratio"] += float((base_lap.abs().flatten(1).mean(dim=1) / target_lap_energy).sum().cpu())
        totals["sr_laplacian_ratio"] += float((sr_lap.abs().flatten(1).mean(dim=1) / target_lap_energy).sum().cpu())
        totals["base_highpass_ratio"] += float((base_high.abs().flatten(1).mean(dim=1) / target_high_energy).sum().cpu())
        totals["sr_highpass_ratio"] += float((sr_high.abs().flatten(1).mean(dim=1) / target_high_energy).sum().cpu())
        totals["residual_l1"] += float(residual.abs().mean().cpu()) * batch_size
        totals["gate_mean"] += float(gate.mean().cpu()) * batch_size
        totals["wins_vs_base"] += float((sr_mse_per < base_mse_per).float().sum().cpu())
        totals["detail_wins_vs_base"] += float((sr_lap_l1_per < base_lap_l1_per).float().sum().cpu())
        if output_dir is not None and len(grid_rows) < sample_count:
            lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)
            residual_vis = (residual / max(float(model.residual_scale), 1e-6) * 0.5 + 0.5).clamp(0.0, 1.0)
            for item_idx in range(batch_size):
                if len(grid_rows) >= sample_count:
                    break
                grid_rows.append(
                    [
                        ("LR", tensor_to_pil(lr_nearest[item_idx])),
                        ("bicubic", tensor_to_pil(bicubic[item_idx])),
                        ("base", tensor_to_pil(base_sr[item_idx])),
                        ("detail", tensor_to_pil(sr[item_idx])),
                        ("residual", tensor_to_pil(residual_vis[item_idx])),
                        ("GT", tensor_to_pil(hr[item_idx])),
                    ]
                )
        count += batch_size
    count = max(1, count)
    metrics = {
        "eval/bicubic_mse": totals["bicubic_mse"] / count,
        "eval/base_mse": totals["base_mse"] / count,
        "eval/sr_mse": totals["sr_mse"] / count,
        "eval/bicubic_mean_psnr": totals["bicubic_psnr"] / count,
        "eval/base_mean_psnr": totals["base_psnr"] / count,
        "eval/sr_mean_psnr": totals["sr_psnr"] / count,
        "eval/base_ssim": totals["base_ssim"] / count,
        "eval/sr_ssim": totals["sr_ssim"] / count,
        "eval/base_highpass_l1": totals["base_highpass_l1"] / count,
        "eval/sr_highpass_l1": totals["sr_highpass_l1"] / count,
        "eval/base_laplacian_l1": totals["base_laplacian_l1"] / count,
        "eval/sr_laplacian_l1": totals["sr_laplacian_l1"] / count,
        "eval/base_laplacian_ratio": totals["base_laplacian_ratio"] / count,
        "eval/sr_laplacian_ratio": totals["sr_laplacian_ratio"] / count,
        "eval/base_highpass_ratio": totals["base_highpass_ratio"] / count,
        "eval/sr_highpass_ratio": totals["sr_highpass_ratio"] / count,
        "eval/residual_l1": totals["residual_l1"] / count,
        "eval/gate_mean": totals["gate_mean"] / count,
        "eval/wins_vs_base": totals["wins_vs_base"],
        "eval/detail_wins_vs_base": totals["detail_wins_vs_base"],
        "eval/num_images": float(count),
    }
    metrics["eval/bicubic_psnr"] = psnr_from_mse(metrics["eval/bicubic_mse"])
    metrics["eval/base_psnr"] = psnr_from_mse(metrics["eval/base_mse"])
    metrics["eval/sr_psnr"] = psnr_from_mse(metrics["eval/sr_mse"])
    metrics["eval/sr_vs_base_psnr"] = metrics["eval/sr_psnr"] - metrics["eval/base_psnr"]
    metrics["eval/sr_vs_base_mean_psnr"] = metrics["eval/sr_mean_psnr"] - metrics["eval/base_mean_psnr"]
    metrics["eval/sr_vs_base_ssim"] = metrics["eval/sr_ssim"] - metrics["eval/base_ssim"]
    metrics["eval/sr_vs_base_laplacian_l1"] = metrics["eval/base_laplacian_l1"] - metrics["eval/sr_laplacian_l1"]
    metrics["eval/sr_vs_base_highpass_l1"] = metrics["eval/base_highpass_l1"] - metrics["eval/sr_highpass_l1"]
    metrics["eval/sr_vs_base_laplacian_ratio"] = metrics["eval/sr_laplacian_ratio"] - metrics["eval/base_laplacian_ratio"]
    metrics["eval/sr_vs_base_highpass_ratio"] = metrics["eval/sr_highpass_ratio"] - metrics["eval/base_highpass_ratio"]
    metrics["eval/detail_score"] = (
        metrics["eval/sr_mean_psnr"]
        + 0.25 * metrics["eval/sr_vs_base_laplacian_ratio"]
        + 0.25 * metrics["eval/sr_vs_base_highpass_ratio"]
        + 0.5 * metrics["eval/sr_vs_base_ssim"]
    )
    if output_dir is not None and grid_rows:
        make_grid(grid_rows, output_dir / "eval_grid_lr_bicubic_base_detail_residual_gt.png")
    if was_training:
        model.train()
    return metrics


def training_loss(
    model: GatedHighFrequencyDetailBranch,
    base_sr: torch.Tensor,
    bicubic: torch.Tensor,
    condition: torch.Tensor,
    hr: torch.Tensor,
    domain_id: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    eps = float(loss_cfg.get("charbonnier_eps", 1e-3))
    highpass_kernel = int(loss_cfg.get("highpass_kernel", 15))
    lowpass_kernel = int(loss_cfg.get("lowpass_kernel", 31))
    sr, residual, gate, raw_residual = model(base_sr, bicubic, condition, domain_id)
    target_residual = metric_highpass(hr - base_sr.detach(), kernel_size=highpass_kernel)
    image_loss = charbonnier(sr, hr, eps)
    residual_target_loss = charbonnier(residual, target_residual, eps)
    highpass_loss = charbonnier(metric_highpass(sr, kernel_size=highpass_kernel), metric_highpass(hr, kernel_size=highpass_kernel), eps)
    laplacian_loss = charbonnier(laplacian_response(sr), laplacian_response(hr), eps)
    lowpass_anchor_loss = charbonnier(lowpass(sr, lowpass_kernel), lowpass(base_sr.detach(), lowpass_kernel), eps)
    gate_l1 = gate.float().mean()
    residual_l1 = raw_residual.float().abs().mean()
    loss = (
        float(loss_cfg.get("image_weight", 1.0)) * image_loss
        + float(loss_cfg.get("residual_target_weight", 0.5)) * residual_target_loss
        + float(loss_cfg.get("highpass_weight", 1.0)) * highpass_loss
        + float(loss_cfg.get("laplacian_weight", 0.5)) * laplacian_loss
        + float(loss_cfg.get("lowpass_anchor_weight", 1.0)) * lowpass_anchor_loss
        + float(loss_cfg.get("gate_l1_weight", 0.001)) * gate_l1
        + float(loss_cfg.get("residual_l1_weight", 0.001)) * residual_l1
    )
    return loss, {
        "image": image_loss,
        "residual_target": residual_target_loss,
        "highpass": highpass_loss,
        "laplacian": laplacian_loss,
        "lowpass_anchor": lowpass_anchor_loss,
        "gate": gate_l1,
        "residual_l1": residual_l1,
        "sr": sr,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["project"]["output_dir"] = str(args.output_dir)
    if args.disable_wandb:
        config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
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
    model = GatedHighFrequencyDetailBranch.from_config(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    init_cfg = config.get("initialization", {})
    if init_cfg.get("checkpoint"):
        init_stats = init_model_from_checkpoint(Path(init_cfg["checkpoint"]), model, device)
        print(f"model_init={json.dumps(init_stats, sort_keys=True)}", flush=True)
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed={args.resume} step={start_step}", flush=True)
    run = init_wandb(config, output_dir, model)

    train_dataset = make_dataset(config, split=str(config["data"].get("split", "train")), seed=seed, deterministic=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    eval_loader = make_eval_loader(config, seed=seed, device=device)

    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    loss_cfg = config.get("loss", {})
    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 12000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every", 25))
    save_every = int(train_cfg.get("save_every", 2000))
    eval_every = int(eval_cfg.get("every", 500))
    sample_count = int(eval_cfg.get("sample_count", 8))
    best_metric_name = str(eval_cfg.get("best_metric", "eval/detail_score"))
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
            f"eval step={step} sr_psnr={metrics['eval/sr_psnr']:.4f} "
            f"base_psnr={metrics['eval/base_psnr']:.4f} "
            f"delta={metrics['eval/sr_vs_base_psnr']:+.4f} "
            f"mean_delta={metrics['eval/sr_vs_base_mean_psnr']:+.4f} "
            f"ssim_delta={metrics['eval/sr_vs_base_ssim']:+.5f} "
            f"hp_delta={metrics['eval/sr_vs_base_highpass_ratio']:+.4f} "
            f"lap_delta={metrics['eval/sr_vs_base_laplacian_ratio']:+.4f} "
            f"wins={metrics['eval/wins_vs_base']:.0f}/{metrics['eval/num_images']:.0f}",
            flush=True,
        )
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
        wandb_data: dict[str, Any] = dict(metrics)
        grid_path = eval_dir / "eval_grid_lr_bicubic_base_detail_residual_gt.png"
        if run is not None and grid_path.exists():
            import wandb

            wandb_data["samples/eval_grid"] = wandb.Image(str(grid_path), caption=f"eval step {step}")
        wandb_log(run, wandb_data, step=step)
        return metrics

    if args.eval_only_checkpoint is not None:
        checkpoint_step = load_checkpoint(args.eval_only_checkpoint, model, optimizer, device)
        metrics = run_eval(checkpoint_step)
        summary = {"config": str(args.config), "checkpoint": str(args.eval_only_checkpoint), "checkpoint_step": checkpoint_step, "metrics": metrics}
        (output_dir / f"eval_only_step_{checkpoint_step:06d}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if run is not None:
            run.finish()
        return

    if bool(eval_cfg.get("run_at_start", True)) and start_step == 0:
        metrics = run_eval(0)
        best_metric = float(metrics[best_metric_name])
        best_metrics = metrics
        save_checkpoint(checkpoints_dir / "best_eval_detail.pt", model, optimizer, 0, config, metrics)

    step = start_step
    optimizer_updates = start_step // max(grad_accum_steps, 1)
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    last_log_time = time.time()
    last_log_step = step
    model.train()
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
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
            loss, loss_parts = training_loss(
                model=model,
                base_sr=base_sr.detach(),
                bicubic=bicubic,
                condition=condition.detach(),
                hr=hr,
                domain_id=domain_id,
                loss_cfg=loss_cfg,
            )
        (loss / grad_accum_steps).backward()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
        step += 1

        if step % log_every == 0 or step == 1:
            elapsed = max(time.time() - last_log_time, 1e-6)
            last_log_time = time.time()
            logged_steps = max(step - last_log_step, 1)
            last_log_step = step
            steps_per_s = logged_steps / elapsed
            train_metrics = {
                "train/loss": float(loss.detach().cpu()),
                "train/image": float(loss_parts["image"].detach().cpu()),
                "train/residual_target": float(loss_parts["residual_target"].detach().cpu()),
                "train/highpass": float(loss_parts["highpass"].detach().cpu()),
                "train/laplacian": float(loss_parts["laplacian"].detach().cpu()),
                "train/lowpass_anchor": float(loss_parts["lowpass_anchor"].detach().cpu()),
                "train/gate": float(loss_parts["gate"].detach().cpu()),
                "train/residual_l1": float(loss_parts["residual_l1"].detach().cpu()),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/optimizer_updates": float(optimizer_updates),
                "system/steps_per_s": steps_per_s,
            }
            print(
                f"step={step} loss={train_metrics['train/loss']:.5f} "
                f"image={train_metrics['train/image']:.5f} "
                f"residual={train_metrics['train/residual_target']:.5f} "
                f"highpass={train_metrics['train/highpass']:.5f} "
                f"lap={train_metrics['train/laplacian']:.5f} "
                f"low_anchor={train_metrics['train/lowpass_anchor']:.5f} "
                f"gate={train_metrics['train/gate']:.5f} "
                f"updates={optimizer_updates} "
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
                save_checkpoint(checkpoints_dir / "best_eval_detail.pt", model, optimizer, step, config, metrics)
            model.train()

    final_metrics = run_eval(step)
    save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config, final_metrics)
    final_metric_value = float(final_metrics[best_metric_name])
    final_improved = final_metric_value > best_metric if best_mode == "max" else final_metric_value < best_metric
    if final_improved:
        best_metric = final_metric_value
        best_metrics = final_metrics
        save_checkpoint(checkpoints_dir / "best_eval_detail.pt", model, optimizer, step, config, final_metrics)

    summary = {
        "config": str(args.config),
        "output_dir": str(output_dir),
        "finished_step": step,
        "best_metric_name": best_metric_name,
        "best_metric_mode": best_mode,
        "best_metric_value": best_metric,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "checkpoint_latest": str(checkpoints_dir / "latest.pt"),
        "checkpoint_best": str(checkpoints_dir / "best_eval_detail.pt"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
