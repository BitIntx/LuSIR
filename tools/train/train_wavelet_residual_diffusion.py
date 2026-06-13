from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.models import ConditionalUNet, NoiseScheduler
from sr_diffusion.utils import autocast_context, get_device, load_config, save_config, seed_everything, seed_worker
from sr_diffusion.wavelet import haar_dwt2, haar_high_bands, image_from_haar_high
from tools.train.train_detail_branch import (
    GatedHighFrequencyDetailBranch,
    load_autoencoder,
    load_checkpoint as load_detail_checkpoint,
    load_condition_encoder,
    make_base_prediction,
    make_dataset,
)
from tools.train.train_residual_refiner import (
    charbonnier,
    clean_config,
    laplacian_response,
    lowpass,
    make_grid,
    metric_highpass,
    ssim_per_image,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion over signed Haar high-frequency residuals.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_detail_branch(config: dict[str, Any], device: torch.device) -> GatedHighFrequencyDetailBranch:
    detail_cfg = config["detail_branch"]
    detail_config = load_config(detail_cfg["config"])
    model = GatedHighFrequencyDetailBranch.from_config(detail_config["model"]).to(device)
    step = load_detail_checkpoint(Path(detail_cfg["checkpoint"]), model, optimizer=None, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"loaded_detail_branch={detail_cfg['checkpoint']} step={step}", flush=True)
    return model


@torch.no_grad()
def make_detail_prediction(
    vae: nn.Module,
    condition_encoder: nn.Module,
    detail_branch: GatedHighFrequencyDetailBranch,
    hr: torch.Tensor,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    device: torch.device,
    dtype_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    condition_latent, base_sr, bicubic = make_base_prediction(
        vae=vae,
        condition_encoder=condition_encoder,
        hr=hr,
        lr=lr,
        domain_id=domain_id,
        device=device,
        dtype_name=dtype_name,
    )
    with autocast_context(device, dtype_name):
        detail_sr, _, _, _ = detail_branch(base_sr, bicubic, condition_latent, domain_id)
    return detail_sr.float(), base_sr.float(), bicubic.float()


def make_wavelet_condition(detail_sr: torch.Tensor, bicubic: torch.Tensor) -> torch.Tensor:
    detail_coefficients = haar_dwt2(detail_sr.mul(2.0).sub(1.0))
    bicubic_coefficients = haar_dwt2(bicubic.mul(2.0).sub(1.0))
    return torch.cat([detail_coefficients, bicubic_coefficients], dim=1)


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    gathered = values.to(device=timesteps.device).gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


@torch.no_grad()
def ddim_sample(
    model: ConditionalUNet,
    scheduler: NoiseScheduler,
    condition: torch.Tensor,
    domain_id: torch.Tensor,
    steps: int,
    seed: int,
    clip_x0: float,
    start_timestep: int,
) -> torch.Tensor:
    generator = torch.Generator(device=condition.device)
    generator.manual_seed(int(seed))
    batch_size = int(condition.shape[0])
    residual_channels = int(model.latent_channels)
    noise = torch.randn(
        batch_size,
        residual_channels,
        *condition.shape[-2:],
        generator=generator,
        device=condition.device,
        dtype=torch.float32,
    )
    start_timestep = max(0, min(int(start_timestep), scheduler.num_train_timesteps - 1))
    start_alpha = scheduler.alphas_cumprod[start_timestep].to(device=condition.device, dtype=torch.float32)
    sample = (1.0 - start_alpha).sqrt() * noise
    timestep_values = torch.linspace(
        start_timestep,
        0,
        steps=max(2, int(steps)),
        device=condition.device,
    ).round().long()
    timestep_values = torch.unique_consecutive(timestep_values)
    alphas = scheduler.alphas_cumprod.to(device=condition.device, dtype=torch.float32)
    for index, timestep_value in enumerate(timestep_values):
        timestep = torch.full((batch_size,), int(timestep_value), device=condition.device, dtype=torch.long)
        predicted_noise = model(sample, timestep, condition, domain_id).float()
        alpha = extract(alphas, timestep, sample.shape)
        predicted_x0 = (sample - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-8)
        predicted_x0 = predicted_x0.clamp(-float(clip_x0), float(clip_x0))
        if index + 1 == len(timestep_values):
            sample = predicted_x0
            continue
        next_timestep = torch.full(
            (batch_size,),
            int(timestep_values[index + 1]),
            device=condition.device,
            dtype=torch.long,
        )
        next_alpha = extract(alphas, next_timestep, sample.shape)
        sample = next_alpha.sqrt() * predicted_x0 + (1.0 - next_alpha).sqrt() * predicted_noise
    return sample


def sample_train_timesteps(
    scheduler: NoiseScheduler,
    batch_size: int,
    device: torch.device,
    diffusion_config: dict[str, Any],
) -> torch.Tensor:
    minimum = max(0, int(diffusion_config.get("train_min_timestep", 0)))
    maximum = min(
        scheduler.num_train_timesteps - 1,
        int(diffusion_config.get("train_max_timestep", scheduler.num_train_timesteps - 1)),
    )
    if maximum < minimum:
        raise ValueError(f"train_max_timestep must be >= train_min_timestep, got {maximum} < {minimum}")
    return torch.randint(minimum, maximum + 1, (batch_size,), device=device, dtype=torch.long)


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


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def init_wandb(config: dict[str, Any], output_dir: Path, model: nn.Module) -> Any | None:
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    import wandb

    wandb_dir = Path(wandb_cfg.get("dir", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = str(wandb_cfg.get("mode", "online"))
    run = wandb.init(
        project=wandb_cfg.get("project", "LuSIR"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", config["project"]["name"]),
        dir=str(wandb_dir),
        mode=wandb_cfg.get("mode", "online"),
        tags=list(wandb_cfg.get("tags") or []),
        group=wandb_cfg.get("group", "stage-detail-diffusion"),
        job_type=wandb_cfg.get("job_type", "wavelet-residual-diffusion"),
        config=clean_config(config),
    )
    print(f"wandb_run={run.url}", flush=True)
    return run


def wandb_log(run: Any | None, data: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def make_eval_loader(config: dict[str, Any], seed: int, device: torch.device) -> DataLoader:
    eval_cfg = config.get("eval", {})
    dataset = make_dataset(config, split=str(eval_cfg.get("split", "val")), seed=seed, deterministic=True)
    limit = int(eval_cfg.get("limit", 8))
    if 0 < limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: ConditionalUNet,
    scheduler: NoiseScheduler,
    vae: nn.Module,
    condition_encoder: nn.Module,
    detail_branch: GatedHighFrequencyDetailBranch,
    dataloader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    dtype_name: str,
    output_dir: Path,
    step: int,
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()
    eval_cfg = config.get("eval", {})
    diffusion_cfg = config.get("diffusion", {})
    residual_scale = float(diffusion_cfg.get("residual_scale", 0.08))
    clip_x0 = float(diffusion_cfg.get("clip_x0", 4.0))
    sample_steps = int(eval_cfg.get("sample_steps", 12))
    start_timestep = int(eval_cfg.get("start_timestep", scheduler.num_train_timesteps - 1))
    seeds = [int(value) for value in eval_cfg.get("seeds", [123, 456, 789])]
    totals = {
        "base_psnr": 0.0,
        "base_ssim": 0.0,
        "base_laplacian_l1": 0.0,
        "base_highpass_l1": 0.0,
        "sr_psnr": 0.0,
        "sr_ssim": 0.0,
        "sr_laplacian_l1": 0.0,
        "sr_highpass_l1": 0.0,
        "signed_wavelet_l1": 0.0,
        "lowpass_drift": 0.0,
        "residual_l1": 0.0,
        "residual_energy_ratio": 0.0,
        "diversity_l1": 0.0,
    }
    image_count = 0
    sampled_count = 0
    grid_rows: list[list[tuple[str, Any]]] = []
    sample_count = int(eval_cfg.get("sample_count", 4))
    for batch_index, batch in enumerate(dataloader):
        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        detail_sr, _, bicubic = make_detail_prediction(
            vae, condition_encoder, detail_branch, hr, lr, domain_id, device, dtype_name
        )
        wavelet_condition = make_wavelet_condition(detail_sr, bicubic)
        target_high = haar_high_bands(hr - detail_sr)
        batch_size = int(hr.shape[0])
        base_mse = (detail_sr - hr).square().flatten(1).mean(dim=1)
        base_ssim = ssim_per_image(detail_sr, hr)
        base_lap = (laplacian_response(detail_sr) - laplacian_response(hr)).abs().flatten(1).mean(dim=1)
        base_high = (metric_highpass(detail_sr) - metric_highpass(hr)).abs().flatten(1).mean(dim=1)
        totals["base_psnr"] += float((-10.0 * torch.log10(base_mse.clamp_min(1e-12))).sum().cpu())
        totals["base_ssim"] += float(base_ssim.sum().cpu())
        totals["base_laplacian_l1"] += float(base_lap.sum().cpu())
        totals["base_highpass_l1"] += float(base_high.sum().cpu())
        sampled_sr: list[torch.Tensor] = []
        sampled_high: list[torch.Tensor] = []
        for seed in seeds:
            with autocast_context(device, dtype_name):
                normalized_high = ddim_sample(
                    model=model,
                    scheduler=scheduler,
                    condition=wavelet_condition,
                    domain_id=domain_id,
                    steps=sample_steps,
                    seed=seed + batch_index * 100_000,
                    clip_x0=clip_x0,
                    start_timestep=start_timestep,
                )
            high = normalized_high.float() * residual_scale
            residual = image_from_haar_high(high, channels=3)
            sr = (detail_sr + residual).clamp(0.0, 1.0)
            sampled_sr.append(sr)
            sampled_high.append(high)
            sr_mse = (sr - hr).square().flatten(1).mean(dim=1)
            sr_ssim = ssim_per_image(sr, hr)
            sr_lap = (laplacian_response(sr) - laplacian_response(hr)).abs().flatten(1).mean(dim=1)
            sr_high = (metric_highpass(sr) - metric_highpass(hr)).abs().flatten(1).mean(dim=1)
            signed_wavelet = (high - target_high).abs().flatten(1).mean(dim=1)
            lowpass_drift = (lowpass(sr, 31) - lowpass(detail_sr, 31)).abs().flatten(1).mean(dim=1)
            target_energy = target_high.abs().flatten(1).mean(dim=1).clamp_min(1e-8)
            residual_energy_ratio = high.abs().flatten(1).mean(dim=1) / target_energy
            totals["sr_psnr"] += float((-10.0 * torch.log10(sr_mse.clamp_min(1e-12))).sum().cpu())
            totals["sr_ssim"] += float(sr_ssim.sum().cpu())
            totals["sr_laplacian_l1"] += float(sr_lap.sum().cpu())
            totals["sr_highpass_l1"] += float(sr_high.sum().cpu())
            totals["signed_wavelet_l1"] += float(signed_wavelet.sum().cpu())
            totals["lowpass_drift"] += float(lowpass_drift.sum().cpu())
            totals["residual_l1"] += float(residual.abs().flatten(1).mean(dim=1).sum().cpu())
            totals["residual_energy_ratio"] += float(residual_energy_ratio.sum().cpu())
            sampled_count += batch_size
        pairwise = []
        for left in range(len(sampled_sr)):
            for right in range(left + 1, len(sampled_sr)):
                pairwise.append((sampled_sr[left] - sampled_sr[right]).abs().flatten(1).mean(dim=1))
        if pairwise:
            diversity = torch.stack(pairwise).mean(dim=0)
            totals["diversity_l1"] += float(diversity.sum().cpu())
        lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)
        for item_index in range(batch_size):
            if len(grid_rows) >= sample_count:
                break
            row: list[tuple[str, Any]] = [
                ("LR", tensor_to_pil(lr_nearest[item_index])),
                ("v1d base", tensor_to_pil(detail_sr[item_index])),
            ]
            row.extend(
                (f"seed {seed}", tensor_to_pil(sampled_sr[seed_index][item_index]))
                for seed_index, seed in enumerate(seeds)
            )
            row.append(("GT", tensor_to_pil(hr[item_index])))
            grid_rows.append(row)
        image_count += batch_size
    image_count = max(1, image_count)
    sampled_count = max(1, sampled_count)
    metrics = {
        "eval/base_psnr": totals["base_psnr"] / image_count,
        "eval/base_ssim": totals["base_ssim"] / image_count,
        "eval/base_laplacian_l1": totals["base_laplacian_l1"] / image_count,
        "eval/base_highpass_l1": totals["base_highpass_l1"] / image_count,
        "eval/sr_psnr": totals["sr_psnr"] / sampled_count,
        "eval/sr_ssim": totals["sr_ssim"] / sampled_count,
        "eval/sr_laplacian_l1": totals["sr_laplacian_l1"] / sampled_count,
        "eval/sr_highpass_l1": totals["sr_highpass_l1"] / sampled_count,
        "eval/signed_wavelet_l1": totals["signed_wavelet_l1"] / sampled_count,
        "eval/lowpass_drift": totals["lowpass_drift"] / sampled_count,
        "eval/residual_l1": totals["residual_l1"] / sampled_count,
        "eval/residual_energy_ratio": totals["residual_energy_ratio"] / sampled_count,
        "eval/diversity_l1": totals["diversity_l1"] / image_count,
        "eval/num_images": float(image_count),
        "eval/num_seeds": float(len(seeds)),
    }
    metrics["eval/psnr_delta_vs_base"] = metrics["eval/sr_psnr"] - metrics["eval/base_psnr"]
    metrics["eval/ssim_delta_vs_base"] = metrics["eval/sr_ssim"] - metrics["eval/base_ssim"]
    metrics["eval/laplacian_gain_vs_base"] = metrics["eval/base_laplacian_l1"] - metrics["eval/sr_laplacian_l1"]
    metrics["eval/highpass_gain_vs_base"] = metrics["eval/base_highpass_l1"] - metrics["eval/sr_highpass_l1"]
    eval_dir = output_dir / f"eval_step_{step:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if grid_rows:
        make_grid(grid_rows, eval_dir / "grid_lr_v1d_seed1_seed2_seed3_gt.png")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if model_was_training:
        model.train()
    return metrics


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
    detail_branch = load_detail_branch(config, device)
    model = ConditionalUNet.from_config(config["model"]).to(device)
    scheduler = NoiseScheduler.from_config(config["diffusion"])
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    diffusion_cfg = config.get("diffusion", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    start_step = 0
    if args.resume is not None:
        start_step = load_training_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed={args.resume} step={start_step}", flush=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} dtype={dtype_name} model_params={parameter_count} "
        f"diffusion_timesteps={scheduler.num_train_timesteps}",
        flush=True,
    )
    run = init_wandb(config, output_dir, model)

    train_dataset = make_dataset(config, split=str(config["data"].get("split", "train")), seed=seed, deterministic=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    eval_loader = make_eval_loader(config, seed=seed, device=device)
    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 2000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 8))
    log_every = int(train_cfg.get("log_every", 25))
    save_every = int(train_cfg.get("save_every", 500))
    eval_cfg = config.get("eval", {})
    eval_every = int(eval_cfg.get("every", 250))
    residual_scale = float(diffusion_cfg.get("residual_scale", 0.08))
    clip_x0 = float(diffusion_cfg.get("clip_x0", 4.0))
    best_metric_name = str(eval_cfg.get("best_metric", "eval/signed_wavelet_l1"))
    best_mode = str(eval_cfg.get("best_mode", "min"))
    best_metric = float("inf") if best_mode == "min" else float("-inf")
    metrics_log_path = output_dir / "metrics.jsonl"

    def run_eval(step: int) -> dict[str, float]:
        metrics = evaluate(
            model=model,
            scheduler=scheduler,
            vae=vae,
            condition_encoder=condition_encoder,
            detail_branch=detail_branch,
            dataloader=eval_loader,
            config=config,
            device=device,
            dtype_name=dtype_name,
            output_dir=output_dir,
            step=step,
        )
        print(
            f"eval step={step} sr_psnr={metrics['eval/sr_psnr']:.4f} "
            f"base_psnr={metrics['eval/base_psnr']:.4f} "
            f"delta={metrics['eval/psnr_delta_vs_base']:+.4f} "
            f"wavelet_l1={metrics['eval/signed_wavelet_l1']:.5f} "
            f"lap_gain={metrics['eval/laplacian_gain_vs_base']:+.6f} "
            f"hp_gain={metrics['eval/highpass_gain_vs_base']:+.6f} "
            f"low_drift={metrics['eval/lowpass_drift']:.6f} "
            f"diversity={metrics['eval/diversity_l1']:.6f}",
            flush=True,
        )
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
        wandb_data: dict[str, Any] = dict(metrics)
        grid_path = output_dir / f"eval_step_{step:06d}" / "grid_lr_v1d_seed1_seed2_seed3_gt.png"
        if run is not None and grid_path.exists():
            import wandb

            wandb_data["samples/eval_grid"] = wandb.Image(str(grid_path), caption=f"wavelet diffusion step {step}")
        wandb_log(run, wandb_data, step=step)
        return metrics

    step = start_step
    optimizer_updates = start_step // max(1, grad_accum_steps)
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
            detail_sr, _, bicubic = make_detail_prediction(
                vae, condition_encoder, detail_branch, hr, lr, domain_id, device, dtype_name
            )
            wavelet_condition = make_wavelet_condition(detail_sr, bicubic)
            target_high = (haar_high_bands(hr - detail_sr) / residual_scale).clamp(-clip_x0, clip_x0)
            noise = torch.randn_like(target_high)
            timesteps = sample_train_timesteps(
                scheduler,
                int(hr.shape[0]),
                device=device,
                diffusion_config=diffusion_cfg,
            )
            noisy_high = scheduler.add_noise(target_high, noise, timesteps)
        with autocast_context(device, dtype_name):
            predicted_noise = model(noisy_high, timesteps, wavelet_condition, domain_id)
            predicted_x0 = scheduler.predict_x0_from_noise(noisy_high, timesteps, predicted_noise)
            predicted_x0_clipped = predicted_x0.clamp(-clip_x0, clip_x0)
            predicted_residual = image_from_haar_high(predicted_x0_clipped * residual_scale, channels=3)
            predicted_sr = detail_sr + predicted_residual
            noise_loss = F.mse_loss(predicted_noise, noise)
            x0_loss = charbonnier(predicted_x0_clipped, target_high, float(loss_cfg.get("charbonnier_eps", 1e-3)))
            image_loss = charbonnier(predicted_sr, hr, float(loss_cfg.get("charbonnier_eps", 1e-3)))
            laplacian_loss = charbonnier(
                laplacian_response(predicted_sr),
                laplacian_response(hr),
                float(loss_cfg.get("charbonnier_eps", 1e-3)),
            )
            loss = (
                float(loss_cfg.get("noise_weight", 1.0)) * noise_loss
                + float(loss_cfg.get("x0_weight", 0.1)) * x0_loss
                + float(loss_cfg.get("image_weight", 0.05)) * image_loss
                + float(loss_cfg.get("laplacian_weight", 0.1)) * laplacian_loss
            )
        (loss / grad_accum_steps).backward()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
        step += 1
        if step % log_every == 0 or step == 1:
            elapsed = max(time.time() - last_log_time, 1e-6)
            interval = max(step - last_log_step, 1)
            last_log_time = time.time()
            last_log_step = step
            train_metrics = {
                "train/loss": float(loss.detach().cpu()),
                "train/noise_mse": float(noise_loss.detach().cpu()),
                "train/signed_x0_charbonnier": float(x0_loss.detach().cpu()),
                "train/image_charbonnier": float(image_loss.detach().cpu()),
                "train/laplacian_charbonnier": float(laplacian_loss.detach().cpu()),
                "train/optimizer_updates": float(optimizer_updates),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "system/steps_per_s": interval / elapsed,
            }
            print(
                f"step={step} loss={train_metrics['train/loss']:.5f} "
                f"noise={train_metrics['train/noise_mse']:.5f} "
                f"x0={train_metrics['train/signed_x0_charbonnier']:.5f} "
                f"image={train_metrics['train/image_charbonnier']:.5f} "
                f"lap={train_metrics['train/laplacian_charbonnier']:.5f} "
                f"updates={optimizer_updates} steps_per_s={train_metrics['system/steps_per_s']:.3f}",
                flush=True,
            )
            wandb_log(run, train_metrics, step=step)
        if step % save_every == 0:
            save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, optimizer, step, config)
            save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config)
        if eval_every > 0 and step % eval_every == 0:
            metrics = run_eval(step)
            metric = float(metrics[best_metric_name])
            improved = metric < best_metric if best_mode == "min" else metric > best_metric
            if improved:
                best_metric = metric
                save_checkpoint(checkpoints_dir / "best_eval.pt", model, optimizer, step, config, metrics)
            model.train()

    final_metrics = None
    if not args.skip_final_eval and (eval_every <= 0 or step % eval_every != 0):
        final_metrics = run_eval(step)
    save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config, final_metrics)
    summary = {
        "config": str(args.config),
        "output_dir": str(output_dir),
        "finished_step": step,
        "optimizer_updates": optimizer_updates,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "final_metrics": final_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
