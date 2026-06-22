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
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.models import ConditionalUNet
from sr_diffusion.residual_shift import (
    apply_masked_correction,
    masked_latent_target,
    residual_shift_eta,
    residual_shift_forward_sample,
    residual_shift_step,
)
from sr_diffusion.utils import autocast_context, get_device, load_config, save_config, seed_everything, seed_worker
from tools.train.train_detail_branch import (
    apply_detail_mask_policy,
    load_autoencoder,
    load_condition_encoder,
    load_detail_mask_predictor,
    make_base_prediction,
    make_dataset,
    make_perceptual_model,
    masked_charbonnier,
)
from tools.train.train_latent_pretrain import denormalize, normalize_image
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
    parser = argparse.ArgumentParser(description="Train masked residual-shift diffusion in Stage 1 latent space.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    return parser.parse_args()


def bounded_correction(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    maximum = float(maximum)
    if maximum <= 0.0:
        return raw
    return maximum * torch.tanh(raw / maximum)


def make_condition(base_latent: torch.Tensor, lr: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
    lr_condition = normalize_image(lr.float())
    if lr_condition.shape[-2:] != base_latent.shape[-2:]:
        lr_condition = F.interpolate(lr_condition, size=base_latent.shape[-2:], mode="bicubic", align_corners=False)
    return torch.cat([base_latent.float(), lr_condition, latent_mask.float()], dim=1)


@torch.no_grad()
def prepare_frozen_batch(
    vae: nn.Module,
    condition_encoder: nn.Module,
    mask_predictor: nn.Module,
    hr: torch.Tensor,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    mask_cfg: dict[str, Any],
    device: torch.device,
    dtype_name: str,
) -> dict[str, torch.Tensor]:
    base_latent, base_sr, bicubic = make_base_prediction(
        vae=vae,
        condition_encoder=condition_encoder,
        hr=hr,
        lr=lr,
        domain_id=domain_id,
        device=device,
        dtype_name=dtype_name,
    )
    with autocast_context(device, dtype_name):
        detail_mask = mask_predictor(base_sr, bicubic, base_latent, domain_id)
    detail_mask = apply_detail_mask_policy(detail_mask, mask_cfg).float()
    latent_mask = F.interpolate(detail_mask, size=base_latent.shape[-2:], mode="area").clamp(0.0, 1.0)
    with autocast_context(device, dtype_name):
        target_latent, _ = vae.encode(normalize_image(hr))
    base_latent = base_latent.float()
    target_latent = target_latent.float()
    masked_target = masked_latent_target(base_latent, target_latent, latent_mask)
    return {
        "base_latent": base_latent,
        "base_sr": base_sr.float(),
        "bicubic": bicubic.float(),
        "detail_mask": detail_mask,
        "latent_mask": latent_mask,
        "target_latent": target_latent,
        "masked_target": masked_target,
        "condition": make_condition(base_latent, lr, latent_mask),
    }


@torch.no_grad()
def residual_shift_sample(
    model: ConditionalUNet,
    base_latent: torch.Tensor,
    condition: torch.Tensor,
    latent_mask: torch.Tensor,
    domain_id: torch.Tensor,
    diffusion_cfg: dict[str, Any],
    steps: int,
    seed: int,
) -> torch.Tensor:
    num_timesteps = int(diffusion_cfg.get("num_train_timesteps", 100))
    eta_power = float(diffusion_cfg.get("eta_power", 1.0))
    noise_scale = float(diffusion_cfg.get("noise_scale", 0.15))
    maximum = float(diffusion_cfg.get("max_correction", 4.0))
    generator = torch.Generator(device=base_latent.device).manual_seed(int(seed))
    noise = torch.randn(
        base_latent.shape,
        generator=generator,
        device=base_latent.device,
        dtype=base_latent.dtype,
    )
    sample = base_latent + noise_scale * noise
    timestep_values = torch.linspace(
        num_timesteps - 1,
        0,
        steps=max(2, int(steps)),
        device=base_latent.device,
    ).round().long()
    timestep_values = torch.unique_consecutive(timestep_values)
    for index, timestep_value in enumerate(timestep_values):
        timestep = torch.full(
            (base_latent.shape[0],),
            int(timestep_value),
            device=base_latent.device,
            dtype=torch.long,
        )
        raw_correction = model(sample, timestep, condition, domain_id).float()
        predicted_target = apply_masked_correction(
            base_latent,
            bounded_correction(raw_correction, maximum),
            latent_mask,
        )
        eta = residual_shift_eta(timestep, num_timesteps, eta_power)
        if index + 1 < len(timestep_values):
            next_timestep = torch.full_like(timestep, int(timestep_values[index + 1]))
            next_eta = residual_shift_eta(next_timestep, num_timesteps, eta_power)
        else:
            next_eta = torch.zeros_like(eta)
        sample = residual_shift_step(sample, predicted_target, base_latent, eta, next_eta, noise_scale)
    return sample


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    optimizer_updates: int,
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "optimizer_updates": int(optimizer_updates),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": clean_config(config),
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = int(checkpoint.get("step", 0))
    updates = int(checkpoint.get("optimizer_updates", step))
    return step, updates


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
        group=wandb_cfg.get("group", "stage-latent-diffusion"),
        job_type=wandb_cfg.get("job_type", "masked-latent-residual-shift"),
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
    limit = int(eval_cfg.get("limit", 20))
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


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (value.float() * mask.float()).flatten(1).sum(dim=1)
    denominator = mask.float().flatten(1).sum(dim=1).clamp_min(1e-8)
    return numerator / denominator


@torch.no_grad()
def evaluate(
    model: ConditionalUNet,
    vae: nn.Module,
    condition_encoder: nn.Module,
    mask_predictor: nn.Module,
    dataloader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    dtype_name: str,
    output_dir: Path,
    step: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    eval_cfg = config.get("eval", {})
    mask_cfg = config.get("detail_mask", {})
    diffusion_cfg = config.get("diffusion", {})
    seeds = [int(seed) for seed in eval_cfg.get("seeds", [123])]
    sample_steps = int(eval_cfg.get("sample_steps", 8))
    sample_count = int(eval_cfg.get("sample_count", 6))
    totals = {name: 0.0 for name in (
        "base_psnr", "base_ssim", "base_highpass_l1", "base_detail_ratio",
        "sr_psnr", "sr_ssim", "sr_highpass_l1", "sr_detail_ratio",
        "masked_highpass_l1", "outside_drift", "lowpass_drift", "correction_l1", "diversity_l1",
    )}
    image_count = 0
    sampled_count = 0
    grid_rows: list[list[tuple[str, Any]]] = []
    for batch_index, batch in enumerate(dataloader):
        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        frozen = prepare_frozen_batch(
            vae, condition_encoder, mask_predictor, hr, lr, domain_id, mask_cfg, device, dtype_name
        )
        base_sr = frozen["base_sr"]
        mask = frozen["detail_mask"]
        base_mse = (base_sr - hr).square().flatten(1).mean(dim=1)
        base_high = metric_highpass(base_sr)
        target_high = metric_highpass(hr)
        target_energy = target_high.abs().flatten(1).mean(dim=1).clamp_min(1e-8)
        totals["base_psnr"] += float((-10.0 * torch.log10(base_mse.clamp_min(1e-12))).sum().cpu())
        totals["base_ssim"] += float(ssim_per_image(base_sr, hr).sum().cpu())
        totals["base_highpass_l1"] += float((base_high - target_high).abs().flatten(1).mean(dim=1).sum().cpu())
        totals["base_detail_ratio"] += float((base_high.abs().flatten(1).mean(dim=1) / target_energy).sum().cpu())
        sampled_images: list[torch.Tensor] = []
        for seed in seeds:
            with autocast_context(device, dtype_name):
                sampled_latent = residual_shift_sample(
                    model=model,
                    base_latent=frozen["base_latent"],
                    condition=frozen["condition"],
                    latent_mask=frozen["latent_mask"],
                    domain_id=domain_id,
                    diffusion_cfg=diffusion_cfg,
                    steps=sample_steps,
                    seed=seed + batch_index * 100_000,
                )
                sr = denormalize(vae.decode(sampled_latent)).float()
            sampled_images.append(sr)
            sr_mse = (sr - hr).square().flatten(1).mean(dim=1)
            sr_high = metric_highpass(sr)
            high_error = (sr_high - target_high).abs().mean(dim=1, keepdim=True)
            inverse_mask = 1.0 - mask
            totals["sr_psnr"] += float((-10.0 * torch.log10(sr_mse.clamp_min(1e-12))).sum().cpu())
            totals["sr_ssim"] += float(ssim_per_image(sr, hr).sum().cpu())
            totals["sr_highpass_l1"] += float(high_error.flatten(1).mean(dim=1).sum().cpu())
            totals["sr_detail_ratio"] += float((sr_high.abs().flatten(1).mean(dim=1) / target_energy).sum().cpu())
            totals["masked_highpass_l1"] += float(_masked_mean(high_error, mask).sum().cpu())
            totals["outside_drift"] += float(_masked_mean((sr - base_sr).abs().mean(1, keepdim=True), inverse_mask).sum().cpu())
            totals["lowpass_drift"] += float((lowpass(sr, 31) - lowpass(base_sr, 31)).abs().flatten(1).mean(dim=1).sum().cpu())
            totals["correction_l1"] += float((sampled_latent - frozen["base_latent"]).abs().flatten(1).mean(dim=1).sum().cpu())
            sampled_count += int(hr.shape[0])
        if len(sampled_images) > 1:
            pairs = [
                (sampled_images[left] - sampled_images[right]).abs().flatten(1).mean(dim=1)
                for left in range(len(sampled_images))
                for right in range(left + 1, len(sampled_images))
            ]
            totals["diversity_l1"] += float(torch.stack(pairs).mean(dim=0).sum().cpu())
        lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)
        for item_index in range(int(hr.shape[0])):
            if len(grid_rows) >= sample_count:
                break
            row: list[tuple[str, Any]] = [
                ("LR", tensor_to_pil(lr_nearest[item_index])),
                ("detail mask", tensor_to_pil(mask[item_index].repeat(3, 1, 1))),
                ("Stage 2 base", tensor_to_pil(base_sr[item_index])),
            ]
            row.extend(
                (f"shift seed {seed}", tensor_to_pil(sampled_images[index][item_index]))
                for index, seed in enumerate(seeds)
            )
            row.append(("GT", tensor_to_pil(hr[item_index])))
            grid_rows.append(row)
        image_count += int(hr.shape[0])
    image_count = max(1, image_count)
    sampled_count = max(1, sampled_count)
    metrics = {
        "eval/base_decoded_psnr": totals["base_psnr"] / image_count,
        "eval/base_ssim": totals["base_ssim"] / image_count,
        "eval/base_highpass_l1": totals["base_highpass_l1"] / image_count,
        "eval/base_detail_ratio": totals["base_detail_ratio"] / image_count,
        "eval/decoded_psnr": totals["sr_psnr"] / sampled_count,
        "eval/decoded_ssim": totals["sr_ssim"] / sampled_count,
        "eval/highpass_l1": totals["sr_highpass_l1"] / sampled_count,
        "eval/detail_ratio": totals["sr_detail_ratio"] / sampled_count,
        "eval/masked_highpass_l1": totals["masked_highpass_l1"] / sampled_count,
        "eval/outside_mask_drift": totals["outside_drift"] / sampled_count,
        "eval/lowpass_drift": totals["lowpass_drift"] / sampled_count,
        "eval/latent_correction_l1": totals["correction_l1"] / sampled_count,
        "eval/diversity_l1": totals["diversity_l1"] / image_count,
        "eval/num_images": float(image_count),
        "eval/num_seeds": float(len(seeds)),
    }
    metrics["eval/psnr_delta_vs_base"] = metrics["eval/decoded_psnr"] - metrics["eval/base_decoded_psnr"]
    metrics["eval/ssim_delta_vs_base"] = metrics["eval/decoded_ssim"] - metrics["eval/base_ssim"]
    metrics["eval/highpass_gain_vs_base"] = metrics["eval/base_highpass_l1"] - metrics["eval/highpass_l1"]
    metrics["eval/detail_ratio_delta_vs_base"] = metrics["eval/detail_ratio"] - metrics["eval/base_detail_ratio"]
    metrics["eval/detail_score"] = (
        metrics["eval/decoded_psnr"]
        + 0.25 * metrics["eval/detail_ratio_delta_vs_base"]
        + 0.5 * metrics["eval/ssim_delta_vs_base"]
    )
    eval_dir = output_dir / f"eval_step_{step:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if grid_rows:
        make_grid(grid_rows, eval_dir / "grid_lr_mask_base_shift_gt.png")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if was_training:
        model.train()
    return metrics


def lr_factor(update: int, total_updates: int, scheduler_cfg: dict[str, Any]) -> float:
    warmup = max(0, int(scheduler_cfg.get("warmup_updates", 0)))
    minimum = float(scheduler_cfg.get("min_lr_ratio", 0.2))
    if warmup > 0 and update <= warmup:
        return max(1e-8, update / warmup)
    progress = (update - warmup) / max(1, total_updates - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["project"]["output_dir"] = str(args.output_dir)
    if args.disable_wandb:
        config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.grad_accum_steps is not None:
        config.setdefault("train", {})["grad_accum_steps"] = int(args.grad_accum_steps)
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
    mask_predictor = load_detail_mask_predictor(config, device)
    if mask_predictor is None:
        raise ValueError("detail_mask.checkpoint is required")
    model = ConditionalUNet.from_config(config["model"]).to(device)
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})
    diffusion_cfg = config.get("diffusion", {})
    perceptual_model = make_perceptual_model(loss_cfg, device)
    base_lr = float(train_cfg.get("lr", 5e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    step = 0
    optimizer_updates = 0
    if args.resume is not None:
        step, optimizer_updates = load_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed={args.resume} step={step} optimizer_updates={optimizer_updates}", flush=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} dtype={dtype_name} model_params={parameter_count}", flush=True)
    run = init_wandb(config, output_dir, model)

    train_dataset = make_dataset(config, split=str(config["data"].get("split", "train")), seed=seed, deterministic=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    eval_loader = make_eval_loader(config, seed, device)
    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 5000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 2))
    total_updates = max(1, math.ceil(max_steps / grad_accum_steps))
    log_every = int(train_cfg.get("log_every", 25))
    save_every = int(train_cfg.get("save_every", 500))
    eval_cfg = config.get("eval", {})
    eval_every = int(eval_cfg.get("every", 500))
    run_at_start = bool(eval_cfg.get("run_at_start", True))
    best_metric_name = str(eval_cfg.get("best_metric", "eval/detail_score"))
    best_mode = str(eval_cfg.get("best_mode", "max"))
    best_metric = float("-inf") if best_mode == "max" else float("inf")
    metrics_log_path = output_dir / "metrics.jsonl"
    scheduler_cfg = train_cfg.get("scheduler", {}) or {}

    wandb_log(run, {
        "dataset/num_images": len(train_dataset),
        "train/batch_size": int(train_cfg.get("batch_size", 4)),
        "train/grad_accum_steps": grad_accum_steps,
        "train/effective_batch_size": int(train_cfg.get("batch_size", 4)) * grad_accum_steps,
        "model/parameters": parameter_count,
    }, step=step)

    def run_eval(eval_step: int) -> dict[str, float]:
        metrics = evaluate(
            model, vae, condition_encoder, mask_predictor, eval_loader, config, device, dtype_name, output_dir, eval_step
        )
        print(
            f"eval step={eval_step} decoded_psnr={metrics['eval/decoded_psnr']:.4f} "
            f"base_psnr={metrics['eval/base_decoded_psnr']:.4f} "
            f"delta={metrics['eval/psnr_delta_vs_base']:+.4f} "
            f"ssim={metrics['eval/decoded_ssim']:.5f} "
            f"detail_ratio={metrics['eval/detail_ratio']:.4f} "
            f"hp_gain={metrics['eval/highpass_gain_vs_base']:+.6f} "
            f"outside={metrics['eval/outside_mask_drift']:.6f}",
            flush=True,
        )
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": eval_step, **metrics}, sort_keys=True) + "\n")
        wandb_data: dict[str, Any] = dict(metrics)
        grid_path = output_dir / f"eval_step_{eval_step:06d}" / "grid_lr_mask_base_shift_gt.png"
        if run is not None and grid_path.exists():
            import wandb

            wandb_data["samples/eval_grid"] = wandb.Image(str(grid_path), caption=f"residual shift step {eval_step}")
        wandb_log(run, wandb_data, eval_step)
        return metrics

    if run_at_start and step == 0:
        run_eval(0)

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
        frozen = prepare_frozen_batch(
            vae, condition_encoder, mask_predictor, hr, lr, domain_id, config.get("detail_mask", {}), device, dtype_name
        )
        timesteps = torch.randint(
            0,
            int(diffusion_cfg.get("num_train_timesteps", 100)),
            (hr.shape[0],),
            device=device,
            dtype=torch.long,
        )
        eta = residual_shift_eta(
            timesteps,
            int(diffusion_cfg.get("num_train_timesteps", 100)),
            float(diffusion_cfg.get("eta_power", 1.0)),
        )
        noise = torch.randn_like(frozen["masked_target"])
        noisy_latent = residual_shift_forward_sample(
            frozen["masked_target"],
            frozen["base_latent"],
            eta,
            noise,
            float(diffusion_cfg.get("noise_scale", 0.15)),
        )
        with autocast_context(device, dtype_name):
            raw_correction = model(noisy_latent, timesteps, frozen["condition"], domain_id)
            correction = bounded_correction(raw_correction, float(diffusion_cfg.get("max_correction", 4.0)))
            predicted_latent = apply_masked_correction(
                frozen["base_latent"], correction, frozen["latent_mask"]
            )
            predicted_sr = denormalize(vae.decode(predicted_latent))
            latent_loss = masked_charbonnier(
                predicted_latent,
                frozen["target_latent"],
                frozen["latent_mask"],
                float(loss_cfg.get("charbonnier_eps", 1e-3)),
            )
            image_loss = charbonnier(predicted_sr, hr, float(loss_cfg.get("charbonnier_eps", 1e-3)))
            highpass_loss = masked_charbonnier(
                metric_highpass(predicted_sr),
                metric_highpass(hr),
                frozen["detail_mask"],
                float(loss_cfg.get("charbonnier_eps", 1e-3)),
            )
            laplacian_loss = masked_charbonnier(
                laplacian_response(predicted_sr),
                laplacian_response(hr),
                frozen["detail_mask"],
                float(loss_cfg.get("charbonnier_eps", 1e-3)),
            )
            outside_anchor = masked_charbonnier(
                predicted_sr,
                frozen["base_sr"],
                1.0 - frozen["detail_mask"],
                float(loss_cfg.get("charbonnier_eps", 1e-3)),
            )
            lowpass_anchor = charbonnier(
                lowpass(predicted_sr, int(loss_cfg.get("lowpass_kernel", 31))),
                lowpass(frozen["base_sr"], int(loss_cfg.get("lowpass_kernel", 31))),
                float(loss_cfg.get("charbonnier_eps", 1e-3)),
            )
            perceptual_loss = predicted_sr.new_zeros(())
            if perceptual_model is not None and float(loss_cfg.get("masked_perceptual_weight", 0.0)) > 0.0:
                perceptual_loss = perceptual_model(normalize_image(predicted_sr), normalize_image(hr), frozen["detail_mask"])
            correction_l1 = (correction.float().abs() * frozen["latent_mask"]).mean()
            loss = (
                float(loss_cfg.get("latent_weight", 1.0)) * latent_loss
                + float(loss_cfg.get("image_weight", 0.5)) * image_loss
                + float(loss_cfg.get("highpass_weight", 1.0)) * highpass_loss
                + float(loss_cfg.get("laplacian_weight", 0.5)) * laplacian_loss
                + float(loss_cfg.get("outside_anchor_weight", 1.0)) * outside_anchor
                + float(loss_cfg.get("lowpass_anchor_weight", 1.0)) * lowpass_anchor
                + float(loss_cfg.get("masked_perceptual_weight", 0.0)) * perceptual_loss
                + float(loss_cfg.get("correction_l1_weight", 0.0)) * correction_l1
            )
        (loss / grad_accum_steps).backward()
        step += 1
        if step % grad_accum_steps == 0:
            factor = lr_factor(optimizer_updates + 1, total_updates, scheduler_cfg)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * factor
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
        if step % log_every == 0 or step == 1:
            elapsed = max(time.time() - last_log_time, 1e-6)
            interval = max(step - last_log_step, 1)
            last_log_time = time.time()
            last_log_step = step
            metrics = {
                "train/loss": float(loss.detach().cpu()),
                "train/latent": float(latent_loss.detach().cpu()),
                "train/image": float(image_loss.detach().cpu()),
                "train/highpass": float(highpass_loss.detach().cpu()),
                "train/laplacian": float(laplacian_loss.detach().cpu()),
                "train/outside_anchor": float(outside_anchor.detach().cpu()),
                "train/lowpass_anchor": float(lowpass_anchor.detach().cpu()),
                "train/masked_perceptual": float(perceptual_loss.detach().cpu()),
                "train/correction_l1": float(correction_l1.detach().cpu()),
                "train/mask_mean": float(frozen["detail_mask"].mean().cpu()),
                "train/eta_mean": float(eta.mean().cpu()),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/optimizer_updates": float(optimizer_updates),
                "system/steps_per_s": interval / elapsed,
            }
            if device.type == "cuda":
                metrics["system/gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                metrics["system/gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            print(
                f"step={step} loss={metrics['train/loss']:.5f} latent={metrics['train/latent']:.5f} "
                f"image={metrics['train/image']:.5f} highpass={metrics['train/highpass']:.5f} "
                f"outside={metrics['train/outside_anchor']:.5f} correction={metrics['train/correction_l1']:.5f} "
                f"lr={metrics['train/lr']:.2e} steps_per_s={metrics['system/steps_per_s']:.3f} "
                f"peak_vram={metrics.get('system/gpu_peak_allocated_gib', 0.0):.1f}GiB",
                flush=True,
            )
            wandb_log(run, metrics, step)
        if step % save_every == 0:
            save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, optimizer, step, optimizer_updates, config)
            save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, optimizer_updates, config)
        if eval_every > 0 and step % eval_every == 0:
            metrics = run_eval(step)
            metric = float(metrics[best_metric_name])
            improved = metric > best_metric if best_mode == "max" else metric < best_metric
            if improved:
                best_metric = metric
                save_checkpoint(
                    checkpoints_dir / "best_eval.pt", model, optimizer, step, optimizer_updates, config, metrics
                )
            model.train()

    final_metrics = None
    if not args.skip_final_eval and (eval_every <= 0 or step % eval_every != 0):
        final_metrics = run_eval(step)
    save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, optimizer_updates, config, final_metrics)
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
