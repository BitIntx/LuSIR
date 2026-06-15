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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.detail_mask import (
    DetailMaskPredictor,
    detail_need_components,
    lowpass,
    normalize_score,
    observable_detail_proxies,
    top_fraction_mask,
)
from sr_diffusion.utils import autocast_context, get_device, load_config, save_config, seed_everything, seed_worker
from tools.train.train_detail_branch import (
    clean_config,
    load_autoencoder,
    load_condition_encoder,
    make_base_prediction,
    make_dataset,
    make_grid,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned detail-need mask predictor.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def pearson_per_image(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    first_centered = first_flat - first_flat.mean(dim=1, keepdim=True)
    second_centered = second_flat - second_flat.mean(dim=1, keepdim=True)
    numerator = (first_centered * second_centered).mean(dim=1)
    denominator = first_centered.square().mean(dim=1).sqrt() * second_centered.square().mean(dim=1).sqrt()
    return numerator / denominator.clamp_min(1e-12)


def selected_capture(energy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (energy.float() * mask.float()).flatten(1).sum(dim=1) / energy.float().flatten(1).sum(dim=1).clamp_min(1e-12)


def selected_concentration(energy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected_mean = (energy.float() * mask.float()).flatten(1).sum(dim=1) / mask.float().flatten(1).sum(dim=1).clamp_min(1.0)
    return selected_mean / energy.float().flatten(1).mean(dim=1).clamp_min(1e-12)


def heatmap(score: torch.Tensor) -> Image.Image:
    score = score.detach().float().cpu().clamp(0.0, 1.0).repeat(3, 1, 1)
    score[1] = score[1].sqrt() * 0.65
    score[2] = (1.0 - score[2]) * 0.35
    return tensor_to_pil(score)


def make_eval_loader(config: dict[str, Any], seed: int, device: torch.device) -> DataLoader:
    eval_cfg = config.get("eval", {})
    dataset = make_dataset(config, split=str(eval_cfg.get("split", "val")), seed=seed, deterministic=True)
    limit = int(eval_cfg.get("limit", 100))
    if 0 < limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))
    return DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


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
        name=wandb_cfg.get("name", config.get("project", {}).get("name")),
        dir=str(wandb_dir),
        mode=wandb_cfg.get("mode", "online"),
        tags=list(wandb_cfg.get("tags") or []),
        group=wandb_cfg.get("group", "stage-detail-mask"),
        job_type=wandb_cfg.get("job_type", "detail-mask-predictor"),
        config=clean_config(config),
    )
    if bool(wandb_cfg.get("watch", False)):
        wandb.watch(model, log="gradients", log_freq=int(wandb_cfg.get("watch_log_freq", 200)))
    print(f"wandb_run={run.url}", flush=True)
    return run


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
        {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": clean_config(config), "metrics": metrics or {}},
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
    model.load_state_dict(checkpoint["model"])
    return {"checkpoint": str(path), "checkpoint_step": int(checkpoint.get("step", 0))}


def mask_target(base: torch.Tensor, hr: torch.Tensor, model: DetailMaskPredictor) -> dict[str, torch.Tensor]:
    return detail_need_components(
        base,
        hr,
        highpass_kernel=model.highpass_kernel,
        patch_kernel=model.patch_kernel,
        score_quantile=model.score_quantile,
    )


def low_score_patch_mask(score: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    if score.ndim != 4 or score.shape[1] != 1:
        raise ValueError(f"score must have shape [B, 1, H, W], got {tuple(score.shape)}")
    _, _, height, width = score.shape
    patch_size = max(1, min(int(patch_size), height, width))
    stride = max(1, int(stride))
    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))
    if y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    if x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)
    candidates: list[tuple[int, int]] = []
    patch_scores: list[torch.Tensor] = []
    score_float = score.float()
    for y in y_positions:
        for x in x_positions:
            candidates.append((y, x))
            patch_scores.append(score_float[:, :, y : y + patch_size, x : x + patch_size].mean(dim=(1, 2, 3)))
    best_index = torch.stack(patch_scores, dim=1).argmin(dim=1)
    mask = torch.zeros_like(score.float())
    for item_idx, index in enumerate(best_index.tolist()):
        y, x = candidates[index]
        mask[item_idx : item_idx + 1, :, y : y + patch_size, x : x + patch_size] = 1.0
    return mask


def apply_noise_negative_augmentation(
    base: torch.Tensor,
    clean_target: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not bool(cfg.get("enabled", False)):
        return None
    probability = float(cfg.get("probability", 1.0))
    if probability <= 0.0:
        return None
    patch_size = int(cfg.get("patch_size", 96))
    stride = int(cfg.get("stride", max(1, patch_size // 2)))
    sigma = float(cfg.get("sigma", 0.08))
    mask = low_score_patch_mask(clean_target, patch_size=patch_size, stride=stride).to(device=base.device, dtype=base.dtype)
    if probability < 1.0:
        active = (torch.rand(base.shape[0], 1, 1, 1, device=base.device) < probability).to(dtype=base.dtype)
        mask = mask * active
    if float(mask.sum().detach().cpu()) <= 0.0:
        return None
    noise = torch.randn_like(base.float()) * sigma
    noisy_base = (base.float() + noise * mask.float()).clamp(0.0, 1.0).to(dtype=base.dtype)
    return noisy_base, mask.float()


def training_loss(
    model: DetailMaskPredictor,
    base: torch.Tensor,
    bicubic: torch.Tensor,
    condition: torch.Tensor,
    hr: torch.Tensor,
    domain_id: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        components = mask_target(base, hr, model)
        target = components["score"]
        excess = normalize_score(lowpass(components["excess"], model.patch_kernel), quantile=model.score_quantile)
    prediction = model(base, bicubic, condition, domain_id)
    positive_weight = float(loss_cfg.get("positive_weight", 2.0))
    regression_map = F.smooth_l1_loss(prediction.float(), target, reduction="none")
    regression = (regression_map * (1.0 + positive_weight * target)).mean()
    correlation = pearson_per_image(prediction, target).mean()
    correlation_loss = 1.0 - correlation
    excess_penalty = (prediction.float() * excess).mean()
    mean_loss = (prediction.float().flatten(1).mean(dim=1) - target.flatten(1).mean(dim=1)).abs().mean()
    loss = (
        float(loss_cfg.get("regression_weight", 1.0)) * regression
        + float(loss_cfg.get("correlation_weight", 0.35)) * correlation_loss
        + float(loss_cfg.get("excess_weight", 0.15)) * excess_penalty
        + float(loss_cfg.get("mean_weight", 0.1)) * mean_loss
    )
    noise_negative_loss = prediction.new_zeros(())
    noise_region_prediction = prediction.new_zeros(())
    noise_target_mean = prediction.new_zeros(())
    noise_excess_mean = prediction.new_zeros(())
    noise_cfg = loss_cfg.get("noise_negative", {})
    augmented = apply_noise_negative_augmentation(base, target, noise_cfg)
    if augmented is not None:
        noisy_base, noise_mask = augmented
        with torch.no_grad():
            noisy_components = mask_target(noisy_base, hr, model)
            noisy_target = noisy_components["score"]
            noisy_excess = normalize_score(
                lowpass(noisy_components["excess"], model.patch_kernel),
                quantile=model.score_quantile,
            )
        noisy_prediction = model(noisy_base, bicubic, condition, domain_id)
        negative_regression = F.smooth_l1_loss(noisy_prediction.float(), noisy_target, reduction="none")
        negative_regression = (negative_regression * noise_mask).sum() / noise_mask.sum().clamp_min(1.0)
        noise_region_prediction = (noisy_prediction.float() * noise_mask).sum() / noise_mask.sum().clamp_min(1.0)
        negative_excess = (noisy_prediction.float() * noisy_excess).mean()
        noise_target_mean = (noisy_target * noise_mask).sum() / noise_mask.sum().clamp_min(1.0)
        noise_excess_mean = (noisy_excess * noise_mask).sum() / noise_mask.sum().clamp_min(1.0)
        noise_negative_loss = (
            float(noise_cfg.get("regression_weight", 0.25)) * negative_regression
            + float(noise_cfg.get("region_weight", 0.5)) * noise_region_prediction
            + float(noise_cfg.get("excess_weight", 0.25)) * negative_excess
        )
        loss = loss + noise_negative_loss
    return loss, {
        "regression": regression,
        "correlation": correlation,
        "excess_penalty": excess_penalty,
        "mean_loss": mean_loss,
        "prediction_mean": prediction.float().mean(),
        "target_mean": target.mean(),
        "noise_negative": noise_negative_loss,
        "noise_region_prediction": noise_region_prediction,
        "noise_target_mean": noise_target_mean,
        "noise_excess_mean": noise_excess_mean,
    }


@torch.no_grad()
def evaluate(
    model: DetailMaskPredictor,
    vae: nn.Module,
    condition_encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    fraction: float,
    output_dir: Path | None = None,
    sample_count: int = 0,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {
        name: 0.0
        for name in (
            "corr",
            "mae",
            "capture",
            "concentration",
            "excess_capture",
            "prediction_mean",
            "target_mean",
            "baseline_corr",
            "baseline_capture",
            "baseline_concentration",
            "baseline_excess_capture",
        )
    }
    count = 0
    grid_rows: list[list[tuple[str, Image.Image]]] = []
    for batch in dataloader:
        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        condition, base, bicubic = make_base_prediction(vae, condition_encoder, hr, lr, domain_id, device, dtype_name)
        components = mask_target(base, hr, model)
        with autocast_context(device, dtype_name):
            prediction = model(base, bicubic, condition, domain_id)
        prediction = prediction.float()
        target = components["score"]
        mask = top_fraction_mask(prediction, fraction)
        baseline = observable_detail_proxies(
            base,
            bicubic,
            highpass_kernel=model.highpass_kernel,
            patch_kernel=model.patch_kernel,
            score_quantile=model.score_quantile,
        )["highpass_disagreement"]
        baseline_mask = top_fraction_mask(baseline, fraction)
        batch_size = int(hr.shape[0])
        totals["corr"] += float(pearson_per_image(prediction, target).sum().cpu())
        totals["mae"] += float((prediction - target).abs().flatten(1).mean(dim=1).sum().cpu())
        totals["capture"] += float(selected_capture(components["missing"], mask).sum().cpu())
        totals["concentration"] += float(selected_concentration(components["missing"], mask).sum().cpu())
        totals["excess_capture"] += float(selected_capture(components["excess"], mask).sum().cpu())
        totals["prediction_mean"] += float(prediction.flatten(1).mean(dim=1).sum().cpu())
        totals["target_mean"] += float(target.flatten(1).mean(dim=1).sum().cpu())
        totals["baseline_corr"] += float(pearson_per_image(baseline, target).sum().cpu())
        totals["baseline_capture"] += float(selected_capture(components["missing"], baseline_mask).sum().cpu())
        totals["baseline_concentration"] += float(selected_concentration(components["missing"], baseline_mask).sum().cpu())
        totals["baseline_excess_capture"] += float(selected_capture(components["excess"], baseline_mask).sum().cpu())
        lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest")
        for item_idx in range(batch_size):
            if len(grid_rows) >= sample_count:
                break
            grid_rows.append(
                [
                    ("LR", tensor_to_pil(lr_nearest[item_idx])),
                    ("base", tensor_to_pil(base[item_idx])),
                    ("GT", tensor_to_pil(hr[item_idx])),
                    ("target", heatmap(target[item_idx])),
                    ("prediction", heatmap(prediction[item_idx])),
                    (f"prediction top {fraction:.0%}", heatmap(mask[item_idx])),
                ]
            )
        count += batch_size
    count = max(1, count)
    metrics = {f"eval/{name}": value / count for name, value in totals.items()}
    metrics["eval/selection_score"] = metrics["eval/corr"] + metrics["eval/capture"] - metrics["eval/excess_capture"]
    metrics["eval/baseline_selection_score"] = (
        metrics["eval/baseline_corr"] + metrics["eval/baseline_capture"] - metrics["eval/baseline_excess_capture"]
    )
    metrics["eval/delta_selection_score"] = metrics["eval/selection_score"] - metrics["eval/baseline_selection_score"]
    metrics["eval/num_images"] = float(count)
    if output_dir is not None and grid_rows:
        make_grid(grid_rows, output_dir / "detail_mask_prediction_grid.png")
    if was_training:
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
    model = DetailMaskPredictor.from_config(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    init_cfg = config.get("initialization", {})
    if init_cfg.get("checkpoint"):
        init_stats = init_model_from_checkpoint(Path(init_cfg["checkpoint"]), model, device)
        print(f"model_init={json.dumps(init_stats, sort_keys=True)}", flush=True)
    start_step = load_checkpoint(args.resume, model, optimizer, device) if args.resume else 0
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
    eval_loader = make_eval_loader(config, seed, device)
    train_cfg = config["train"]
    eval_cfg = config["eval"]
    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 4000))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every", 25))
    save_every = int(train_cfg.get("save_every", 500))
    eval_every = int(eval_cfg.get("every", 250))
    fraction = float(eval_cfg.get("selection_fraction", 0.2))
    sample_count = int(eval_cfg.get("sample_count", 8))
    best_metric_name = str(eval_cfg.get("best_metric", "eval/selection_score"))
    best_metric = float("-inf")
    best_metrics: dict[str, float] | None = None
    metrics_path = output_dir / "metrics.jsonl"

    def run_eval(step: int) -> dict[str, float]:
        metrics = evaluate(model, vae, condition_encoder, eval_loader, device, dtype_name, fraction, output_dir / f"eval_step_{step:06d}", sample_count)
        print(
            f"eval step={step} corr={metrics['eval/corr']:.4f} capture={metrics['eval/capture']:.4f} "
            f"excess={metrics['eval/excess_capture']:.4f} score={metrics['eval/selection_score']:.4f} "
            f"baseline={metrics['eval/baseline_selection_score']:.4f} delta={metrics['eval/delta_selection_score']:+.4f}",
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
        if run is not None:
            data: dict[str, Any] = dict(metrics)
            grid_path = output_dir / f"eval_step_{step:06d}" / "detail_mask_prediction_grid.png"
            if grid_path.exists():
                import wandb

                data["samples/detail_mask_grid"] = wandb.Image(str(grid_path), caption=f"step {step}")
            run.log(data, step=step)
        return metrics

    if args.eval_only_checkpoint:
        step = load_checkpoint(args.eval_only_checkpoint, model, None, device)
        print(json.dumps(run_eval(step), indent=2, sort_keys=True))
        if run is not None:
            run.finish()
        return
    if bool(eval_cfg.get("run_at_start", True)) and start_step == 0:
        best_metrics = run_eval(0)
        best_metric = float(best_metrics[best_metric_name])
        save_checkpoint(checkpoints_dir / "best_eval_mask.pt", model, optimizer, 0, config, best_metrics)

    step = start_step
    optimizer_updates = start_step // max(1, grad_accum)
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    last_log_time = time.time()
    last_log_step = step
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
            condition, base, bicubic = make_base_prediction(vae, condition_encoder, hr, lr, domain_id, device, dtype_name)
        with autocast_context(device, dtype_name):
            loss, parts = training_loss(model, base, bicubic, condition, hr, domain_id, config.get("loss", {}))
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
        step += 1
        if step == 1 or step % log_every == 0:
            elapsed = max(time.time() - last_log_time, 1e-6)
            logged_steps = max(step - last_log_step, 1)
            last_log_time = time.time()
            last_log_step = step
            train_metrics = {
                "train/loss": float(loss.detach().cpu()),
                **{f"train/{name}": float(value.detach().cpu()) for name, value in parts.items()},
                "train/optimizer_updates": float(optimizer_updates),
                "system/steps_per_s": logged_steps / elapsed,
            }
            print(
                f"step={step} loss={train_metrics['train/loss']:.5f} corr={train_metrics['train/correlation']:.4f} "
                f"pred={train_metrics['train/prediction_mean']:.4f} target={train_metrics['train/target_mean']:.4f} "
                f"noise_neg={train_metrics['train/noise_negative']:.5f} "
                f"noise_pred={train_metrics['train/noise_region_prediction']:.4f} "
                f"updates={optimizer_updates} steps_per_s={train_metrics['system/steps_per_s']:.3f}",
                flush=True,
            )
            if run is not None:
                run.log(train_metrics, step=step)
        if step % save_every == 0:
            save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, optimizer, step, config)
            save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config)
        if eval_every > 0 and step % eval_every == 0:
            metrics = run_eval(step)
            if float(metrics[best_metric_name]) > best_metric:
                best_metric = float(metrics[best_metric_name])
                best_metrics = metrics
                save_checkpoint(checkpoints_dir / "best_eval_mask.pt", model, optimizer, step, config, metrics)
            model.train()

    final_metrics = run_eval(step)
    save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config, final_metrics)
    if float(final_metrics[best_metric_name]) > best_metric:
        best_metric = float(final_metrics[best_metric_name])
        best_metrics = final_metrics
        save_checkpoint(checkpoints_dir / "best_eval_mask.pt", model, optimizer, step, config, final_metrics)
    summary = {
        "config": str(args.config),
        "finished_step": step,
        "optimizer_updates": optimizer_updates,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "checkpoint_best": str(checkpoints_dir / "best_eval_mask.pt"),
        "checkpoint_latest": str(checkpoints_dir / "latest.pt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
