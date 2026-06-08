from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.losses import FrozenVGGFeatureLoss
from sr_diffusion.losses.reconstruction import (
    charbonnier_loss,
    laplacian_loss,
    laplacian_residual_magnitude_loss,
    sobel_edge_loss,
)
from sr_diffusion.models import AutoencoderKL, LRToLatentPredictor
from sr_diffusion.utils import (
    autocast_context,
    format_partial_load_report,
    get_device,
    load_config,
    load_matching_weights,
    save_config,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 deterministic LR to HR-latent pretraining.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--partial-init",
        action="store_true",
        help="Load only shape-compatible tensors from --init-checkpoint. Useful when widening or deepening the model.",
    )
    return parser.parse_args()


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    array = np.round(array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def latent_loss(prediction: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(prediction, target)
    if kind == "mse":
        return F.mse_loss(prediction, target)
    if kind == "charbonnier":
        return torch.sqrt((prediction - target).pow(2) + 1e-6).mean()
    raise ValueError(f"Unsupported latent loss: {kind}")


def psnr_from_mse(mse: float, peak: float = 2.0) -> float:
    return 20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(max(mse, 1e-12)))


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


def compute_stage2_loss(
    prediction: torch.Tensor,
    target_latent: torch.Tensor,
    target_image: torch.Tensor,
    reference_image: torch.Tensor,
    vae: AutoencoderKL,
    loss_config: dict[str, Any],
    perceptual_model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latent = latent_loss(prediction, target_latent, str(loss_config.get("latent", "charbonnier")))
    decoded_weight = float(loss_config.get("decoded_weight", 0.0))
    edge_weight = float(loss_config.get("edge_weight", 0.0))
    highpass_weight = float(loss_config.get("highpass_weight", 0.0))
    highpass_magnitude_weight = float(loss_config.get("highpass_magnitude_weight", 0.0))
    perceptual_weight = float(loss_config.get("perceptual_weight", 0.0))
    if (
        decoded_weight > 0.0
        or edge_weight > 0.0
        or highpass_weight > 0.0
        or highpass_magnitude_weight > 0.0
        or perceptual_weight > 0.0
    ):
        decoded = vae.decode(prediction)
        eps = float(loss_config.get("charbonnier_eps", 1e-3))
        pixel = charbonnier_loss(decoded, target_image, eps=eps)
        edge = sobel_edge_loss(decoded, target_image, eps=eps)
        highpass = laplacian_loss(decoded, target_image, eps=eps)
        highpass_magnitude = laplacian_residual_magnitude_loss(
            decoded,
            target_image,
            reference_image,
            eps=eps,
        )
        if perceptual_weight > 0.0:
            if perceptual_model is None:
                raise ValueError("loss.perceptual_weight requires a perceptual model")
            perceptual = perceptual_model(decoded, target_image)
        else:
            perceptual = prediction.new_zeros(())
    else:
        decoded = prediction.new_empty(0)
        pixel = prediction.new_zeros(())
        edge = prediction.new_zeros(())
        highpass = prediction.new_zeros(())
        highpass_magnitude = prediction.new_zeros(())
        perceptual = prediction.new_zeros(())
    total = (
        float(loss_config.get("latent_weight", 1.0)) * latent
        + decoded_weight * pixel
        + edge_weight * edge
        + highpass_weight * highpass
        + highpass_magnitude_weight * highpass_magnitude
        + perceptual_weight * perceptual
    )
    return total, {
        "latent": latent,
        "decoded": pixel,
        "edge": edge,
        "highpass": highpass,
        "highpass_magnitude": highpass_magnitude,
        "perceptual": perceptual,
        "decoded_image": decoded,
    }


def make_perceptual_model(loss_config: dict[str, Any], device: torch.device) -> torch.nn.Module | None:
    if float(loss_config.get("perceptual_weight", 0.0)) <= 0.0:
        return None
    perceptual_config = loss_config.get("perceptual", {})
    model = FrozenVGGFeatureLoss(
        resize=int(perceptual_config.get("resize", 256)),
        layer_indices=perceptual_config.get("layer_indices", [3, 8, 15]),
        layer_weights=perceptual_config.get("layer_weights", [1.0, 1.0, 1.0]),
    ).to(device)
    model.eval()
    print(
        "perceptual_model=vgg16_imagenet_features "
        f"resize={model.resize} layers={list(model.layer_indices)} weights={list(model.layer_weights)}"
    )
    return model


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


def make_fixed_sample_batch(config: dict[str, Any], seed: int) -> dict[str, Any] | None:
    sample_config = config.get("logging", {}).get("samples", {})
    if not bool(sample_config.get("enabled", True)):
        return None
    count = int(sample_config.get("count", 4))
    if count <= 0:
        return None
    split = str(sample_config.get("split", "val"))
    fallback_split = str(sample_config.get("fallback_split", "train"))
    try:
        dataset = make_dataset(config, split=split, seed=seed, deterministic=True)
    except ValueError:
        if split == fallback_split:
            raise
        print(f"sample split '{split}' is empty; falling back to '{fallback_split}'")
        split = fallback_split
        dataset = make_dataset(config, split=split, seed=seed, deterministic=True)
    configured_indices = sample_config.get("indices")
    if configured_indices is None:
        indices = list(range(min(count, len(dataset))))
    else:
        indices = [int(index) % len(dataset) for index in configured_indices[:count]]
    items = [dataset[index] for index in indices]
    return {
        "hr": torch.stack([item["hr"] for item in items], dim=0),
        "lr": torch.stack([item["lr"] for item in items], dim=0),
        "domain_id": torch.stack([item["domain_id"] for item in items], dim=0),
        "path": [item["path"] for item in items],
        "split": split,
        "indices": indices,
    }


def init_wandb(config: dict[str, Any], output_dir: Path, model: torch.nn.Module) -> Any | None:
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging is enabled, but wandb is not installed") from exc
    wandb_dir = Path(wandb_cfg.get("dir", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = wandb_cfg.get("mode", "offline")
    os.environ["WANDB_MODE"] = str(mode)
    tags = list(wandb_cfg.get("tags") or [])
    if "stage2" not in tags:
        tags.append("stage2")
    run = wandb.init(
        project=wandb_cfg.get("project", "sr-diffusion"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", config.get("project", {}).get("name")),
        dir=str(wandb_dir),
        mode=mode,
        tags=tags,
        group=wandb_cfg.get("group", "stage2"),
        job_type=wandb_cfg.get("job_type", "latent-pretrain"),
        config=clean_config(config),
    )
    if bool(wandb_cfg.get("watch", False)):
        wandb.watch(model, log="gradients", log_freq=int(wandb_cfg.get("watch_log_freq", 100)))
    return run


def wandb_log(run: Any | None, data: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def load_autoencoder(config: dict[str, Any], device: torch.device) -> AutoencoderKL:
    auto_cfg = config["autoencoder"]
    vae_config = load_config(auto_cfg["config"])
    vae = AutoencoderKL.from_config(vae_config["model"]).to(device)
    checkpoint = torch.load(auto_cfg["checkpoint"], map_location=device)
    vae.load_state_dict(checkpoint["model"])
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"loaded_autoencoder={auto_cfg['checkpoint']} step={checkpoint.get('step', 'unknown')}")
    return vae


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": clean_config(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def load_model_weights(path: Path, model: torch.nn.Module, device: torch.device, partial: bool = False) -> int:
    checkpoint = torch.load(path, map_location=device)
    if partial:
        stats = load_matching_weights(model, checkpoint["model"])
        print(format_partial_load_report("model", stats))
    else:
        model.load_state_dict(checkpoint["model"])
    return int(checkpoint.get("step", 0))


def evaluate(
    model: LRToLatentPredictor,
    vae: AutoencoderKL,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    loss_config: dict[str, Any],
    perceptual_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {
        "loss": 0.0,
        "latent_loss": 0.0,
        "latent_mse": 0.0,
        "decoded_mse": 0.0,
        "decoded_edge": 0.0,
        "decoded_highpass": 0.0,
        "perceptual": 0.0,
        "laplacian_energy_ratio": 0.0,
        "oracle_decoded_mse": 0.0,
        "oracle_laplacian_energy_ratio": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            target = normalize_image(hr)
            lr_input = normalize_image(lr)
            reference = F.interpolate(lr_input, size=target.shape[-2:], mode="bicubic", align_corners=False)
            batch_size = int(hr.shape[0])
            with autocast_context(device, dtype_name):
                target_latent, _ = vae.encode(target)
                prediction = model(lr_input, domain_id)
                loss, components = compute_stage2_loss(
                    prediction,
                    target_latent,
                    target,
                    reference,
                    vae,
                    loss_config,
                    perceptual_model,
                )
                latent_mse = F.mse_loss(prediction, target_latent)
                decoded = components["decoded_image"] if components["decoded_image"].numel() > 0 else vae.decode(prediction)
                decoded_mse = F.mse_loss(decoded, target)
                oracle_decoded = vae.decode(target_latent)
                oracle_decoded_mse = F.mse_loss(oracle_decoded, target)
            target_laplacian = laplacian_response(target)
            decoded_laplacian = laplacian_response(decoded)
            oracle_laplacian = laplacian_response(oracle_decoded)
            target_energy = target_laplacian.abs().flatten(1).mean(dim=1).clamp_min(1e-12)
            decoded_energy_ratio = decoded_laplacian.abs().flatten(1).mean(dim=1) / target_energy
            oracle_energy_ratio = oracle_laplacian.abs().flatten(1).mean(dim=1) / target_energy
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["latent_loss"] += float(components["latent"].detach().cpu()) * batch_size
            totals["latent_mse"] += float(latent_mse.detach().cpu()) * batch_size
            totals["decoded_mse"] += float(decoded_mse.detach().cpu()) * batch_size
            totals["decoded_edge"] += float(sobel_edge_loss(decoded, target).detach().cpu()) * batch_size
            totals["decoded_highpass"] += float(laplacian_loss(decoded, target).detach().cpu()) * batch_size
            totals["perceptual"] += float(components["perceptual"].detach().cpu()) * batch_size
            totals["laplacian_energy_ratio"] += float(decoded_energy_ratio.sum().cpu())
            totals["oracle_decoded_mse"] += float(oracle_decoded_mse.detach().cpu()) * batch_size
            totals["oracle_laplacian_energy_ratio"] += float(oracle_energy_ratio.sum().cpu())
            count += batch_size
    if was_training:
        model.train()
    count = max(1, count)
    decoded_mse = totals["decoded_mse"] / count
    metrics = {
        "eval/loss": totals["loss"] / count,
        "eval/latent_loss": totals["latent_loss"] / count,
        "eval/latent_mse": totals["latent_mse"] / count,
        "eval/decoded_mse": decoded_mse,
        "eval/decoded_psnr": psnr_from_mse(decoded_mse),
        "eval/decoded_edge": totals["decoded_edge"] / count,
        "eval/decoded_highpass": totals["decoded_highpass"] / count,
        "eval/perceptual": totals["perceptual"] / count,
        "eval/laplacian_energy_ratio": totals["laplacian_energy_ratio"] / count,
        "eval/oracle_decoded_psnr": psnr_from_mse(totals["oracle_decoded_mse"] / count),
        "eval/oracle_laplacian_energy_ratio": totals["oracle_laplacian_energy_ratio"] / count,
        "eval/num_images": float(count),
    }
    detail_score_weight = float(loss_config.get("detail_score_weight", 0.0))
    metrics["eval/psnr_detail_score"] = metrics["eval/decoded_psnr"] + detail_score_weight * metrics[
        "eval/laplacian_energy_ratio"
    ]
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["project"]["output_dir"] = str(args.output_dir)
    if args.disable_wandb:
        config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    output_dir = Path(config["project"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    train_cfg = config["train"]
    device = get_device(train_cfg.get("device", "auto"))
    dtype_name = train_cfg.get("dtype", "bf16")
    loss_config = config.get("loss", {})
    print(f"device={device} dtype={dtype_name}")

    train_dataset = make_dataset(config, split=config["data"].get("split", "train"), seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )
    fixed_sample_batch = make_fixed_sample_batch(config, seed=seed)
    if fixed_sample_batch is not None:
        print(
            "sample_logging="
            f"split={fixed_sample_batch['split']} "
            f"indices={fixed_sample_batch['indices']} "
            f"count={len(fixed_sample_batch['path'])}"
        )

    eval_cfg = config.get("eval", {})
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_loader = None
    eval_every = int(eval_cfg.get("every", 1000))
    eval_run_at_start = bool(eval_cfg.get("run_at_start", True))
    best_metric_name = str(eval_cfg.get("best_metric", "eval/latent_loss"))
    best_metric_mode = str(eval_cfg.get("best_mode", "min"))
    best_checkpoint_name = str(eval_cfg.get("best_checkpoint", "best_eval_latent.pt"))
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported eval.best_mode: {best_metric_mode}")
    if eval_enabled:
        eval_dataset = make_dataset(config, split=str(eval_cfg.get("split", "val")), seed=seed, deterministic=True)
        limit = int(eval_cfg.get("limit", 0))
        if limit > 0 and limit < len(eval_dataset):
            from torch.utils.data import Subset

            eval_dataset = Subset(eval_dataset, list(range(limit)))
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 1))),
            shuffle=False,
            num_workers=int(eval_cfg.get("num_workers", config["data"].get("num_workers", 0))),
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        print(
            "eval="
            f"split={eval_cfg.get('split', 'val')} "
            f"limit={eval_cfg.get('limit', 0)} "
            f"batch_size={eval_cfg.get('batch_size', train_cfg.get('batch_size', 1))}"
        )

    vae = load_autoencoder(config, device=device)
    perceptual_model = make_perceptual_model(loss_config, device=device)
    model = LRToLatentPredictor.from_config(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed step={start_step}")
    else:
        init_config = config.get("initialization", {})
        init_checkpoint = args.init_checkpoint or init_config.get("checkpoint")
        if init_checkpoint:
            partial_init = bool(args.partial_init or init_config.get("partial", False))
            init_step = load_model_weights(Path(init_checkpoint), model, device, partial=partial_init)
            print(f"initialized_from={init_checkpoint} source_step={init_step} partial_init={partial_init}")

    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 1000))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    sample_every = int(train_cfg.get("sample_every", 500))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    run = init_wandb(config, output_dir, model)
    wandb_log(
        run,
        {
            "dataset/num_images": len(train_dataset),
            "train/batch_size": int(train_cfg.get("batch_size", 1)),
            "train/grad_accum_steps": grad_accum_steps,
        },
        step=start_step,
    )

    model.train()
    step = start_step
    best_eval = float("inf") if best_metric_mode == "min" else float("-inf")
    last_log = time.time()
    last_log_step = step
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in train_loader:
            step += 1
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            target = normalize_image(hr)
            lr_input = normalize_image(lr)
            reference = F.interpolate(lr_input, size=target.shape[-2:], mode="bicubic", align_corners=False)

            with torch.no_grad():
                with autocast_context(device, dtype_name):
                    target_latent, _ = vae.encode(target)

            with autocast_context(device, dtype_name):
                prediction = model(lr_input, domain_id)
                loss, loss_components = compute_stage2_loss(
                    prediction,
                    target_latent,
                    target,
                    reference,
                    vae,
                    loss_config,
                    perceptual_model,
                )
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()
            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0 or step == 1:
                elapsed = max(1e-6, time.time() - last_log)
                interval_steps = max(1, step - last_log_step)
                last_log = time.time()
                last_log_step = step
                latent_mse = F.mse_loss(prediction.detach(), target_latent.detach())
                print(
                    f"step={step} loss={float(loss.detach().cpu()):.5f} "
                    f"latent={float(loss_components['latent'].detach().cpu()):.5f} "
                    f"decoded={float(loss_components['decoded'].detach().cpu()):.5f} "
                    f"edge={float(loss_components['edge'].detach().cpu()):.5f} "
                    f"highpass={float(loss_components['highpass'].detach().cpu()):.5f} "
                    f"highpass_mag={float(loss_components['highpass_magnitude'].detach().cpu()):.5f} "
                    f"perceptual={float(loss_components['perceptual'].detach().cpu()):.5f} "
                    f"latent_mse={float(latent_mse.detach().cpu()):.5f} "
                    f"steps_per_sec={interval_steps / elapsed:.2f}"
                )
                wandb_log(
                    run,
                    {
                        "train/loss": float(loss.detach().cpu()),
                        "train/latent_loss": float(loss_components["latent"].detach().cpu()),
                        "train/decoded": float(loss_components["decoded"].detach().cpu()),
                        "train/edge": float(loss_components["edge"].detach().cpu()),
                        "train/highpass": float(loss_components["highpass"].detach().cpu()),
                        "train/highpass_magnitude": float(loss_components["highpass_magnitude"].detach().cpu()),
                        "train/perceptual": float(loss_components["perceptual"].detach().cpu()),
                        "train/latent_mse": float(latent_mse.detach().cpu()),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "system/steps_per_sec": interval_steps / elapsed,
                    },
                    step=step,
                )

            should_eval = (
                eval_enabled
                and eval_loader is not None
                and eval_every > 0
                and (step % eval_every == 0 or (step == 1 and eval_run_at_start))
            )
            if should_eval:
                metrics = evaluate(model, vae, eval_loader, device, dtype_name, loss_config, perceptual_model)
                (eval_dir / f"step_{step:07d}_metrics.json").write_text(
                    json.dumps({"step": step, "metrics": metrics}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"eval step={step} latent_loss={metrics['eval/latent_loss']:.5f} "
                    f"decoded_psnr={metrics['eval/decoded_psnr']:.2f} "
                    f"detail_ratio={metrics['eval/laplacian_energy_ratio']:.3f} "
                    f"perceptual={metrics['eval/perceptual']:.5f} "
                    f"psnr_detail_score={metrics['eval/psnr_detail_score']:.3f}"
                )
                wandb_log(run, metrics, step=step)
                metric_value = float(metrics[best_metric_name])
                is_better = metric_value < best_eval if best_metric_mode == "min" else metric_value > best_eval
                if is_better:
                    best_eval = metric_value
                    save_checkpoint(checkpoints_dir / best_checkpoint_name, model, optimizer, step, config)

            if step % sample_every == 0 or step == 1:
                sample_source = fixed_sample_batch if fixed_sample_batch is not None else batch
                with torch.no_grad():
                    sample_hr = sample_source["hr"].to(device, non_blocking=True)
                    sample_lr = sample_source["lr"].to(device, non_blocking=True)
                    sample_domain = sample_source["domain_id"].to(device, non_blocking=True)
                    sample_target = normalize_image(sample_hr)
                    sample_lr_input = normalize_image(sample_lr)
                    with autocast_context(device, dtype_name):
                        sample_pred = model(sample_lr_input, sample_domain)
                        sample_decoded = vae.decode(sample_pred)
                    sample_count = sample_hr.shape[0]
                    lr_display = F.interpolate(sample_source["lr"].float().cpu(), size=sample_hr.shape[-2:], mode="nearest")
                    gt = sample_hr.float().cpu()
                    pred = denormalize(sample_decoded).float().cpu()

                    lr_path = samples_dir / f"step_{step:07d}_lr.png"
                    gt_path = samples_dir / f"step_{step:07d}_gt.png"
                    pred_path = samples_dir / f"step_{step:07d}_pred.png"
                    save_image(lr_display, lr_path, nrow=sample_count)
                    save_image(gt, gt_path, nrow=sample_count)
                    save_image(pred, pred_path, nrow=sample_count)
                    if run is not None:
                        import wandb

                        paths = sample_source.get("path", [""] * sample_count)
                        captions = [Path(str(path)).name or f"sample_{idx}" for idx, path in enumerate(paths[:sample_count])]
                        wandb_log(
                            run,
                            {
                                "samples/LR": [
                                    wandb.Image(tensor_to_pil(image), caption=caption)
                                    for image, caption in zip(lr_display, captions, strict=True)
                                ],
                                "samples/GT": [
                                    wandb.Image(tensor_to_pil(image), caption=caption)
                                    for image, caption in zip(gt, captions, strict=True)
                                ],
                                "samples/Pred": [
                                    wandb.Image(tensor_to_pil(image), caption=caption)
                                    for image, caption in zip(pred, captions, strict=True)
                                ],
                            },
                            step=step,
                        )

            if step % save_every == 0 or step == max_steps:
                save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, optimizer, step, config)
                save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config)

            if step >= max_steps:
                break

    print(f"finished step={step}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
