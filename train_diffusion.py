from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.losses.reconstruction import (
    charbonnier_loss,
    laplacian_loss,
    laplacian_detail_gate_anchor_loss,
    laplacian_residual_magnitude_loss,
    lowpass_anchor_loss,
    sobel_edge_loss,
    sobel_residual_magnitude_loss,
)
from sr_diffusion.models import AutoencoderKL, ConditionalUNet, LRToLatentPredictor, NoiseScheduler, predict_x0_and_noise
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
from train_residual_refiner import BoundedResidualRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 conditional latent diffusion training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--partial-init",
        action="store_true",
        help="Load only shape-compatible diffusion U-Net tensors from --init-checkpoint.",
    )
    parser.add_argument(
        "--init-condition-encoder",
        action="store_true",
        help="Also load condition_encoder weights from --init-checkpoint. By default init uses the config checkpoint.",
    )
    parser.add_argument(
        "--partial-init-condition-encoder",
        action="store_true",
        help="When --init-condition-encoder is set, load only shape-compatible condition encoder tensors.",
    )
    return parser.parse_args()


def distributed_is_available() -> bool:
    return dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed() -> tuple[bool, int, int, int]:
    if not distributed_is_available():
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group(backend=backend)
    return True, rank, world_size, local_rank


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def print_main(message: str, rank: int) -> None:
    if is_main_process(rank):
        print(message)


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


def psnr_from_mse(mse: float, peak: float = 2.0) -> float:
    return 20.0 * float(np.log10(peak)) - 10.0 * float(np.log10(max(mse, 1e-12)))


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
    default_group = next((stage for stage in ("stage4", "stage3") if stage in tags), "stage3")
    run = wandb.init(
        project=wandb_cfg.get("project", "sr-diffusion"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", config.get("project", {}).get("name")),
        dir=str(wandb_dir),
        mode=mode,
        tags=tags,
        group=wandb_cfg.get("group", default_group),
        job_type=wandb_cfg.get("job_type", "diffusion"),
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


def load_condition_encoder(config: dict[str, Any], device: torch.device) -> tuple[LRToLatentPredictor, bool]:
    cond_cfg = config["condition_encoder"]
    model_config = cond_cfg.get("model")
    if model_config is None:
        model_config = load_config(cond_cfg["config"])["model"]
    encoder = LRToLatentPredictor.from_config(model_config).to(device)
    checkpoint = torch.load(cond_cfg["checkpoint"], map_location=device)
    encoder.load_state_dict(checkpoint["model"])
    trainable = bool(cond_cfg.get("trainable", False))
    encoder.train(mode=trainable)
    for parameter in encoder.parameters():
        parameter.requires_grad_(trainable)
    print(f"loaded_condition_encoder={cond_cfg['checkpoint']} step={checkpoint.get('step', 'unknown')} trainable={trainable}")
    return encoder, trainable


def load_residual_refiner_teacher(config: dict[str, Any], device: torch.device) -> BoundedResidualRefiner | None:
    teacher_cfg = config.get("teacher", {}).get("residual_refiner", {})
    if not bool(teacher_cfg.get("enabled", False)):
        return None
    teacher_config = load_config(teacher_cfg["config"])
    model_config = teacher_cfg.get("model", teacher_config["model"])
    teacher = BoundedResidualRefiner.from_config(model_config).to(device)
    checkpoint = torch.load(teacher_cfg["checkpoint"], map_location=device)
    teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(
        f"loaded_residual_refiner_teacher={teacher_cfg['checkpoint']} "
        f"step={checkpoint.get('step', 'unknown')}"
    )
    return teacher


def gated_residual_components(
    model_output: torch.Tensor,
    condition: torch.Tensor,
    diffusion_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction_type = str(diffusion_config.get("prediction_type", "noise"))
    if prediction_type != "gated_residual_x0":
        raise ValueError("teacher residual supervision requires diffusion.prediction_type='gated_residual_x0'")
    if model_output.shape[1] != condition.shape[1] * 2:
        raise ValueError(
            "gated_residual_x0 requires model out_channels to be exactly "
            f"2 * condition channels, got {model_output.shape[1]} and {condition.shape[1]}"
        )
    residual_logits, gate_logits = torch.chunk(model_output, 2, dim=1)
    residual_scale = float(diffusion_config.get("residual_scale", 1.0))
    gate_bias = float(diffusion_config.get("gate_bias", 0.0))
    gate = torch.sigmoid(gate_logits + gate_bias)
    residual = residual_scale * torch.tanh(residual_logits) * gate
    return residual, gate


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    condition_encoder: LRToLatentPredictor,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": unwrap_model(model).state_dict(),
            "condition_encoder": unwrap_model(condition_encoder).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": clean_config(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    condition_encoder: LRToLatentPredictor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "condition_encoder" in checkpoint:
        condition_encoder.load_state_dict(checkpoint["condition_encoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def load_model_weights(
    path: Path,
    model: torch.nn.Module,
    condition_encoder: LRToLatentPredictor,
    device: torch.device,
    load_condition_encoder: bool = False,
    partial: bool = False,
    partial_condition_encoder: bool = False,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    if partial:
        stats = load_matching_weights(model, checkpoint["model"])
        print(format_partial_load_report("model", stats))
    else:
        model.load_state_dict(checkpoint["model"])
    if load_condition_encoder and "condition_encoder" in checkpoint:
        if partial_condition_encoder:
            stats = load_matching_weights(condition_encoder, checkpoint["condition_encoder"])
            print(format_partial_load_report("condition_encoder", stats))
        else:
            condition_encoder.load_state_dict(checkpoint["condition_encoder"])
    return int(checkpoint.get("step", 0))


def sample_train_timesteps(
    scheduler: NoiseScheduler,
    batch_size: int,
    device: torch.device,
    diffusion_config: dict[str, Any],
) -> torch.Tensor:
    min_timestep = int(diffusion_config.get("train_min_timestep", 0))
    max_timestep = int(diffusion_config.get("train_max_timestep", scheduler.num_train_timesteps - 1))
    min_timestep = max(0, min(min_timestep, scheduler.num_train_timesteps - 1))
    max_timestep = max(min_timestep, min(max_timestep, scheduler.num_train_timesteps - 1))
    return torch.randint(min_timestep, max_timestep + 1, (batch_size,), device=device, dtype=torch.long)


def make_diffusion_inputs(
    scheduler: NoiseScheduler,
    target_latent: torch.Tensor,
    condition: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    init_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if init_mode == "target":
        return scheduler.add_noise(target_latent, noise, timesteps), noise
    if init_mode == "condition":
        noisy = scheduler.add_noise(condition, noise, timesteps)
        return noisy, scheduler.noise_from_x0(noisy, target_latent, timesteps)
    raise ValueError(f"Unsupported diffusion init mode: {init_mode}")


def make_latents(
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    hr: torch.Tensor,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    dtype_name: str,
    train_condition_encoder: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = normalize_image(hr)
    lr_input = normalize_image(lr)
    with torch.no_grad():
        target_latent, _ = vae.encode(target)
    if train_condition_encoder:
        condition = condition_encoder(lr_input, domain_id)
    else:
        with torch.no_grad():
            condition = condition_encoder(lr_input, domain_id)
    return target_latent, condition


def evaluate(
    model: ConditionalUNet,
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    scheduler: NoiseScheduler,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    train_condition_encoder: bool,
    eval_timestep: int,
    init_mode: str,
    diffusion_config: dict[str, Any],
) -> dict[str, float]:
    model_was_training = model.training
    cond_was_training = condition_encoder.training
    model.eval()
    condition_encoder.eval()
    totals = {"noise_mse": 0.0, "x0_mse": 0.0, "decoded_mse": 0.0}
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)
            batch_size = int(hr.shape[0])
            timestep = torch.full(
                (batch_size,),
                min(eval_timestep, scheduler.num_train_timesteps - 1),
                device=device,
                dtype=torch.long,
            )
            with autocast_context(device, dtype_name):
                target_latent, condition = make_latents(
                    vae, condition_encoder, hr, lr, domain_id, dtype_name, train_condition_encoder=False
                )
                noise = torch.randn_like(target_latent)
                noisy, target_noise = make_diffusion_inputs(
                    scheduler, target_latent, condition, noise, timestep, init_mode
                )
                model_output = model(noisy, timestep, condition, domain_id)
                x0, predicted_noise = predict_x0_and_noise(
                    scheduler,
                    noisy,
                    timestep,
                    model_output,
                    condition,
                    diffusion_config,
                )
                decoded = vae.decode(x0)
                target = normalize_image(hr)
                noise_mse = F.mse_loss(predicted_noise, target_noise)
                x0_mse = F.mse_loss(x0, target_latent)
                decoded_mse = F.mse_loss(decoded, target)
            totals["noise_mse"] += float(noise_mse.detach().cpu()) * batch_size
            totals["x0_mse"] += float(x0_mse.detach().cpu()) * batch_size
            totals["decoded_mse"] += float(decoded_mse.detach().cpu()) * batch_size
            count += batch_size
    if model_was_training:
        model.train()
    condition_encoder.train(mode=cond_was_training)
    count = max(1, count)
    decoded_mse = totals["decoded_mse"] / count
    return {
        "eval/noise_mse": totals["noise_mse"] / count,
        "eval/x0_mse": totals["x0_mse"] / count,
        "eval/decoded_mse": decoded_mse,
        "eval/decoded_psnr": psnr_from_mse(decoded_mse),
        "eval/num_images": float(count),
    }


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed + rank)

    output_dir = Path(config["project"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    eval_dir = output_dir / "eval"
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")
    barrier(distributed)

    train_cfg = config["train"]
    device = torch.device(f"cuda:{local_rank}") if distributed and torch.cuda.is_available() else get_device(train_cfg.get("device", "auto"))
    dtype_name = train_cfg.get("dtype", "bf16")
    print_main(
        f"device={device} dtype={dtype_name} distributed={distributed} world_size={world_size}",
        rank,
    )

    train_dataset = make_dataset(config, split=config["data"].get("split", "train"), seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )
    fixed_sample_batch = make_fixed_sample_batch(config, seed=seed) if is_main_process(rank) else None
    if fixed_sample_batch is not None and is_main_process(rank):
        print_main(
            "sample_logging="
            f"split={fixed_sample_batch['split']} "
            f"indices={fixed_sample_batch['indices']} "
            f"count={len(fixed_sample_batch['path'])}",
            rank,
        )

    eval_cfg = config.get("eval", {})
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_loader = None
    eval_every = int(eval_cfg.get("every", 1000))
    eval_run_at_start = bool(eval_cfg.get("run_at_start", True))
    eval_timestep = int(eval_cfg.get("timestep", config.get("diffusion", {}).get("sample_timestep", 500)))
    diffusion_cfg = config.get("diffusion", {})
    train_init_mode = str(diffusion_cfg.get("train_init", "target"))
    eval_init_mode = str(eval_cfg.get("init", "target"))
    if eval_enabled and is_main_process(rank):
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
        print_main(
            "eval="
            f"split={eval_cfg.get('split', 'val')} "
            f"limit={eval_cfg.get('limit', 0)} "
            f"batch_size={eval_cfg.get('batch_size', train_cfg.get('batch_size', 1))} "
            f"timestep={eval_timestep} "
            f"init={eval_init_mode}",
            rank,
        )

    vae = load_autoencoder(config, device=device)
    condition_encoder, train_condition_encoder = load_condition_encoder(config, device=device)
    residual_refiner_teacher = load_residual_refiner_teacher(config, device=device)
    model = ConditionalUNet.from_config(config["model"]).to(device)
    scheduler = NoiseScheduler.from_config(diffusion_cfg)
    print_main(f"diffusion_timesteps={scheduler.num_train_timesteps} train_init={train_init_mode}", rank)

    parameters = list(model.parameters())
    if train_condition_encoder:
        parameters += list(condition_encoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, condition_encoder, optimizer, device)
        print_main(f"resumed step={start_step}", rank)
    elif args.init_checkpoint:
        init_step = load_model_weights(
            args.init_checkpoint,
            model,
            condition_encoder,
            device,
            load_condition_encoder=bool(args.init_condition_encoder),
            partial=bool(args.partial_init),
            partial_condition_encoder=bool(args.partial_init_condition_encoder),
        )
        print_main(
            f"initialized_from={args.init_checkpoint} source_step={init_step} "
            f"partial_init={bool(args.partial_init)} "
            f"loaded_init_condition_encoder={bool(args.init_condition_encoder)} "
            f"partial_init_condition_encoder={bool(args.partial_init_condition_encoder)}",
            rank,
        )

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if train_condition_encoder:
            condition_encoder = DistributedDataParallel(
                condition_encoder,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )

    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 1000))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    sample_every = int(train_cfg.get("sample_every", 500))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    loss_cfg = config.get("loss", {})
    noise_loss_weight = float(loss_cfg.get("noise_weight", 1.0))
    x0_loss_weight = float(loss_cfg.get("x0_weight", 0.0))
    decoded_loss_weight = float(loss_cfg.get("decoded_weight", 0.0))
    edge_loss_weight = float(loss_cfg.get("edge_weight", 0.0))
    highpass_loss_weight = float(loss_cfg.get("highpass_weight", 0.0))
    residual_edge_magnitude_loss_weight = float(loss_cfg.get("residual_edge_magnitude_weight", 0.0))
    residual_highpass_magnitude_loss_weight = float(loss_cfg.get("residual_highpass_magnitude_weight", 0.0))
    lowpass_anchor_loss_weight = float(loss_cfg.get("lowpass_anchor_weight", 0.0))
    detail_gate_anchor_loss_weight = float(loss_cfg.get("detail_gate_anchor_weight", 0.0))
    lowpass_anchor_kernel_size = int(loss_cfg.get("lowpass_anchor_kernel_size", 9))
    detail_gate_threshold = float(loss_cfg.get("detail_gate_threshold", 0.035))
    teacher_residual_loss_weight = float(loss_cfg.get("teacher_residual_weight", 0.0))
    teacher_highpass_loss_weight = float(loss_cfg.get("teacher_highpass_weight", 0.0))
    teacher_gate_loss_weight = float(loss_cfg.get("teacher_gate_weight", 0.0))
    teacher_x0_loss_weight = float(loss_cfg.get("teacher_x0_weight", 0.0))
    teacher_loss_enabled = (
        teacher_residual_loss_weight > 0.0
        or teacher_highpass_loss_weight > 0.0
        or teacher_gate_loss_weight > 0.0
        or teacher_x0_loss_weight > 0.0
    )
    if teacher_loss_enabled and residual_refiner_teacher is None:
        raise ValueError("Teacher loss weights are non-zero, but teacher.residual_refiner.enabled is false.")
    charbonnier_eps = float(loss_cfg.get("charbonnier_eps", 1e-3))
    best_metric = str(eval_cfg.get("best_metric", "eval/noise_mse"))
    best_mode = str(eval_cfg.get("best_mode", "min"))
    best_checkpoint = str(eval_cfg.get("best_checkpoint", "best_eval_noise.pt"))

    run = init_wandb(config, output_dir, unwrap_model(model)) if is_main_process(rank) else None
    wandb_log(
        run,
        {
            "dataset/num_images": len(train_dataset),
            "train/batch_size": int(train_cfg.get("batch_size", 1)),
            "train/grad_accum_steps": grad_accum_steps,
            "train/world_size": world_size,
            "train/effective_batch_size": int(train_cfg.get("batch_size", 1)) * grad_accum_steps * world_size,
        },
        step=start_step,
    )

    model.train()
    condition_encoder.train(mode=train_condition_encoder)
    step = start_step
    best_eval = float("-inf") if best_mode == "max" else float("inf")
    last_log = time.time()
    last_log_step = step
    optimizer.zero_grad(set_to_none=True)
    epoch = 0

    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            hr = batch["hr"].to(device, non_blocking=True)
            lr = batch["lr"].to(device, non_blocking=True)
            domain_id = batch["domain_id"].to(device, non_blocking=True)

            sync_gradients = step % grad_accum_steps == 0
            sync_context = (
                model.no_sync()
                if distributed and isinstance(model, DistributedDataParallel) and not sync_gradients
                else contextlib.nullcontext()
            )
            with sync_context:
                with autocast_context(device, dtype_name):
                    target_latent, condition = make_latents(
                        vae, condition_encoder, hr, lr, domain_id, dtype_name, train_condition_encoder
                    )
                    noise = torch.randn_like(target_latent)
                    timesteps = sample_train_timesteps(
                        scheduler,
                        int(target_latent.shape[0]),
                        device=device,
                        diffusion_config=diffusion_cfg,
                    )
                    noisy, target_noise = make_diffusion_inputs(
                        scheduler, target_latent, condition, noise, timesteps, train_init_mode
                    )
                    model_output = model(noisy, timesteps, condition, domain_id)
                    predicted_x0, predicted_noise = predict_x0_and_noise(
                        scheduler,
                        noisy,
                        timesteps,
                        model_output,
                        condition,
                        diffusion_cfg,
                    )
                    if teacher_loss_enabled:
                        with torch.no_grad():
                            teacher_lr_input = normalize_image(lr)
                            teacher_x0, teacher_residual, teacher_gate = residual_refiner_teacher(
                                condition.detach(),
                                teacher_lr_input,
                                domain_id,
                            )
                        predicted_residual, predicted_gate = gated_residual_components(
                            model_output,
                            condition,
                            diffusion_cfg,
                        )
                        teacher_residual_loss = charbonnier_loss(
                            predicted_residual,
                            teacher_residual.detach(),
                            eps=charbonnier_eps,
                        )
                        teacher_highpass_loss = laplacian_loss(
                            predicted_residual,
                            teacher_residual.detach(),
                            eps=charbonnier_eps,
                        )
                        teacher_gate_loss = charbonnier_loss(
                            predicted_gate,
                            teacher_gate.detach(),
                            eps=charbonnier_eps,
                        )
                        teacher_x0_loss = charbonnier_loss(
                            predicted_x0,
                            teacher_x0.detach(),
                            eps=charbonnier_eps,
                        )
                    else:
                        teacher_residual_loss = torch.zeros((), device=device, dtype=predicted_x0.dtype)
                        teacher_highpass_loss = torch.zeros((), device=device, dtype=predicted_x0.dtype)
                        teacher_gate_loss = torch.zeros((), device=device, dtype=predicted_x0.dtype)
                        teacher_x0_loss = torch.zeros((), device=device, dtype=predicted_x0.dtype)
                    noise_loss = F.mse_loss(predicted_noise, target_noise)
                    if x0_loss_weight > 0.0:
                        x0_loss = F.mse_loss(predicted_x0, target_latent)
                    else:
                        x0_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                    if (
                        decoded_loss_weight > 0.0
                        or edge_loss_weight > 0.0
                        or highpass_loss_weight > 0.0
                        or residual_edge_magnitude_loss_weight > 0.0
                        or residual_highpass_magnitude_loss_weight > 0.0
                        or lowpass_anchor_loss_weight > 0.0
                        or detail_gate_anchor_loss_weight > 0.0
                    ):
                        decoded = vae.decode(predicted_x0)
                        target_image = normalize_image(hr)
                        decoded_loss = charbonnier_loss(decoded, target_image, eps=charbonnier_eps)
                        edge_loss = sobel_edge_loss(decoded, target_image, eps=charbonnier_eps)
                        highpass_loss = laplacian_loss(decoded, target_image, eps=charbonnier_eps)
                        if (
                            residual_edge_magnitude_loss_weight > 0.0
                            or residual_highpass_magnitude_loss_weight > 0.0
                            or lowpass_anchor_loss_weight > 0.0
                            or detail_gate_anchor_loss_weight > 0.0
                        ):
                            with torch.no_grad():
                                condition_decoded = vae.decode(condition.detach())
                        else:
                            condition_decoded = None
                        if residual_edge_magnitude_loss_weight > 0.0 or residual_highpass_magnitude_loss_weight > 0.0:
                            if condition_decoded is None:
                                raise RuntimeError("condition_decoded is required for residual magnitude losses")
                            residual_edge_magnitude_loss = sobel_residual_magnitude_loss(
                                decoded,
                                target_image,
                                condition_decoded,
                                eps=charbonnier_eps,
                            )
                            residual_highpass_magnitude_loss = laplacian_residual_magnitude_loss(
                                decoded,
                                target_image,
                                condition_decoded,
                                eps=charbonnier_eps,
                            )
                        else:
                            residual_edge_magnitude_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                            residual_highpass_magnitude_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        if lowpass_anchor_loss_weight > 0.0:
                            if condition_decoded is None:
                                raise RuntimeError("condition_decoded is required for lowpass anchor loss")
                            lowpass_anchor = lowpass_anchor_loss(
                                decoded,
                                condition_decoded,
                                kernel_size=lowpass_anchor_kernel_size,
                                eps=charbonnier_eps,
                            )
                        else:
                            lowpass_anchor = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        if detail_gate_anchor_loss_weight > 0.0:
                            if condition_decoded is None:
                                raise RuntimeError("condition_decoded is required for detail gate anchor loss")
                            detail_gate_anchor = laplacian_detail_gate_anchor_loss(
                                decoded,
                                target_image,
                                condition_decoded,
                                threshold=detail_gate_threshold,
                                eps=charbonnier_eps,
                            )
                        else:
                            detail_gate_anchor = torch.zeros((), device=device, dtype=noise_loss.dtype)
                    else:
                        decoded_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        edge_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        highpass_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        residual_edge_magnitude_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        residual_highpass_magnitude_loss = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        lowpass_anchor = torch.zeros((), device=device, dtype=noise_loss.dtype)
                        detail_gate_anchor = torch.zeros((), device=device, dtype=noise_loss.dtype)
                    loss = (
                        noise_loss_weight * noise_loss
                        + x0_loss_weight * x0_loss
                        + decoded_loss_weight * decoded_loss
                        + edge_loss_weight * edge_loss
                        + highpass_loss_weight * highpass_loss
                        + residual_edge_magnitude_loss_weight * residual_edge_magnitude_loss
                        + residual_highpass_magnitude_loss_weight * residual_highpass_magnitude_loss
                        + lowpass_anchor_loss_weight * lowpass_anchor
                        + detail_gate_anchor_loss_weight * detail_gate_anchor
                        + teacher_residual_loss_weight * teacher_residual_loss
                        + teacher_highpass_loss_weight * teacher_highpass_loss
                        + teacher_gate_loss_weight * teacher_gate_loss
                        + teacher_x0_loss_weight * teacher_x0_loss
                    )
                    scaled_loss = loss / grad_accum_steps

                scaled_loss.backward()
            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if is_main_process(rank) and (step % log_every == 0 or step == 1):
                elapsed = max(1e-6, time.time() - last_log)
                interval_steps = max(1, step - last_log_step)
                last_log = time.time()
                last_log_step = step
                print(
                    f"step={step} loss={float(loss.detach().cpu()):.5f} "
                    f"noise_mse={float(noise_loss.detach().cpu()):.5f} "
                    f"x0_mse={float(x0_loss.detach().cpu()):.5f} "
                    f"decoded={float(decoded_loss.detach().cpu()):.5f} "
                    f"edge={float(edge_loss.detach().cpu()):.5f} "
                    f"highpass={float(highpass_loss.detach().cpu()):.5f} "
                    f"res_edge_mag={float(residual_edge_magnitude_loss.detach().cpu()):.5f} "
                    f"res_high_mag={float(residual_highpass_magnitude_loss.detach().cpu()):.5f} "
                    f"low_anchor={float(lowpass_anchor.detach().cpu()):.5f} "
                    f"detail_gate={float(detail_gate_anchor.detach().cpu()):.5f} "
                    f"teacher_res={float(teacher_residual_loss.detach().cpu()):.5f} "
                    f"teacher_high={float(teacher_highpass_loss.detach().cpu()):.5f} "
                    f"teacher_gate={float(teacher_gate_loss.detach().cpu()):.5f} "
                    f"teacher_x0={float(teacher_x0_loss.detach().cpu()):.5f} "
                    f"steps_per_sec={interval_steps / elapsed:.2f}"
                )
                wandb_log(
                    run,
                    {
                        "train/loss": float(loss.detach().cpu()),
                        "train/noise_mse": float(noise_loss.detach().cpu()),
                        "train/x0_mse": float(x0_loss.detach().cpu()),
                        "train/decoded_charbonnier": float(decoded_loss.detach().cpu()),
                        "train/edge_sobel": float(edge_loss.detach().cpu()),
                        "train/highpass_laplacian": float(highpass_loss.detach().cpu()),
                        "train/residual_edge_magnitude": float(residual_edge_magnitude_loss.detach().cpu()),
                        "train/residual_highpass_magnitude": float(residual_highpass_magnitude_loss.detach().cpu()),
                        "train/lowpass_anchor": float(lowpass_anchor.detach().cpu()),
                        "train/detail_gate_anchor": float(detail_gate_anchor.detach().cpu()),
                        "train/teacher_residual": float(teacher_residual_loss.detach().cpu()),
                        "train/teacher_highpass": float(teacher_highpass_loss.detach().cpu()),
                        "train/teacher_gate": float(teacher_gate_loss.detach().cpu()),
                        "train/teacher_x0": float(teacher_x0_loss.detach().cpu()),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "system/steps_per_sec": interval_steps / elapsed,
                    },
                    step=step,
                )

            should_eval = (
                eval_enabled
                and eval_every > 0
                and (step % eval_every == 0 or (step == 1 and eval_run_at_start))
            )
            if should_eval:
                barrier(distributed)
                if is_main_process(rank) and eval_loader is not None:
                    metrics = evaluate(
                        unwrap_model(model),
                        vae,
                        unwrap_model(condition_encoder),
                        scheduler,
                        eval_loader,
                        device,
                        dtype_name,
                        train_condition_encoder,
                        eval_timestep,
                        eval_init_mode,
                        diffusion_cfg,
                    )
                    (eval_dir / f"step_{step:07d}_metrics.json").write_text(
                        json.dumps({"step": step, "metrics": metrics}, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"eval step={step} noise_mse={metrics['eval/noise_mse']:.5f} "
                        f"decoded_psnr={metrics['eval/decoded_psnr']:.2f}"
                    )
                    wandb_log(run, metrics, step=step)
                    metric_value = metrics[best_metric]
                    improved = metric_value > best_eval if best_mode == "max" else metric_value < best_eval
                    if improved:
                        best_eval = metric_value
                        save_checkpoint(checkpoints_dir / best_checkpoint, model, condition_encoder, optimizer, step, config)
                barrier(distributed)

            if step % sample_every == 0 or step == 1:
                barrier(distributed)
            if is_main_process(rank) and (step % sample_every == 0 or step == 1):
                sample_source = fixed_sample_batch if fixed_sample_batch is not None else batch
                sample_init_mode = str(
                    config.get("logging", {}).get("samples", {}).get("init", eval_init_mode)
                )
                with torch.no_grad():
                    sample_hr = sample_source["hr"].to(device, non_blocking=True)
                    sample_lr = sample_source["lr"].to(device, non_blocking=True)
                    sample_domain = sample_source["domain_id"].to(device, non_blocking=True)
                    sample_timestep = int(config.get("diffusion", {}).get("sample_timestep", eval_timestep))
                    timestep = torch.full(
                        (int(sample_hr.shape[0]),),
                        min(sample_timestep, scheduler.num_train_timesteps - 1),
                        device=device,
                        dtype=torch.long,
                    )
                    with autocast_context(device, dtype_name):
                        sample_target_latent, sample_condition = make_latents(
                            vae, condition_encoder, sample_hr, sample_lr, sample_domain, dtype_name, False
                        )
                        sample_noise = torch.randn_like(sample_target_latent)
                        sample_noisy, _ = make_diffusion_inputs(
                            scheduler,
                            sample_target_latent,
                            sample_condition,
                            sample_noise,
                            timestep,
                            sample_init_mode,
                        )
                        sample_model_output = unwrap_model(model)(sample_noisy, timestep, sample_condition, sample_domain)
                        sample_x0, _ = predict_x0_and_noise(
                            scheduler,
                            sample_noisy,
                            timestep,
                            sample_model_output,
                            sample_condition,
                            diffusion_cfg,
                        )
                        sample_decoded = vae.decode(sample_x0)
                    sample_count = sample_hr.shape[0]
                    lr_display = F.interpolate(sample_source["lr"].float().cpu(), size=sample_hr.shape[-2:], mode="nearest")
                    gt = sample_hr.float().cpu()
                    pred = denormalize(sample_decoded).float().cpu()

                    lr_path = samples_dir / f"step_{step:07d}_lr.png"
                    gt_path = samples_dir / f"step_{step:07d}_gt.png"
                    pred_path = samples_dir / f"step_{step:07d}_pred_x0.png"
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
                                "samples/PredX0": [
                                    wandb.Image(tensor_to_pil(image), caption=caption)
                                    for image, caption in zip(pred, captions, strict=True)
                                ],
                            },
                            step=step,
                        )
            if step % sample_every == 0 or step == 1:
                barrier(distributed)

            if step % save_every == 0 or step == max_steps:
                barrier(distributed)
                if is_main_process(rank):
                    save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, condition_encoder, optimizer, step, config)
                    save_checkpoint(checkpoints_dir / "latest.pt", model, condition_encoder, optimizer, step, config)
                barrier(distributed)

            if step >= max_steps:
                break
        epoch += 1

    print_main(f"finished step={step}", rank)
    if run is not None:
        run.finish()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
