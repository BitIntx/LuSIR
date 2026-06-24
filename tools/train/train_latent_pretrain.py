from __future__ import annotations

import argparse
import contextlib
import json
import math
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.detail_mask import detail_need_components, top_fraction_mask
from sr_diffusion.losses import FrozenVGGFeatureLoss
from sr_diffusion.losses.reconstruction import (
    charbonnier_loss,
    laplacian_loss,
    laplacian_residual_magnitude_loss,
    sobel_edge_loss,
)
from sr_diffusion.models import AutoencoderKL, LRToLatentPredictor, LatentResidualRefiner
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
    parser.add_argument("--override-lr", type=float, default=None, help="Override optimizer LR after resume/init.")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--partial-init",
        action="store_true",
        help="Load only shape-compatible tensors from --init-checkpoint. Useful when widening or deepening the model.",
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


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor, peak: float = 2.0) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return 20.0 * math.log10(float(peak)) - 10.0 * torch.log10(mse.clamp_min(1e-12))


def metric_highpass(x: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return x.float()
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


def selected_capture(energy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (energy.float() * mask.float()).flatten(1).sum(dim=1) / energy.float().flatten(1).sum(dim=1).clamp_min(1e-12)


def masked_charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[0] != prediction.shape[0] or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B, 1, H, W], got {tuple(mask.shape)}")
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(mask.float(), size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    weight = mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
    per_pixel = torch.sqrt((prediction - target).pow(2) + eps * eps).mean(dim=1, keepdim=True)
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1e-8)


def local_highpass_energy_hinge_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    highpass_kernel: int = 15,
    patch_kernel: int = 9,
    excess_margin: float = 0.002,
    missing_margin: float = 0.002,
    temperature: float = 0.006,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft hinges for unsupported excess and missing local detail."""
    prediction_highpass = metric_highpass(denormalize(prediction), highpass_kernel)
    target_highpass = metric_highpass(denormalize(target).detach(), highpass_kernel)
    prediction_detail = prediction_highpass.abs().mean(dim=1, keepdim=True)
    target_detail = target_highpass.abs().mean(dim=1, keepdim=True)
    prediction_aligned_detail = (prediction_highpass * target_highpass.sign()).mean(dim=1, keepdim=True)
    patch_kernel = int(patch_kernel)
    if patch_kernel > 1:
        if patch_kernel % 2 == 0:
            raise ValueError(f"artifact_excess.patch_kernel must be odd, got {patch_kernel}")
        padding = patch_kernel // 2
        prediction_detail = F.avg_pool2d(
            F.pad(prediction_detail, (padding, padding, padding, padding), mode="reflect"),
            kernel_size=patch_kernel,
            stride=1,
        )
        target_detail = F.avg_pool2d(
            F.pad(target_detail, (padding, padding, padding, padding), mode="reflect"),
            kernel_size=patch_kernel,
            stride=1,
        )
        prediction_aligned_detail = F.avg_pool2d(
            F.pad(prediction_aligned_detail, (padding, padding, padding, padding), mode="reflect"),
            kernel_size=patch_kernel,
            stride=1,
        )
    temperature = max(float(temperature), 1e-8)
    excess = F.softplus((prediction_detail - target_detail - float(excess_margin)) / temperature) * temperature
    missing = (
        F.softplus((target_detail - prediction_aligned_detail - float(missing_margin)) / temperature) * temperature
    )
    excess_active = (prediction_detail > target_detail + float(excess_margin)).float().mean()
    missing_active = (target_detail > prediction_aligned_detail + float(missing_margin)).float().mean()
    return excess.mean(), missing.mean(), excess_active, missing_active


def artifact_excess_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    highpass_kernel: int = 15,
    patch_kernel: int = 9,
    margin: float = 0.002,
    temperature: float = 0.006,
) -> tuple[torch.Tensor, torch.Tensor]:
    excess, _, excess_active, _ = local_highpass_energy_hinge_losses(
        prediction,
        target,
        highpass_kernel=highpass_kernel,
        patch_kernel=patch_kernel,
        excess_margin=margin,
        missing_margin=margin,
        temperature=temperature,
    )
    return excess, excess_active


def make_stage2_detail_training_mask(
    decoded: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    source = str(config.get("source", "prediction_missing"))
    if source in {"prediction_missing", "decoded_missing"}:
        base = denormalize(decoded.detach())
    elif source in {"reference_missing", "bicubic_missing"}:
        base = denormalize(reference.detach())
    else:
        raise ValueError(f"unsupported detail_weighted.source {source!r}")
    target_01 = denormalize(target.detach())
    components = detail_need_components(
        base.float(),
        target_01.float(),
        highpass_kernel=int(config.get("highpass_kernel", 15)),
        patch_kernel=int(config.get("patch_kernel", 9)),
        score_quantile=float(config.get("score_quantile", 0.95)),
    )
    top_fraction = float(config.get("top_fraction", 0.20))
    mask = top_fraction_mask(components["score"], top_fraction)
    if str(config.get("top_mode", "binary")) == "soft":
        mask = mask * components["score"]
    floor = float(config.get("mask_floor", 0.0))
    if floor > 0.0:
        mask = mask * (1.0 - floor) + floor
    return mask.clamp(0.0, 1.0).to(device=decoded.device, dtype=decoded.dtype)


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
    artifact_excess_weight = float(loss_config.get("artifact_excess_weight", 0.0))
    artifact_missing_weight = float(loss_config.get("artifact_missing_weight", 0.0))
    detail_weighted_config = loss_config.get("detail_weighted", {}) or {}
    detail_decoded_weight = float(detail_weighted_config.get("decoded_weight", 0.0))
    detail_highpass_weight = float(detail_weighted_config.get("highpass_weight", 0.0))
    if (
        decoded_weight > 0.0
        or edge_weight > 0.0
        or highpass_weight > 0.0
        or highpass_magnitude_weight > 0.0
        or perceptual_weight > 0.0
        or artifact_excess_weight > 0.0
        or artifact_missing_weight > 0.0
        or detail_decoded_weight > 0.0
        or detail_highpass_weight > 0.0
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
        if artifact_excess_weight > 0.0 or artifact_missing_weight > 0.0:
            artifact_config = loss_config.get("artifact_excess", {}) or {}
            artifact_excess, artifact_missing, artifact_excess_active, artifact_missing_active = (
                local_highpass_energy_hinge_losses(
                    decoded,
                    target_image,
                    highpass_kernel=int(artifact_config.get("highpass_kernel", 15)),
                    patch_kernel=int(artifact_config.get("patch_kernel", 9)),
                    excess_margin=float(artifact_config.get("margin", 0.002)),
                    missing_margin=float(
                        artifact_config.get("missing_margin", artifact_config.get("margin", 0.002))
                    ),
                    temperature=float(artifact_config.get("temperature", 0.006)),
                )
            )
        else:
            artifact_excess = prediction.new_zeros(())
            artifact_missing = prediction.new_zeros(())
            artifact_excess_active = prediction.new_zeros(())
            artifact_missing_active = prediction.new_zeros(())
        if detail_decoded_weight > 0.0 or detail_highpass_weight > 0.0:
            detail_mask = make_stage2_detail_training_mask(
                decoded,
                target_image,
                reference_image,
                detail_weighted_config,
            )
            detail_decoded = masked_charbonnier_loss(decoded, target_image, detail_mask, eps=eps)
            decoded_highpass = metric_highpass(decoded, int(detail_weighted_config.get("laplacian_kernel", 3)))
            target_highpass = metric_highpass(target_image, int(detail_weighted_config.get("laplacian_kernel", 3)))
            detail_highpass = masked_charbonnier_loss(decoded_highpass, target_highpass, detail_mask, eps=eps)
            detail_mask_mean = detail_mask.float().mean()
        else:
            detail_mask = prediction.new_empty(0)
            detail_decoded = prediction.new_zeros(())
            detail_highpass = prediction.new_zeros(())
            detail_mask_mean = prediction.new_zeros(())
    else:
        decoded = prediction.new_empty(0)
        pixel = prediction.new_zeros(())
        edge = prediction.new_zeros(())
        highpass = prediction.new_zeros(())
        highpass_magnitude = prediction.new_zeros(())
        perceptual = prediction.new_zeros(())
        artifact_excess = prediction.new_zeros(())
        artifact_missing = prediction.new_zeros(())
        artifact_excess_active = prediction.new_zeros(())
        artifact_missing_active = prediction.new_zeros(())
        detail_mask = prediction.new_empty(0)
        detail_decoded = prediction.new_zeros(())
        detail_highpass = prediction.new_zeros(())
        detail_mask_mean = prediction.new_zeros(())
    total = (
        float(loss_config.get("latent_weight", 1.0)) * latent
        + decoded_weight * pixel
        + edge_weight * edge
        + highpass_weight * highpass
        + highpass_magnitude_weight * highpass_magnitude
        + perceptual_weight * perceptual
        + artifact_excess_weight * artifact_excess
        + artifact_missing_weight * artifact_missing
        + detail_decoded_weight * detail_decoded
        + detail_highpass_weight * detail_highpass
    )
    return total, {
        "latent": latent,
        "decoded": pixel,
        "edge": edge,
        "highpass": highpass,
        "highpass_magnitude": highpass_magnitude,
        "perceptual": perceptual,
        "artifact_excess": artifact_excess,
        "artifact_missing": artifact_missing,
        "artifact_excess_active": artifact_excess_active,
        "artifact_missing_active": artifact_missing_active,
        "detail_decoded": detail_decoded,
        "detail_highpass": detail_highpass,
        "detail_mask_mean": detail_mask_mean,
        "detail_mask": detail_mask,
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


def make_dataset(
    config: dict[str, Any],
    split: str,
    seed: int,
    deterministic: bool | None = None,
    data_overrides: dict[str, Any] | None = None,
) -> ManifestImageDataset:
    data_config = dict(config["data"])
    if data_overrides:
        data_config.update(data_overrides)
    if (
        deterministic is None
        and split == data_config.get("split", "train")
        and bool(data_config.get("deterministic_train", False))
    ):
        deterministic = True
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
        max_items=data_config.get("max_items"),
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
        project=wandb_cfg.get("project", "LuSIR"),
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
    if isinstance(model, LatentResidualRefiner) and not any(key.startswith("base.") for key in checkpoint["model"]):
        model.load_base_state_dict(checkpoint["model"])
        print(f"loaded_base_model={path} tensors={len(checkpoint['model'])}")
        return int(checkpoint.get("step", 0))
    if partial:
        stats = load_matching_weights(model, checkpoint["model"])
        print(format_partial_load_report("model", stats))
    else:
        model.load_state_dict(checkpoint["model"])
    return int(checkpoint.get("step", 0))


def build_stage2_model(model_config: dict[str, Any]) -> torch.nn.Module:
    model_type = str(model_config.get("type", "lr_to_latent_predictor"))
    if model_type == "lr_to_latent_predictor":
        return LRToLatentPredictor.from_config(model_config)
    if model_type == "latent_residual_refiner":
        return LatentResidualRefiner.from_config(model_config)
    raise ValueError(f"Unsupported Stage 2 model type: {model_type}")


def matches_any_pattern(name: str, patterns: list[str]) -> bool:
    return any(pattern and pattern in name for pattern in patterns)


def build_optimizer(
    model: torch.nn.Module,
    train_config: dict[str, Any],
    rank: int,
) -> torch.optim.Optimizer:
    base_lr = float(train_config.get("lr", 2e-4))
    base_weight_decay = float(train_config.get("weight_decay", 0.0))
    parameter_groups = train_config.get("parameter_groups") or train_config.get("param_groups") or []
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    if not parameter_groups:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_weight_decay)
        for group in optimizer.param_groups:
            group.setdefault("name", "default")
            group.setdefault("base_lr", group["lr"])
        return optimizer

    assigned: set[int] = set()
    optimizer_groups: list[dict[str, Any]] = []
    for index, group_config in enumerate(parameter_groups):
        group_name = str(group_config.get("name", f"group_{index}"))
        patterns = [str(pattern) for pattern in group_config.get("patterns", [])]
        if not patterns:
            raise ValueError(f"parameter_groups[{index}] must define at least one pattern")
        group_parameters = []
        for name, parameter in named_parameters:
            parameter_id = id(parameter)
            if parameter_id in assigned:
                continue
            if matches_any_pattern(name, patterns):
                group_parameters.append(parameter)
                assigned.add(parameter_id)
        if not group_parameters:
            print_main(f"optimizer_group={group_name} matched no parameters patterns={patterns}", rank)
            continue
        lr = float(group_config.get("lr", base_lr))
        weight_decay = float(group_config.get("weight_decay", base_weight_decay))
        optimizer_groups.append(
            {
                "params": group_parameters,
                "lr": lr,
                "base_lr": lr,
                "weight_decay": weight_decay,
                "name": group_name,
            }
        )

    default_parameters = [parameter for _, parameter in named_parameters if id(parameter) not in assigned]
    if default_parameters:
        optimizer_groups.append(
            {
                "params": default_parameters,
                "lr": base_lr,
                "base_lr": base_lr,
                "weight_decay": base_weight_decay,
                "name": "default",
            }
        )
    if not optimizer_groups:
        raise ValueError("No trainable optimizer parameter groups were created")

    optimizer = torch.optim.AdamW(optimizer_groups)
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
    return optimizer


def format_optimizer_groups(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        parameters = [parameter for parameter in group["params"] if parameter.requires_grad]
        parameter_count = sum(parameter.numel() for parameter in parameters)
        parts.append(
            f"{name}: tensors={len(parameters)} params={parameter_count:,} "
            f"lr={float(group['lr']):.8g} wd={float(group.get('weight_decay', 0.0)):.8g}"
        )
    return "; ".join(parts)


def scheduler_enabled(scheduler_config: dict[str, Any]) -> bool:
    scheduler_type = str(scheduler_config.get("type", "none")).lower()
    return bool(scheduler_config) and bool(scheduler_config.get("enabled", True)) and scheduler_type not in {
        "",
        "none",
        "constant",
    }


def apply_lr_schedule(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
    update_step: int,
    total_updates: int,
) -> float:
    scheduler_type = str(scheduler_config.get("type", "none")).lower()
    warmup_updates = max(0, int(scheduler_config.get("warmup_updates", 0)))
    min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.0))
    update_step = max(1, int(update_step))
    total_updates = max(1, int(total_updates))

    if warmup_updates > 0 and update_step <= warmup_updates:
        factor = update_step / warmup_updates
    elif scheduler_type == "warmup_cosine":
        decay_updates = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, (update_step - warmup_updates) / decay_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    elif scheduler_type == "constant_with_warmup":
        factor = 1.0
    else:
        raise ValueError(f"Unsupported train.scheduler.type: {scheduler_type}")

    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", group["lr"]))
        group["lr"] = base_lr * factor
    return factor


def optimizer_lr_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}")).replace("/", "_")
        metrics[f"train/lr/{name}"] = float(group["lr"])
    default_group = next((group for group in optimizer.param_groups if group.get("name") == "default"), None)
    if default_group is not None:
        metrics["train/lr"] = float(default_group["lr"])
    elif optimizer.param_groups:
        metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
    return metrics


def format_optimizer_lrs(optimizer: torch.optim.Optimizer) -> str:
    return ",".join(
        f"{str(group.get('name', f'group_{index}'))}:{float(group['lr']):.3g}"
        for index, group in enumerate(optimizer.param_groups)
    )


def add_eval_selection_metric(metrics: dict[str, float], eval_config: dict[str, Any]) -> None:
    selection = eval_config.get("selection") or {}
    if not selection:
        return
    name = str(selection.get("name", "eval/selection_score"))
    terms = selection.get("terms") or []
    if not terms:
        raise ValueError("eval.selection.terms must contain at least one weighted metric")
    score = 0.0
    for term in terms:
        metric_name = str(term["metric"])
        if metric_name not in metrics:
            raise KeyError(f"eval.selection metric not found: {metric_name}")
        score += float(term.get("weight", 1.0)) * float(metrics[metric_name])

    valid = True
    for guardrail in selection.get("guardrails") or []:
        metric_name = str(guardrail["metric"])
        if metric_name not in metrics:
            raise KeyError(f"eval.selection guardrail metric not found: {metric_name}")
        value = float(metrics[metric_name])
        if "min" in guardrail and value < float(guardrail["min"]):
            valid = False
        if "max" in guardrail and value > float(guardrail["max"]):
            valid = False

    metrics[f"{name}_raw"] = score
    metrics["eval/selection_valid"] = float(valid)
    metrics[name] = score if valid else float(selection.get("invalid_value", -1.0e9))


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
        "decoded_mean_psnr": 0.0,
        "decoded_ssim": 0.0,
        "decoded_edge": 0.0,
        "decoded_highpass": 0.0,
        "decoded_highpass_energy_ratio": 0.0,
        "decoded_highpass_l1": 0.0,
        "decoded_missing_energy": 0.0,
        "decoded_excess_energy": 0.0,
        "decoded_top_missing_capture": 0.0,
        "decoded_top_excess_capture": 0.0,
        "perceptual": 0.0,
        "laplacian_energy_ratio": 0.0,
        "oracle_decoded_mse": 0.0,
        "oracle_decoded_mean_psnr": 0.0,
        "oracle_laplacian_energy_ratio": 0.0,
    }
    count = 0
    detail_eval_config = loss_config.get("detail_eval", {})
    detail_highpass_kernel = int(detail_eval_config.get("highpass_kernel", 15))
    detail_patch_kernel = int(detail_eval_config.get("patch_kernel", 9))
    detail_score_quantile = float(detail_eval_config.get("score_quantile", 0.95))
    detail_top_fraction = float(detail_eval_config.get("top_fraction", 0.10))
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
            decoded_01 = denormalize(decoded).float()
            oracle_decoded_01 = denormalize(oracle_decoded).float()
            target_01 = hr.float()
            target_laplacian = laplacian_response(target)
            decoded_laplacian = laplacian_response(decoded)
            oracle_laplacian = laplacian_response(oracle_decoded)
            target_energy = target_laplacian.abs().flatten(1).mean(dim=1).clamp_min(1e-12)
            decoded_energy_ratio = decoded_laplacian.abs().flatten(1).mean(dim=1) / target_energy
            oracle_energy_ratio = oracle_laplacian.abs().flatten(1).mean(dim=1) / target_energy
            target_highpass = metric_highpass(target_01, detail_highpass_kernel)
            decoded_highpass = metric_highpass(decoded_01, detail_highpass_kernel)
            target_highpass_energy = target_highpass.abs().flatten(1).mean(dim=1).clamp_min(1e-12)
            decoded_highpass_ratio = decoded_highpass.abs().flatten(1).mean(dim=1) / target_highpass_energy
            decoded_detail = detail_need_components(
                decoded_01,
                target_01,
                highpass_kernel=detail_highpass_kernel,
                patch_kernel=detail_patch_kernel,
                score_quantile=detail_score_quantile,
            )
            decoded_detail_mask = top_fraction_mask(decoded_detail["score"], detail_top_fraction)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["latent_loss"] += float(components["latent"].detach().cpu()) * batch_size
            totals["latent_mse"] += float(latent_mse.detach().cpu()) * batch_size
            totals["decoded_mse"] += float(decoded_mse.detach().cpu()) * batch_size
            totals["decoded_mean_psnr"] += float(psnr_per_image(decoded, target).sum().cpu())
            totals["decoded_ssim"] += float(ssim_per_image(decoded_01, target_01).sum().cpu())
            totals["decoded_edge"] += float(sobel_edge_loss(decoded, target).detach().cpu()) * batch_size
            totals["decoded_highpass"] += float(laplacian_loss(decoded, target).detach().cpu()) * batch_size
            totals["decoded_highpass_energy_ratio"] += float(decoded_highpass_ratio.sum().cpu())
            totals["decoded_highpass_l1"] += float((decoded_highpass - target_highpass).abs().flatten(1).mean(dim=1).sum().cpu())
            totals["decoded_missing_energy"] += float(decoded_detail["missing"].flatten(1).mean(dim=1).sum().cpu())
            totals["decoded_excess_energy"] += float(decoded_detail["excess"].flatten(1).mean(dim=1).sum().cpu())
            totals["decoded_top_missing_capture"] += float(
                selected_capture(decoded_detail["missing"], decoded_detail_mask).sum().cpu()
            )
            totals["decoded_top_excess_capture"] += float(
                selected_capture(decoded_detail["excess"], decoded_detail_mask).sum().cpu()
            )
            totals["perceptual"] += float(components["perceptual"].detach().cpu()) * batch_size
            totals["laplacian_energy_ratio"] += float(decoded_energy_ratio.sum().cpu())
            totals["oracle_decoded_mse"] += float(oracle_decoded_mse.detach().cpu()) * batch_size
            totals["oracle_decoded_mean_psnr"] += float(psnr_per_image(oracle_decoded, target).sum().cpu())
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
        "eval/decoded_mean_psnr": totals["decoded_mean_psnr"] / count,
        "eval/decoded_ssim": totals["decoded_ssim"] / count,
        "eval/decoded_edge": totals["decoded_edge"] / count,
        "eval/decoded_highpass": totals["decoded_highpass"] / count,
        "eval/highpass_energy_ratio": totals["decoded_highpass_energy_ratio"] / count,
        "eval/highpass_l1": totals["decoded_highpass_l1"] / count,
        "eval/missing_energy": totals["decoded_missing_energy"] / count,
        "eval/excess_energy": totals["decoded_excess_energy"] / count,
        f"eval/top{detail_top_fraction:.2f}_missing_capture": totals["decoded_top_missing_capture"] / count,
        f"eval/top{detail_top_fraction:.2f}_excess_capture": totals["decoded_top_excess_capture"] / count,
        "eval/perceptual": totals["perceptual"] / count,
        "eval/laplacian_energy_ratio": totals["laplacian_energy_ratio"] / count,
        "eval/oracle_decoded_psnr": psnr_from_mse(totals["oracle_decoded_mse"] / count),
        "eval/oracle_decoded_mean_psnr": totals["oracle_decoded_mean_psnr"] / count,
        "eval/oracle_laplacian_energy_ratio": totals["oracle_laplacian_energy_ratio"] / count,
        "eval/num_images": float(count),
    }
    detail_score_weight = float(loss_config.get("detail_score_weight", 0.0))
    metrics["eval/psnr_detail_score"] = metrics["eval/decoded_psnr"] + detail_score_weight * metrics[
        "eval/laplacian_energy_ratio"
    ]
    metrics["eval/mean_psnr_detail_score"] = metrics["eval/decoded_mean_psnr"] + detail_score_weight * metrics[
        "eval/highpass_energy_ratio"
    ]
    return metrics


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["project"]["output_dir"] = str(args.output_dir)
    if args.disable_wandb:
        config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
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
    loss_config = config.get("loss", {})
    print_main(f"device={device} dtype={dtype_name} distributed={distributed} world_size={world_size}", rank)

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
    additional_eval_loaders: list[tuple[str, DataLoader]] = []
    eval_every = int(eval_cfg.get("every", 1000))
    eval_run_at_start = bool(eval_cfg.get("run_at_start", True))
    best_metric_name = str(eval_cfg.get("best_metric", "eval/latent_loss"))
    best_metric_mode = str(eval_cfg.get("best_mode", "min"))
    best_checkpoint_name = str(eval_cfg.get("best_checkpoint", "best_eval_latent.pt"))
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported eval.best_mode: {best_metric_mode}")
    if eval_enabled and is_main_process(rank):
        eval_dataset = make_dataset(
            config,
            split=str(eval_cfg.get("split", "val")),
            seed=seed,
            deterministic=True,
            data_overrides=eval_cfg.get("data", {}),
        )
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
            f"degradation={eval_cfg.get('data', {}).get('degradation_preset', config['data'].get('degradation_preset'))}",
            rank,
        )
        for additional_cfg in eval_cfg.get("additional", []) or []:
            name = str(additional_cfg["name"]).strip()
            if not name or "/" in name:
                raise ValueError(f"eval.additional name must be a non-empty path-safe label, got {name!r}")
            additional_dataset = make_dataset(
                config,
                split=str(additional_cfg.get("split", "train")),
                seed=seed,
                deterministic=bool(additional_cfg.get("deterministic", True)),
                data_overrides=additional_cfg.get("data", {}),
            )
            additional_limit = int(additional_cfg.get("limit", 0))
            if additional_limit > 0 and additional_limit < len(additional_dataset):
                from torch.utils.data import Subset

                additional_dataset = Subset(additional_dataset, list(range(additional_limit)))
            additional_loader = DataLoader(
                additional_dataset,
                batch_size=int(additional_cfg.get("batch_size", eval_cfg.get("batch_size", 1))),
                shuffle=False,
                num_workers=int(additional_cfg.get("num_workers", eval_cfg.get("num_workers", 0))),
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            additional_eval_loaders.append((name, additional_loader))
            print_main(
                "eval_additional="
                f"name={name} "
                f"split={additional_cfg.get('split', 'train')} "
                f"limit={additional_cfg.get('limit', 0)} "
                f"batch_size={additional_cfg.get('batch_size', eval_cfg.get('batch_size', 1))}",
                rank,
            )

    vae = load_autoencoder(config, device=device)
    perceptual_model = make_perceptual_model(loss_config, device=device)
    model = build_stage2_model(config["model"]).to(device)
    optimizer = build_optimizer(model, train_cfg, rank=rank)
    print_main(f"optimizer_groups={format_optimizer_groups(optimizer)}", rank)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device)
        print_main(f"resumed step={start_step}", rank)
    else:
        init_config = config.get("initialization", {})
        init_checkpoint = args.init_checkpoint or init_config.get("checkpoint")
        if init_checkpoint:
            partial_init = bool(args.partial_init or init_config.get("partial", False))
            init_step = load_model_weights(Path(init_checkpoint), model, device, partial=partial_init)
            print_main(f"initialized_from={init_checkpoint} source_step={init_step} partial_init={partial_init}", rank)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    if args.override_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(args.override_lr)
            group["base_lr"] = float(args.override_lr)
        print_main(f"override_lr={float(args.override_lr):.8g}", rank)

    max_steps = int(train_cfg.get("max_steps", 1000) if args.limit_steps is None else args.limit_steps)
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    sample_every = int(train_cfg.get("sample_every", 500))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    scheduler_config = train_cfg.get("scheduler", {}) or {}
    total_optimizer_updates = max(1, math.ceil(max_steps / max(1, grad_accum_steps)))
    optimizer_updates = max(0, start_step // max(1, grad_accum_steps))
    if scheduler_enabled(scheduler_config):
        lr_factor = apply_lr_schedule(optimizer, scheduler_config, optimizer_updates + 1, total_optimizer_updates)
        print_main(
            "scheduler="
            f"type={scheduler_config.get('type')} "
            f"warmup_updates={scheduler_config.get('warmup_updates', 0)} "
            f"min_lr_ratio={scheduler_config.get('min_lr_ratio', 0.0)} "
            f"total_updates={total_optimizer_updates} "
            f"initial_factor={lr_factor:.6f}",
            rank,
        )

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
    step = start_step
    best_eval = float("inf") if best_metric_mode == "min" else float("-inf")
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
            target = normalize_image(hr)
            lr_input = normalize_image(lr)
            reference = F.interpolate(lr_input, size=target.shape[-2:], mode="bicubic", align_corners=False)

            sync_gradients = step % grad_accum_steps == 0
            sync_context = (
                model.no_sync()
                if distributed and isinstance(model, DistributedDataParallel) and not sync_gradients
                else contextlib.nullcontext()
            )
            with sync_context:
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
                if scheduler_enabled(scheduler_config):
                    apply_lr_schedule(optimizer, scheduler_config, optimizer_updates + 1, total_optimizer_updates)
                optimizer.step()
                optimizer_updates += 1
                optimizer.zero_grad(set_to_none=True)

            if is_main_process(rank) and (step % log_every == 0 or step == 1):
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
                    f"artifact_excess={float(loss_components['artifact_excess'].detach().cpu()):.5f} "
                    f"artifact_active={float(loss_components['artifact_excess_active'].detach().cpu()):.4f} "
                    f"artifact_missing={float(loss_components['artifact_missing'].detach().cpu()):.5f} "
                    f"missing_active={float(loss_components['artifact_missing_active'].detach().cpu()):.4f} "
                    f"detail_decoded={float(loss_components['detail_decoded'].detach().cpu()):.5f} "
                    f"detail_highpass={float(loss_components['detail_highpass'].detach().cpu()):.5f} "
                    f"detail_mask={float(loss_components['detail_mask_mean'].detach().cpu()):.4f} "
                    f"latent_mse={float(latent_mse.detach().cpu()):.5f} "
                    f"lr={format_optimizer_lrs(optimizer)} "
                    f"steps_per_sec={interval_steps / elapsed:.2f}"
                )
                log_metrics = {
                    "train/loss": float(loss.detach().cpu()),
                    "train/latent_loss": float(loss_components["latent"].detach().cpu()),
                    "train/decoded": float(loss_components["decoded"].detach().cpu()),
                    "train/edge": float(loss_components["edge"].detach().cpu()),
                    "train/highpass": float(loss_components["highpass"].detach().cpu()),
                    "train/highpass_magnitude": float(loss_components["highpass_magnitude"].detach().cpu()),
                    "train/perceptual": float(loss_components["perceptual"].detach().cpu()),
                    "train/artifact_excess": float(loss_components["artifact_excess"].detach().cpu()),
                    "train/artifact_missing": float(loss_components["artifact_missing"].detach().cpu()),
                    "train/artifact_excess_active": float(loss_components["artifact_excess_active"].detach().cpu()),
                    "train/artifact_missing_active": float(loss_components["artifact_missing_active"].detach().cpu()),
                    "train/detail_decoded": float(loss_components["detail_decoded"].detach().cpu()),
                    "train/detail_highpass": float(loss_components["detail_highpass"].detach().cpu()),
                    "train/detail_mask_mean": float(loss_components["detail_mask_mean"].detach().cpu()),
                    "train/latent_mse": float(latent_mse.detach().cpu()),
                    "train/optimizer_updates": optimizer_updates,
                    "system/steps_per_sec": interval_steps / elapsed,
                }
                log_metrics.update(optimizer_lr_metrics(optimizer))
                wandb_log(
                    run,
                    log_metrics,
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
                        eval_loader,
                        device,
                        dtype_name,
                        loss_config,
                        perceptual_model,
                    )
                    combined_metrics = dict(metrics)
                    for name, additional_loader in additional_eval_loaders:
                        additional_metrics = evaluate(
                            unwrap_model(model),
                            vae,
                            additional_loader,
                            device,
                            dtype_name,
                            loss_config,
                            perceptual_model,
                        )
                        prefixed_metrics = {
                            key.replace("eval/", f"eval_{name}/", 1): value
                            for key, value in additional_metrics.items()
                        }
                        combined_metrics.update(prefixed_metrics)
                        print(
                            f"eval_additional step={step} name={name} "
                            f"mean_psnr={additional_metrics['eval/decoded_mean_psnr']:.2f} "
                            f"ssim={additional_metrics['eval/decoded_ssim']:.5f} "
                            f"highpass_ratio={additional_metrics['eval/highpass_energy_ratio']:.3f} "
                            f"highpass_l1={additional_metrics['eval/highpass_l1']:.5f} "
                            f"missing={additional_metrics['eval/missing_energy']:.5f} "
                            f"excess={additional_metrics['eval/excess_energy']:.5f}"
                        )
                    add_eval_selection_metric(combined_metrics, eval_cfg)
                    (eval_dir / f"step_{step:07d}_metrics.json").write_text(
                        json.dumps({"step": step, "metrics": combined_metrics}, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"eval step={step} latent_loss={metrics['eval/latent_loss']:.5f} "
                        f"decoded_psnr={metrics['eval/decoded_psnr']:.2f} "
                        f"mean_psnr={metrics['eval/decoded_mean_psnr']:.2f} "
                        f"detail_ratio={metrics['eval/laplacian_energy_ratio']:.3f} "
                        f"highpass_ratio={metrics['eval/highpass_energy_ratio']:.3f} "
                        f"highpass_l1={metrics['eval/highpass_l1']:.5f} "
                        f"missing={metrics['eval/missing_energy']:.5f} "
                        f"excess={metrics['eval/excess_energy']:.5f} "
                        f"perceptual={metrics['eval/perceptual']:.5f} "
                        f"psnr_detail_score={metrics['eval/psnr_detail_score']:.3f}"
                    )
                    wandb_log(run, combined_metrics, step=step)
                    metric_value = float(combined_metrics[best_metric_name])
                    is_better = metric_value < best_eval if best_metric_mode == "min" else metric_value > best_eval
                    if is_better:
                        best_eval = metric_value
                        save_checkpoint(checkpoints_dir / best_checkpoint_name, unwrap_model(model), optimizer, step, config)
                barrier(distributed)

            if step % sample_every == 0 or step == 1:
                barrier(distributed)
                if is_main_process(rank):
                    sample_source = fixed_sample_batch if fixed_sample_batch is not None else batch
                    with torch.no_grad():
                        sample_hr = sample_source["hr"].to(device, non_blocking=True)
                        sample_lr = sample_source["lr"].to(device, non_blocking=True)
                        sample_domain = sample_source["domain_id"].to(device, non_blocking=True)
                        sample_lr_input = normalize_image(sample_lr)
                        with autocast_context(device, dtype_name):
                            sample_pred = unwrap_model(model)(sample_lr_input, sample_domain)
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
                barrier(distributed)

            if step % save_every == 0 or step == max_steps:
                barrier(distributed)
                if is_main_process(rank):
                    if bool(train_cfg.get("save_step_checkpoints", True)):
                        save_checkpoint(
                            checkpoints_dir / f"step_{step:07d}.pt",
                            unwrap_model(model),
                            optimizer,
                            step,
                            config,
                        )
                    if bool(train_cfg.get("save_latest", True)):
                        save_checkpoint(checkpoints_dir / "latest.pt", unwrap_model(model), optimizer, step, config)
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
