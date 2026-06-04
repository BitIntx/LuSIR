from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + eps * eps).mean()


def _depthwise_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = int(image.shape[1])
    weight = kernel.to(device=image.device, dtype=image.dtype).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return F.conv2d(image, weight, padding=1, groups=channels)


def _lowpass(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def sobel_edge_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    sobel_x = prediction.new_tensor(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ]
    ) / 8.0
    sobel_y = prediction.new_tensor(
        [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ]
    ) / 8.0
    pred_x = _depthwise_filter(prediction, sobel_x)
    pred_y = _depthwise_filter(prediction, sobel_y)
    target_x = _depthwise_filter(target, sobel_x)
    target_y = _depthwise_filter(target, sobel_y)
    return 0.5 * (charbonnier_loss(pred_x, target_x, eps=eps) + charbonnier_loss(pred_y, target_y, eps=eps))


def sobel_residual_magnitude_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    sobel_x = prediction.new_tensor(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ]
    ) / 8.0
    sobel_y = prediction.new_tensor(
        [
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ]
    ) / 8.0
    pred_residual = prediction - reference
    target_residual = target - reference
    pred_x = _depthwise_filter(pred_residual, sobel_x)
    pred_y = _depthwise_filter(pred_residual, sobel_y)
    target_x = _depthwise_filter(target_residual, sobel_x)
    target_y = _depthwise_filter(target_residual, sobel_y)
    pred_mag = torch.sqrt(pred_x.pow(2) + pred_y.pow(2) + eps * eps)
    target_mag = torch.sqrt(target_x.pow(2) + target_y.pow(2) + eps * eps)
    return charbonnier_loss(pred_mag, target_mag, eps=eps)


def laplacian_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    kernel = prediction.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ) / 4.0
    pred_high = _depthwise_filter(prediction, kernel)
    target_high = _depthwise_filter(target, kernel)
    return charbonnier_loss(pred_high, target_high, eps=eps)


def laplacian_residual_magnitude_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    kernel = prediction.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ) / 4.0
    pred_high = _depthwise_filter(prediction - reference, kernel)
    target_high = _depthwise_filter(target - reference, kernel)
    return charbonnier_loss(pred_high.abs(), target_high.abs(), eps=eps)


def lowpass_anchor_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    kernel_size: int = 9,
    eps: float = 1e-3,
) -> torch.Tensor:
    return charbonnier_loss(_lowpass(prediction, kernel_size), _lowpass(reference, kernel_size), eps=eps)


def laplacian_detail_gate_anchor_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    threshold: float = 0.035,
    eps: float = 1e-3,
) -> torch.Tensor:
    kernel = prediction.new_tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    ) / 4.0
    pred_detail = _depthwise_filter(prediction - reference, kernel)
    target_detail = _depthwise_filter(target - reference, kernel).detach()
    target_magnitude = target_detail.abs().mean(dim=1, keepdim=True)
    weight = torch.exp(-target_magnitude / max(float(threshold), 1e-6)).detach()
    per_pixel = torch.sqrt(pred_detail.pow(2) + eps * eps).mean(dim=1, keepdim=True)
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1e-8)


def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).mean()


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_kind = config.get("reconstruction", "charbonnier")
    if recon_kind == "l1":
        recon = torch.nn.functional.l1_loss(reconstruction, target)
    elif recon_kind == "mse":
        recon = torch.nn.functional.mse_loss(reconstruction, target)
    elif recon_kind == "charbonnier":
        recon = charbonnier_loss(reconstruction, target)
    else:
        raise ValueError(f"Unsupported reconstruction loss: {recon_kind}")

    kl = kl_loss(mean, logvar)
    kl_weight = float(config.get("kl_weight", 1e-6))
    total = recon + kl_weight * kl
    metrics = {
        "loss": float(total.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "kl": float(kl.detach().cpu()),
    }
    return total, metrics
