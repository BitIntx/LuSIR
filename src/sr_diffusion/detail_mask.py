from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _group_norm(channels: int, groups: int) -> nn.GroupNorm:
    channels = int(channels)
    return nn.GroupNorm(num_groups=max(1, math.gcd(int(groups), channels)), num_channels=channels)


class DetailMaskResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_groups: int = 16) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels, norm_groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _group_norm(channels, norm_groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class DetailMaskPredictor(nn.Module):
    """Predicts an inference-time detail-need score from frozen fidelity features."""

    def __init__(
        self,
        image_channels: int = 3,
        latent_channels: int = 16,
        proxy_channels: int = 4,
        hidden_channels: int = 64,
        num_blocks: int = 6,
        norm_groups: int = 16,
        num_domains: int = 2,
        highpass_kernel: int = 15,
        patch_kernel: int = 9,
        score_quantile: float = 0.95,
    ) -> None:
        super().__init__()
        self.highpass_kernel = int(highpass_kernel)
        self.patch_kernel = int(patch_kernel)
        self.score_quantile = float(score_quantile)
        input_channels = image_channels * 2 + latent_channels + proxy_channels
        self.input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.domain_embedding = nn.Embedding(num_domains, hidden_channels)
        self.blocks = nn.Sequential(
            *[DetailMaskResidualBlock(hidden_channels, norm_groups=norm_groups) for _ in range(int(num_blocks))]
        )
        self.output_norm = _group_norm(hidden_channels, norm_groups)
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "DetailMaskPredictor":
        return cls(
            image_channels=int(config.get("image_channels", 3)),
            latent_channels=int(config.get("latent_channels", 16)),
            proxy_channels=int(config.get("proxy_channels", 4)),
            hidden_channels=int(config.get("hidden_channels", 64)),
            num_blocks=int(config.get("num_blocks", 6)),
            norm_groups=int(config.get("norm_groups", 16)),
            num_domains=int(config.get("num_domains", 2)),
            highpass_kernel=int(config.get("highpass_kernel", 15)),
            patch_kernel=int(config.get("patch_kernel", 9)),
            score_quantile=float(config.get("score_quantile", 0.95)),
        )

    def forward(
        self,
        base: torch.Tensor,
        bicubic: torch.Tensor,
        condition_latent: torch.Tensor,
        domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proxies = observable_detail_proxies(
            base,
            bicubic,
            highpass_kernel=self.highpass_kernel,
            patch_kernel=self.patch_kernel,
            score_quantile=self.score_quantile,
        )
        latent_size = condition_latent.shape[-2:]
        image_features = [
            F.interpolate(base.float(), size=latent_size, mode="area"),
            F.interpolate(bicubic.float(), size=latent_size, mode="area"),
            condition_latent.float(),
            *[F.interpolate(proxy, size=latent_size, mode="area") for proxy in proxies.values()],
        ]
        x = self.input(torch.cat(image_features, dim=1).to(dtype=self.input.weight.dtype))
        if domain_id is not None:
            x = x + self.domain_embedding(domain_id).unsqueeze(-1).unsqueeze(-1)
        x = self.blocks(x)
        logits = self.output(F.silu(self.output_norm(x)))
        logits = F.interpolate(logits.float(), size=base.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


def _validate_kernel(name: str, kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {kernel_size}")
    return kernel_size


def lowpass(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _validate_kernel("kernel_size", kernel_size)
    if kernel_size == 1:
        return image
    padding = kernel_size // 2
    if image.shape[-2] <= padding or image.shape[-1] <= padding:
        raise ValueError(f"image is too small for reflect padding with kernel {kernel_size}: {tuple(image.shape)}")
    padded = F.pad(image.float(), (padding, padding, padding, padding), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def highpass(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    return image - lowpass(image, kernel_size)


def normalize_score(score: torch.Tensor, quantile: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    if score.ndim != 4 or score.shape[1] != 1:
        raise ValueError(f"score must have shape [B, 1, H, W], got {tuple(score.shape)}")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], got {quantile}")
    flat = score.float().flatten(1)
    scale = torch.quantile(flat, quantile, dim=1, keepdim=True).clamp_min(eps)
    return (flat / scale).clamp(0.0, 1.0).view_as(score)


def top_fraction_mask(score: torch.Tensor, fraction: float) -> torch.Tensor:
    if score.ndim != 4 or score.shape[1] != 1:
        raise ValueError(f"score must have shape [B, 1, H, W], got {tuple(score.shape)}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    flat = score.float().flatten(1)
    count = max(1, int(round(flat.shape[1] * float(fraction))))
    indices = torch.topk(flat, k=count, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)
    return mask.view_as(score)


def detail_need_components(
    base: torch.Tensor,
    target: torch.Tensor,
    *,
    highpass_kernel: int = 15,
    patch_kernel: int = 9,
    score_quantile: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Build a GT-supervised target for locations where the base lacks detail.

    The score intentionally excludes locations where the base merely has a
    differently signed or excessive high-frequency response. Those locations
    need correction, not additional generated detail.
    """

    if base.shape != target.shape or base.ndim != 4:
        raise ValueError(f"base and target must have matching BCHW shapes, got {base.shape} and {target.shape}")
    highpass_kernel = _validate_kernel("highpass_kernel", highpass_kernel)
    patch_kernel = _validate_kernel("patch_kernel", patch_kernel)
    base_high = highpass(base.float(), highpass_kernel)
    target_high = highpass(target.float(), highpass_kernel)
    base_energy = base_high.abs().mean(dim=1, keepdim=True)
    target_energy = target_high.abs().mean(dim=1, keepdim=True)
    missing = (target_energy - base_energy).clamp_min(0.0)
    excess = (base_energy - target_energy).clamp_min(0.0)
    mismatch = (target_high - base_high).abs().mean(dim=1, keepdim=True)
    patch_missing = lowpass(missing, patch_kernel)
    patch_mismatch = lowpass(mismatch, patch_kernel)
    raw_score = torch.sqrt((patch_missing * patch_mismatch).clamp_min(0.0))
    return {
        "score": normalize_score(raw_score, quantile=score_quantile),
        "raw_score": raw_score,
        "missing": missing,
        "excess": excess,
        "mismatch": mismatch,
        "base_energy": base_energy,
        "target_energy": target_energy,
        "base_high": base_high,
        "target_high": target_high,
    }


def observable_detail_proxies(
    base: torch.Tensor,
    bicubic: torch.Tensor,
    *,
    highpass_kernel: int = 15,
    patch_kernel: int = 9,
    score_quantile: float = 0.95,
) -> dict[str, torch.Tensor]:
    """Return inference-time maps that a learned mask predictor can observe."""

    if base.shape != bicubic.shape or base.ndim != 4:
        raise ValueError(f"base and bicubic must have matching BCHW shapes, got {base.shape} and {bicubic.shape}")
    highpass_kernel = _validate_kernel("highpass_kernel", highpass_kernel)
    patch_kernel = _validate_kernel("patch_kernel", patch_kernel)
    base_high = highpass(base.float(), highpass_kernel)
    bicubic_high = highpass(bicubic.float(), highpass_kernel)
    raw = {
        "base_detail": lowpass(base_high.abs().mean(dim=1, keepdim=True), patch_kernel),
        "bicubic_detail": lowpass(bicubic_high.abs().mean(dim=1, keepdim=True), patch_kernel),
        "base_bicubic_gap": lowpass((base.float() - bicubic.float()).abs().mean(dim=1, keepdim=True), patch_kernel),
        "highpass_disagreement": lowpass((base_high - bicubic_high).abs().mean(dim=1, keepdim=True), patch_kernel),
    }
    return {name: normalize_score(value, quantile=score_quantile) for name, value in raw.items()}


def masked_highpass_oracle(
    base: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    highpass_kernel: int = 15,
) -> torch.Tensor:
    if base.shape != target.shape or mask.shape != base[:, :1].shape:
        raise ValueError(f"expected base/target BCHW and mask B1HW, got {base.shape}, {target.shape}, {mask.shape}")
    correction = highpass(target.float() - base.float(), highpass_kernel)
    return (base.float() + correction * mask.float()).clamp(0.0, 1.0)
