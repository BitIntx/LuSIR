from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _validate_odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"highpass_kernel must be a positive odd integer, got {kernel_size}")
    return kernel_size


def highpass(image: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    kernel_size = _validate_odd_kernel(kernel_size)
    if kernel_size == 1:
        return image
    padding = kernel_size // 2
    padded = F.pad(image.float(), (padding, padding, padding, padding), mode="reflect")
    return image.float() - F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def masked_patch_mean(logits: torch.Tensor, mask: torch.Tensor | None, mask_floor: float = 0.0) -> torch.Tensor:
    if mask is None:
        return logits.mean()
    if mask.ndim != 4 or mask.shape[0] != logits.shape[0] or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B, 1, H, W], got {tuple(mask.shape)}")
    floor = min(max(float(mask_floor), 0.0), 1.0)
    weights = F.interpolate(mask.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False)
    weights = floor + (1.0 - floor) * weights.clamp(0.0, 1.0)
    numerator = (logits.float() * weights).flatten(1).sum(dim=1)
    denominator = weights.flatten(1).sum(dim=1).clamp_min(1e-8)
    return (numerator / denominator).mean()


class MaskedHighpassPatchDiscriminator(nn.Module):
    """Small conditional PatchGAN over base SR and candidate high-frequency content."""

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        channel_multipliers: Sequence[int] = (1, 2, 4, 4),
        highpass_kernel: int = 15,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.highpass_kernel = _validate_odd_kernel(highpass_kernel)
        layers: list[nn.Module] = []
        in_channels = int(image_channels) * 2
        for multiplier in channel_multipliers:
            out_channels = int(base_channels) * int(multiplier)
            convolution = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            layers.extend(
                [
                    spectral_norm(convolution) if use_spectral_norm else convolution,
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels
        output = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        layers.append(spectral_norm(output) if use_spectral_norm else output)
        self.layers = nn.Sequential(*layers)

    def forward(self, base_sr: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if base_sr.shape != candidate.shape:
            raise ValueError(f"base_sr and candidate shapes must match, got {base_sr.shape} and {candidate.shape}")
        inputs = torch.cat(
            [
                base_sr.detach().float(),
                highpass(candidate, kernel_size=self.highpass_kernel),
            ],
            dim=1,
        )
        return self.layers(inputs)


def discriminator_logistic_loss(
    discriminator: MaskedHighpassPatchDiscriminator,
    base_sr: torch.Tensor,
    real: torch.Tensor,
    fake: torch.Tensor,
    mask: torch.Tensor | None,
    mask_floor: float = 0.0,
) -> torch.Tensor:
    real_logits = discriminator(base_sr, real)
    fake_logits = discriminator(base_sr, fake.detach())
    return 0.5 * (
        masked_patch_mean(F.softplus(-real_logits), mask, mask_floor)
        + masked_patch_mean(F.softplus(fake_logits), mask, mask_floor)
    )


def generator_logistic_loss(
    discriminator: MaskedHighpassPatchDiscriminator,
    base_sr: torch.Tensor,
    fake: torch.Tensor,
    mask: torch.Tensor | None,
    mask_floor: float = 0.0,
) -> torch.Tensor:
    return masked_patch_mean(F.softplus(-discriminator(base_sr, fake)), mask, mask_floor)
