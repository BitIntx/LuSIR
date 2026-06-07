from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, math.gcd(channels, groups)), num_channels=channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = _norm(channels, groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _norm(channels, groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class MultiScaleContext(nn.Module):
    """Add broad spatial context while preserving the pretrained full-resolution trunk."""

    def __init__(
        self,
        base_channels: int,
        context_channels: tuple[int, int],
        context_blocks: tuple[int, int],
        norm_groups: int,
    ) -> None:
        super().__init__()
        level1_channels, level2_channels = context_channels
        level1_blocks, level2_blocks = context_blocks
        self.down1 = nn.Conv2d(base_channels, level1_channels, kernel_size=4, stride=2, padding=1)
        self.level1 = nn.Sequential(
            *[ResidualBlock(level1_channels, groups=norm_groups) for _ in range(level1_blocks)]
        )
        self.down2 = nn.Conv2d(level1_channels, level2_channels, kernel_size=4, stride=2, padding=1)
        self.level2 = nn.Sequential(
            *[ResidualBlock(level2_channels, groups=norm_groups) for _ in range(level2_blocks)]
        )
        self.deep_to_level1 = nn.Conv2d(level2_channels, level1_channels, kernel_size=3, padding=1)
        self.level1_refine = ResidualBlock(level1_channels, groups=norm_groups)
        self.to_base = nn.Conv2d(level1_channels, base_channels, kernel_size=3, padding=1)

        # A partial init from the old Stage 2 starts with exactly the old output.
        nn.init.zeros_(self.to_base.weight)
        nn.init.zeros_(self.to_base.bias)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        level1 = self.level1(self.down1(base))
        level2 = self.level2(self.down2(level1))
        level2 = F.interpolate(level2, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        level1 = self.level1_refine(level1 + self.deep_to_level1(level2))
        context = F.interpolate(level1, size=base.shape[-2:], mode="bilinear", align_corners=False)
        return self.to_base(context)


class LRToLatentPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        num_blocks: int = 8,
        norm_groups: int = 32,
        num_domains: int = 2,
        architecture: str = "flat",
        context_channels: tuple[int, int] = (256, 384),
        context_blocks: tuple[int, int] = (4, 6),
        context_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if architecture not in {"flat", "multiscale_context"}:
            raise ValueError(f"Unsupported LRToLatentPredictor architecture: {architecture}")
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.domain_embedding = nn.Embedding(num_domains, base_channels)
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels, groups=norm_groups) for _ in range(num_blocks)])
        self.context = (
            MultiScaleContext(
                base_channels=base_channels,
                context_channels=context_channels,
                context_blocks=context_blocks,
                norm_groups=norm_groups,
            )
            if architecture == "multiscale_context"
            else None
        )
        self.context_scale = float(context_scale)
        self.output = nn.Sequential(
            _norm(base_channels, norm_groups),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LRToLatentPredictor":
        return cls(
            in_channels=config.get("in_channels", 3),
            latent_channels=config.get("latent_channels", 16),
            base_channels=config.get("base_channels", 128),
            num_blocks=config.get("num_blocks", 8),
            norm_groups=config.get("norm_groups", 32),
            num_domains=config.get("num_domains", 2),
            architecture=config.get("architecture", "flat"),
            context_channels=tuple(config.get("context_channels", (256, 384))),
            context_blocks=tuple(config.get("context_blocks", (4, 6))),
            context_scale=config.get("context_scale", 1.0),
        )

    def forward(self, lr: torch.Tensor, domain_id: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input(lr)
        if domain_id is not None:
            domain_bias = self.domain_embedding(domain_id).unsqueeze(-1).unsqueeze(-1)
            x = x + domain_bias
        x = self.blocks(x)
        if self.context is not None:
            x = x + self.context_scale * self.context(x)
        return self.output(x)
