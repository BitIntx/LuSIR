from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, math.gcd(channels, groups)), num_channels=channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_channels: int, groups: int = 32) -> None:
        super().__init__()
        self.norm1 = _norm(in_channels, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.embedding = nn.Sequential(nn.SiLU(), nn.Linear(embedding_channels, out_channels))
        self.norm2 = _norm(out_channels, groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.embedding(embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, groups: int = 32) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels must be divisible by num_heads: {channels}, {num_heads}")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _norm(channels, groups)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        x = self.norm(x).reshape(batch, channels, height * width)
        qkv = self.qkv(x).reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        q, k, v = qkv.unbind(dim=1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(-1, -2).reshape(batch, channels, height * width)
        x = self.proj(x).reshape(batch, channels, height, width)
        return x + residual


class ResAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        use_attention: bool,
        groups: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, embedding_channels, groups=groups)
        self.attn = AttentionBlock(out_channels, num_heads=num_heads, groups=groups) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = self.res(x, embedding)
        return self.attn(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 16,
        condition_channels: int = 16,
        out_channels: int | None = None,
        base_channels: int = 128,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        norm_groups: int = 32,
        num_heads: int = 4,
        attention_resolutions: list[int] | tuple[int, ...] = (32, 16),
        base_resolution: int = 128,
        num_domains: int = 2,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.condition_channels = condition_channels
        self.out_channels = latent_channels if out_channels is None else out_channels
        self.base_channels = base_channels
        self.base_resolution = base_resolution
        attention_resolutions = tuple(int(value) for value in attention_resolutions)

        embedding_channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.domain_embedding = nn.Embedding(num_domains, embedding_channels)
        self.input = nn.Conv2d(latent_channels + condition_channels, base_channels, kernel_size=3, padding=1)

        channels_by_level = [base_channels * int(multiplier) for multiplier in channel_multipliers]
        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = base_channels
        current_resolution = base_resolution
        for level, level_channels in enumerate(channels_by_level):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResAttnBlock(
                        current_channels,
                        level_channels,
                        embedding_channels,
                        use_attention=current_resolution in attention_resolutions,
                        groups=norm_groups,
                        num_heads=num_heads,
                    )
                )
                current_channels = level_channels
            self.down_levels.append(blocks)
            if level < len(channels_by_level) - 1:
                self.downsamples.append(Downsample(current_channels))
                current_resolution //= 2

        self.mid = nn.ModuleList(
            [
                ResAttnBlock(
                    current_channels,
                    current_channels,
                    embedding_channels,
                    use_attention=True,
                    groups=norm_groups,
                    num_heads=num_heads,
                ),
                ResAttnBlock(
                    current_channels,
                    current_channels,
                    embedding_channels,
                    use_attention=False,
                    groups=norm_groups,
                    num_heads=num_heads,
                ),
            ]
        )

        self.up_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels_by_level))):
            level_channels = channels_by_level[level]
            blocks = nn.ModuleList()
            blocks.append(
                ResAttnBlock(
                    current_channels + level_channels,
                    level_channels,
                    embedding_channels,
                    use_attention=current_resolution in attention_resolutions,
                    groups=norm_groups,
                    num_heads=num_heads,
                )
            )
            current_channels = level_channels
            for _ in range(num_res_blocks):
                blocks.append(
                    ResAttnBlock(
                        current_channels,
                        current_channels,
                        embedding_channels,
                        use_attention=current_resolution in attention_resolutions,
                        groups=norm_groups,
                        num_heads=num_heads,
                    )
                )
            self.up_levels.append(blocks)
            if level > 0:
                previous_channels = channels_by_level[level - 1]
                self.upsamples.append(Upsample(current_channels, previous_channels))
                current_channels = previous_channels
                current_resolution *= 2

        self.output = nn.Sequential(
            _norm(current_channels, norm_groups),
            nn.SiLU(),
            nn.Conv2d(current_channels, self.out_channels, kernel_size=3, padding=1),
        )
        if zero_init_output:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConditionalUNet":
        return cls(
            latent_channels=config.get("latent_channels", 16),
            condition_channels=config.get("condition_channels", 16),
            out_channels=config.get("out_channels"),
            base_channels=config.get("base_channels", 128),
            channel_multipliers=config.get("channel_multipliers", [1, 2, 3, 4]),
            num_res_blocks=config.get("num_res_blocks", 2),
            norm_groups=config.get("norm_groups", 32),
            num_heads=config.get("num_heads", 4),
            attention_resolutions=config.get("attention_resolutions", [32, 16]),
            base_resolution=config.get("base_resolution", 128),
            num_domains=config.get("num_domains", 2),
            zero_init_output=config.get("zero_init_output", False),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_latent.shape[-2:] != condition.shape[-2:]:
            raise ValueError(f"Condition spatial shape must match latent: {condition.shape}, {noisy_latent.shape}")
        embedding = self.time_mlp(timestep_embedding(timesteps, self.base_channels).to(dtype=noisy_latent.dtype))
        if domain_id is not None:
            embedding = embedding + self.domain_embedding(domain_id)

        x = self.input(torch.cat([noisy_latent, condition], dim=1))
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.down_levels):
            for block in blocks:
                x = block(x, embedding)
            skips.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x)

        for block in self.mid:
            x = block(x, embedding)

        for level, blocks in enumerate(self.up_levels):
            x = torch.cat([x, skips.pop()], dim=1)
            for block in blocks:
                x = block(x, embedding)
            if level < len(self.upsamples):
                x = self.upsamples[level](x)

        return self.output(x)
