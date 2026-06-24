from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, math.gcd(channels, groups)), num_channels=channels)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, zero_init_output: bool = False):
        super().__init__()
        self.norm1 = _norm(channels, groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = _norm(channels, groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if zero_init_output:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

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


class GatedDepthwiseFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden_channels = max(channels, int(round(channels * expansion)))
        self.input = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.output = nn.Conv2d(hidden_channels, channels, kernel_size=1)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(self.input(x))
        gate, value = torch.chunk(x, chunks=2, dim=1)
        return self.output(F.silu(gate) * value)


class WindowAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, window_size: int = 8, shift_size: int = 0) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError(f"shift_size must be in [0, {window_size - 1}], got {shift_size}")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = int(channels) // int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _partition_windows(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        window_size = self.window_size
        heads = self.num_heads
        head_dim = self.head_dim
        x = x.view(batch, heads, head_dim, height // window_size, window_size, width // window_size, window_size)
        return x.permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, heads, window_size * window_size, head_dim)

    def _reverse_windows(self, x: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
        window_size = self.window_size
        x = x.view(batch, height // window_size, width // window_size, self.num_heads, window_size, window_size, self.head_dim)
        x = x.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        return x.view(batch, self.channels, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        window_size = self.window_size
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = x.shape[-2:]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))

        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=1)
        q = self._partition_windows(q)
        k = self._partition_windows(k)
        v = self._partition_windows(v)
        attention = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attention = attention + self.relative_position_bias.unsqueeze(0).to(dtype=attention.dtype)
        attention = attention.softmax(dim=-1)
        x = torch.matmul(attention, v)
        x = self._reverse_windows(x, batch, padded_h, padded_w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))
        x = x[..., :height, :width]
        return self.output(x)


class WindowAttentionResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int = 32,
        num_heads: int = 8,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(channels, groups)
        self.attention = WindowAttention2d(
            channels=channels,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
        )
        self.norm2 = _norm(channels, groups)
        self.ffn = GatedDepthwiseFeedForward(channels, expansion=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(F.silu(self.norm1(x)))
        x = x + self.ffn(F.silu(self.norm2(x)))
        return x


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
        extra_context_channels: tuple[int, int] = (384, 512),
        extra_context_blocks: tuple[int, int] = (8, 12),
        extra_context_scale: float = 1.0,
        attention_blocks: int = 0,
        attention_heads: int = 8,
        attention_window_size: int = 8,
        attention_mlp_ratio: float = 2.0,
        identity_init_blocks_from: int | None = None,
    ) -> None:
        super().__init__()
        supported_architectures = {
            "flat",
            "multiscale_context",
            "dual_multiscale_context",
            "dual_multiscale_attention",
        }
        if architecture not in supported_architectures:
            raise ValueError(f"Unsupported LRToLatentPredictor architecture: {architecture}")
        if identity_init_blocks_from is not None and not 0 <= int(identity_init_blocks_from) <= int(num_blocks):
            raise ValueError(
                "identity_init_blocks_from must be between 0 and num_blocks, "
                f"got {identity_init_blocks_from} for num_blocks={num_blocks}"
            )
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.domain_embedding = nn.Embedding(num_domains, base_channels)
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    base_channels,
                    groups=norm_groups,
                    zero_init_output=(
                        identity_init_blocks_from is not None and index >= int(identity_init_blocks_from)
                    ),
                )
                for index in range(num_blocks)
            ]
        )
        uses_attention = architecture == "dual_multiscale_attention" or int(attention_blocks) > 0
        self.attention = (
            nn.Sequential(
                *[
                    WindowAttentionResidualBlock(
                        base_channels,
                        groups=norm_groups,
                        num_heads=attention_heads,
                        window_size=attention_window_size,
                        shift_size=(attention_window_size // 2 if index % 2 else 0),
                        mlp_ratio=attention_mlp_ratio,
                    )
                    for index in range(int(attention_blocks))
                ]
            )
            if uses_attention and int(attention_blocks) > 0
            else None
        )
        self.context = (
            MultiScaleContext(
                base_channels=base_channels,
                context_channels=context_channels,
                context_blocks=context_blocks,
                norm_groups=norm_groups,
            )
            if architecture in {"multiscale_context", "dual_multiscale_context", "dual_multiscale_attention"}
            else None
        )
        self.context_scale = float(context_scale)
        self.extra_context = (
            MultiScaleContext(
                base_channels=base_channels,
                context_channels=extra_context_channels,
                context_blocks=extra_context_blocks,
                norm_groups=norm_groups,
            )
            if architecture in {"dual_multiscale_context", "dual_multiscale_attention"}
            else None
        )
        self.extra_context_scale = float(extra_context_scale)
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
            extra_context_channels=tuple(config.get("extra_context_channels", (384, 512))),
            extra_context_blocks=tuple(config.get("extra_context_blocks", (8, 12))),
            extra_context_scale=config.get("extra_context_scale", 1.0),
            attention_blocks=config.get("attention_blocks", 0),
            attention_heads=config.get("attention_heads", 8),
            attention_window_size=config.get("attention_window_size", 8),
            attention_mlp_ratio=config.get("attention_mlp_ratio", 2.0),
            identity_init_blocks_from=config.get("identity_init_blocks_from"),
        )

    def forward(self, lr: torch.Tensor, domain_id: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input(lr)
        if domain_id is not None:
            domain_bias = self.domain_embedding(domain_id).unsqueeze(-1).unsqueeze(-1)
            x = x + domain_bias
        x = self.blocks(x)
        if self.context is not None:
            x = x + self.context_scale * self.context(x)
        if self.extra_context is not None:
            x = x + self.extra_context_scale * self.extra_context(x)
        if self.attention is not None:
            x = self.attention(x)
        return self.output(x)


class LatentResidualRefiner(nn.Module):
    """Frozen Stage 2 base plus a zero-initialized latent residual adapter."""

    def __init__(
        self,
        base_config: dict[str, Any],
        in_channels: int = 3,
        latent_channels: int = 16,
        adapter_channels: int = 128,
        adapter_blocks: int = 8,
        norm_groups: int = 32,
        num_domains: int = 2,
        residual_scale: float = 0.35,
        base_trainable: bool = False,
        use_lr_input: bool = True,
    ) -> None:
        super().__init__()
        self.base = LRToLatentPredictor.from_config(base_config)
        self.latent_channels = int(latent_channels)
        self.residual_scale = float(residual_scale)
        self.use_lr_input = bool(use_lr_input)
        self.base_trainable = bool(base_trainable)
        for parameter in self.base.parameters():
            parameter.requires_grad_(self.base_trainable)

        adapter_in_channels = self.latent_channels + (int(in_channels) if self.use_lr_input else 0)
        self.input = nn.Conv2d(adapter_in_channels, adapter_channels, kernel_size=3, padding=1)
        self.domain_embedding = nn.Embedding(num_domains, adapter_channels)
        self.blocks = nn.Sequential(
            *[ResidualBlock(adapter_channels, groups=norm_groups) for _ in range(int(adapter_blocks))]
        )
        self.output = nn.Sequential(
            _norm(adapter_channels, norm_groups),
            nn.SiLU(),
            nn.Conv2d(adapter_channels, self.latent_channels, kernel_size=3, padding=1),
        )
        output_conv = self.output[-1]
        if isinstance(output_conv, nn.Conv2d):
            nn.init.zeros_(output_conv.weight)
            nn.init.zeros_(output_conv.bias)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LatentResidualRefiner":
        base_config = dict(config.get("base_model", {}))
        if not base_config:
            raise ValueError("LatentResidualRefiner config requires model.base_model")
        return cls(
            base_config=base_config,
            in_channels=config.get("in_channels", base_config.get("in_channels", 3)),
            latent_channels=config.get("latent_channels", base_config.get("latent_channels", 16)),
            adapter_channels=config.get("adapter_channels", 128),
            adapter_blocks=config.get("adapter_blocks", 8),
            norm_groups=config.get("norm_groups", base_config.get("norm_groups", 32)),
            num_domains=config.get("num_domains", base_config.get("num_domains", 2)),
            residual_scale=config.get("residual_scale", 0.35),
            base_trainable=config.get("base_trainable", False),
            use_lr_input=config.get("use_lr_input", True),
        )

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state_dict)

    def forward(self, lr: torch.Tensor, domain_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.base_trainable:
            base_latent = self.base(lr, domain_id)
        else:
            with torch.no_grad():
                base_latent = self.base(lr, domain_id)
        features = [base_latent.detach() if not self.base_trainable else base_latent]
        if self.use_lr_input:
            if lr.shape[-2:] != base_latent.shape[-2:]:
                lr_features = F.interpolate(lr, size=base_latent.shape[-2:], mode="bilinear", align_corners=False)
            else:
                lr_features = lr
            features.append(lr_features)
        x = self.input(torch.cat(features, dim=1))
        if domain_id is not None:
            domain_bias = self.domain_embedding(domain_id).unsqueeze(-1).unsqueeze(-1)
            x = x + domain_bias
        x = self.blocks(x)
        residual = self.residual_scale * torch.tanh(self.output(x))
        return base_latent + residual
