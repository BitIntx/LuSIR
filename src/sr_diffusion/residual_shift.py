from __future__ import annotations

import torch


def residual_shift_eta(
    timesteps: torch.Tensor,
    num_timesteps: int,
    power: float = 1.0,
) -> torch.Tensor:
    """Map integer timesteps to a monotonic residual-shift coefficient in (0, 1]."""

    num_timesteps = int(num_timesteps)
    power = float(power)
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
    if power <= 0.0:
        raise ValueError(f"power must be positive, got {power}")
    if torch.any(timesteps < 0) or torch.any(timesteps >= num_timesteps):
        raise ValueError(f"timesteps must be in [0, {num_timesteps - 1}]")
    return ((timesteps.float() + 1.0) / float(num_timesteps)).pow(power)


def expand_batch(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.shape[0] != target.shape[0]:
        raise ValueError(f"values must have shape [B], got {tuple(values.shape)} for {tuple(target.shape)}")
    return values.reshape(values.shape[0], *((1,) * (target.ndim - 1)))


def masked_latent_target(
    base_latent: torch.Tensor,
    target_latent: torch.Tensor,
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    if base_latent.shape != target_latent.shape:
        raise ValueError(f"latent shapes must match, got {base_latent.shape} and {target_latent.shape}")
    if latent_mask.shape != (base_latent.shape[0], 1, *base_latent.shape[-2:]):
        raise ValueError(f"latent_mask has incompatible shape {latent_mask.shape} for {base_latent.shape}")
    mask = latent_mask.to(device=base_latent.device, dtype=base_latent.dtype).clamp(0.0, 1.0)
    return base_latent + mask * (target_latent - base_latent)


def apply_masked_correction(
    base_latent: torch.Tensor,
    correction: torch.Tensor,
    latent_mask: torch.Tensor,
    correction_scale: float = 1.0,
) -> torch.Tensor:
    if correction.shape != base_latent.shape:
        raise ValueError(f"correction shape must match base latent, got {correction.shape} and {base_latent.shape}")
    if latent_mask.shape != (base_latent.shape[0], 1, *base_latent.shape[-2:]):
        raise ValueError(f"latent_mask has incompatible shape {latent_mask.shape} for {base_latent.shape}")
    mask = latent_mask.to(device=base_latent.device, dtype=base_latent.dtype).clamp(0.0, 1.0)
    return base_latent + float(correction_scale) * mask * correction


def residual_shift_forward_sample(
    target_latent: torch.Tensor,
    base_latent: torch.Tensor,
    eta: torch.Tensor,
    noise: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    if target_latent.shape != base_latent.shape or noise.shape != target_latent.shape:
        raise ValueError("target_latent, base_latent, and noise must have matching shapes")
    eta_expanded = expand_batch(eta, target_latent).to(device=target_latent.device, dtype=target_latent.dtype)
    mean = (1.0 - eta_expanded) * target_latent + eta_expanded * base_latent
    return mean + float(noise_scale) * eta_expanded.sqrt() * noise


def residual_shift_step(
    current: torch.Tensor,
    predicted_target: torch.Tensor,
    base_latent: torch.Tensor,
    eta: torch.Tensor,
    next_eta: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Deterministic DDIM-style update for the residual-shifting process."""

    if current.shape != predicted_target.shape or current.shape != base_latent.shape:
        raise ValueError("current, predicted_target, and base_latent must have matching shapes")
    eta_expanded = expand_batch(eta, current).to(device=current.device, dtype=current.dtype)
    next_eta_expanded = expand_batch(next_eta, current).to(device=current.device, dtype=current.dtype)
    scale = float(noise_scale)
    current_mean = (1.0 - eta_expanded) * predicted_target + eta_expanded * base_latent
    if scale > 0.0:
        estimated_noise = (current - current_mean) / (scale * eta_expanded.sqrt()).clamp_min(1e-8)
    else:
        estimated_noise = torch.zeros_like(current)
    next_mean = (1.0 - next_eta_expanded) * predicted_target + next_eta_expanded * base_latent
    return next_mean + scale * next_eta_expanded.sqrt() * estimated_noise
