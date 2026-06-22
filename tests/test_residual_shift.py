from __future__ import annotations

import torch

from sr_diffusion.residual_shift import (
    apply_masked_correction,
    masked_latent_target,
    residual_shift_eta,
    residual_shift_forward_sample,
    residual_shift_step,
)


def test_eta_is_monotonic_and_bounded() -> None:
    timesteps = torch.arange(10)
    eta = residual_shift_eta(timesteps, num_timesteps=10, power=2.0)
    assert torch.all(eta[1:] > eta[:-1])
    assert eta[0] > 0.0
    assert eta[-1] == 1.0


def test_masked_target_and_correction_preserve_unselected_latents() -> None:
    base = torch.randn(1, 4, 8, 8)
    target = torch.randn_like(base)
    correction = torch.randn_like(base)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4] = 1.0

    masked_target = masked_latent_target(base, target, mask)
    prediction = apply_masked_correction(base, correction, mask, correction_scale=0.5)

    assert torch.allclose(masked_target[:, :, 4:], base[:, :, 4:])
    assert torch.allclose(prediction[:, :, 4:], base[:, :, 4:])
    assert torch.allclose(prediction[:, :, :4], base[:, :, :4] + 0.5 * correction[:, :, :4])


def test_residual_shift_step_recovers_forward_trajectory() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 4, 8, 8)
    base = torch.randn_like(target)
    noise = torch.randn_like(target)
    eta = torch.tensor([1.0, 0.6])
    next_eta = torch.tensor([0.4, 0.0])
    noise_scale = 0.2
    current = residual_shift_forward_sample(target, base, eta, noise, noise_scale)

    stepped = residual_shift_step(current, target, base, eta, next_eta, noise_scale)
    expected = residual_shift_forward_sample(target, base, next_eta, noise, noise_scale)

    assert torch.allclose(stepped, expected, atol=1e-5)
