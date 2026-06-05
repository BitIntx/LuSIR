from __future__ import annotations

import torch

from sr_diffusion.models import ConditionalUNet, NoiseScheduler, predict_x0_and_noise


def test_noise_scheduler_shapes() -> None:
    scheduler = NoiseScheduler(num_train_timesteps=10)
    x0 = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([0, 9])
    noisy = scheduler.add_noise(x0, noise, timesteps)
    pred_x0 = scheduler.predict_x0_from_noise(noisy, timesteps, noise)
    recovered_noise = scheduler.noise_from_x0(noisy, x0, timesteps)
    assert noisy.shape == x0.shape
    assert pred_x0.shape == x0.shape
    assert torch.isfinite(pred_x0).all()
    assert torch.allclose(recovered_noise, noise, atol=2e-4)


def test_predict_x0_and_noise_noise_mode_matches_scheduler() -> None:
    scheduler = NoiseScheduler(num_train_timesteps=10)
    x0 = torch.randn(2, 4, 8, 8)
    condition = torch.randn_like(x0)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([2, 7])
    noisy = scheduler.add_noise(x0, noise, timesteps)

    pred_x0, pred_noise = predict_x0_and_noise(scheduler, noisy, timesteps, noise, condition)

    assert torch.allclose(pred_noise, noise)
    assert torch.allclose(pred_x0, scheduler.predict_x0_from_noise(noisy, timesteps, noise))


def test_predict_x0_and_noise_gated_residual_mode_is_bounded() -> None:
    scheduler = NoiseScheduler(num_train_timesteps=10)
    noisy = torch.randn(2, 4, 8, 8)
    condition = torch.randn_like(noisy)
    timesteps = torch.tensor([2, 7])
    model_output = torch.randn(2, 8, 8, 8)
    residual_scale = 0.5

    pred_x0, pred_noise = predict_x0_and_noise(
        scheduler,
        noisy,
        timesteps,
        model_output,
        condition,
        {"prediction_type": "gated_residual_x0", "residual_scale": residual_scale},
    )

    residual = pred_x0 - condition
    recovered_noise = scheduler.noise_from_x0(noisy, pred_x0, timesteps)
    assert pred_x0.shape == condition.shape
    assert pred_noise.shape == condition.shape
    assert residual.abs().max() <= residual_scale + 1e-6
    assert torch.allclose(pred_noise, recovered_noise)


def test_conditional_unet_shape() -> None:
    model = ConditionalUNet(
        latent_channels=8,
        condition_channels=8,
        out_channels=8,
        base_channels=32,
        channel_multipliers=[1, 2],
        num_res_blocks=1,
        norm_groups=8,
        num_heads=4,
        attention_resolutions=[16],
        base_resolution=32,
        num_domains=2,
    )
    noisy = torch.randn(2, 8, 32, 32)
    condition = torch.randn(2, 8, 32, 32)
    timesteps = torch.tensor([1, 5])
    domain_id = torch.tensor([0, 1])
    prediction = model(noisy, timesteps, condition, domain_id)
    assert prediction.shape == noisy.shape
