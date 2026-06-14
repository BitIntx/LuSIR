from __future__ import annotations

import torch
import torch.nn.functional as F

from sr_diffusion.detail_adversarial import (
    MaskedHighpassPatchDiscriminator,
    discriminator_logistic_loss,
    generator_logistic_loss,
    masked_patch_mean,
)
from sr_diffusion.losses import masked_feature_l1


def test_masked_feature_l1_focuses_on_selected_region() -> None:
    prediction = torch.zeros(1, 2, 4, 4)
    target = torch.zeros_like(prediction)
    prediction[:, :, :2] = 2.0
    top_mask = torch.zeros(1, 1, 4, 4)
    top_mask[:, :, :2] = 1.0
    bottom_mask = 1.0 - top_mask

    assert torch.allclose(masked_feature_l1(prediction, target, top_mask), torch.tensor(2.0))
    assert torch.allclose(masked_feature_l1(prediction, target, bottom_mask), torch.tensor(0.0))


def test_masked_patch_mean_respects_mask_and_floor() -> None:
    logits = torch.zeros(1, 1, 4, 4)
    logits[:, :, :2] = 2.0
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2] = 1.0

    assert torch.allclose(masked_patch_mean(logits, mask), torch.tensor(2.0))
    assert 1.0 < float(masked_patch_mean(logits, mask, mask_floor=0.5)) < 2.0


def test_masked_highpass_patch_discriminator_losses_backpropagate() -> None:
    torch.manual_seed(0)
    discriminator = MaskedHighpassPatchDiscriminator(
        base_channels=8,
        channel_multipliers=(1, 2),
        highpass_kernel=5,
        use_spectral_norm=False,
    )
    base = torch.rand(2, 3, 32, 32)
    real = torch.rand(2, 3, 32, 32)
    fake = torch.rand(2, 3, 32, 32, requires_grad=True)
    mask = torch.rand(2, 1, 32, 32)

    discriminator_loss = discriminator_logistic_loss(discriminator, base, real, fake, mask, mask_floor=0.05)
    discriminator_loss.backward()
    assert any(parameter.grad is not None for parameter in discriminator.parameters())

    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    generator_loss = generator_logistic_loss(discriminator, base, fake, mask, mask_floor=0.05)
    generator_loss.backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_masked_patch_discriminator_uses_high_frequency_candidate() -> None:
    discriminator = MaskedHighpassPatchDiscriminator(
        base_channels=4,
        channel_multipliers=(1,),
        highpass_kernel=5,
        use_spectral_norm=False,
    )
    base = torch.rand(1, 3, 16, 16)
    offset = torch.full_like(base, 0.2)

    first = discriminator(base, base)
    second = discriminator(base, base + offset)

    assert F.mse_loss(first, second) < 1e-10
