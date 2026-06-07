from __future__ import annotations

import torch

from sr_diffusion.models import LRToLatentPredictor
from sr_diffusion.utils import load_matching_weights


def test_lr_to_latent_predictor_shape() -> None:
    model = LRToLatentPredictor(
        in_channels=3,
        latent_channels=8,
        base_channels=32,
        num_blocks=2,
        norm_groups=8,
        num_domains=2,
    )
    lr = torch.randn(2, 3, 32, 32)
    domain_id = torch.tensor([0, 1])
    latent = model(lr, domain_id)
    assert latent.shape == (2, 8, 32, 32)


def test_multiscale_context_predictor_shape() -> None:
    model = LRToLatentPredictor(
        in_channels=3,
        latent_channels=8,
        base_channels=32,
        num_blocks=2,
        norm_groups=8,
        num_domains=2,
        architecture="multiscale_context",
        context_channels=(48, 64),
        context_blocks=(1, 2),
    )
    latent = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    assert latent.shape == (2, 8, 32, 32)


def test_multiscale_context_partial_init_preserves_flat_output() -> None:
    torch.manual_seed(123)
    flat = LRToLatentPredictor(
        latent_channels=8,
        base_channels=32,
        num_blocks=2,
        norm_groups=8,
    )
    multiscale = LRToLatentPredictor(
        latent_channels=8,
        base_channels=32,
        num_blocks=2,
        norm_groups=8,
        architecture="multiscale_context",
        context_channels=(48, 64),
        context_blocks=(1, 2),
    )
    stats = load_matching_weights(multiscale, flat.state_dict())
    lr = torch.randn(2, 3, 32, 32)
    domain_id = torch.tensor([0, 1])

    torch.testing.assert_close(multiscale(lr, domain_id), flat(lr, domain_id))
    assert stats["matched_tensors"] == len(flat.state_dict())
