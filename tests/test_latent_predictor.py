from __future__ import annotations

import torch

from sr_diffusion.models import LRToLatentPredictor, LatentResidualRefiner
from sr_diffusion.utils import load_matching_weights
from tools.train.train_latent_pretrain import load_model_weights


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


def test_latent_residual_refiner_zero_init_preserves_base_output() -> None:
    torch.manual_seed(7)
    base_config = {
        "in_channels": 3,
        "latent_channels": 8,
        "base_channels": 32,
        "num_blocks": 2,
        "norm_groups": 8,
        "num_domains": 2,
    }
    refiner = LatentResidualRefiner.from_config(
        {
            "base_model": base_config,
            "in_channels": 3,
            "latent_channels": 8,
            "adapter_channels": 16,
            "adapter_blocks": 1,
            "norm_groups": 8,
            "num_domains": 2,
            "residual_scale": 0.5,
        }
    ).eval()
    lr = torch.randn(2, 3, 32, 32)
    domain_id = torch.tensor([0, 1])

    with torch.no_grad():
        expected = refiner.base(lr, domain_id)
        actual = refiner(lr, domain_id)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert all(not parameter.requires_grad for parameter in refiner.base.parameters())


def test_latent_residual_refiner_loads_plain_stage2_checkpoint_into_base(tmp_path) -> None:
    torch.manual_seed(8)
    base_config = {
        "in_channels": 3,
        "latent_channels": 8,
        "base_channels": 32,
        "num_blocks": 2,
        "norm_groups": 8,
        "num_domains": 2,
    }
    base = LRToLatentPredictor.from_config(base_config).eval()
    for parameter in base.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)
    checkpoint = tmp_path / "base.pt"
    torch.save({"step": 99, "model": base.state_dict()}, checkpoint)

    refiner = LatentResidualRefiner.from_config(
        {
            "base_model": base_config,
            "in_channels": 3,
            "latent_channels": 8,
            "adapter_channels": 16,
            "adapter_blocks": 1,
            "norm_groups": 8,
            "num_domains": 2,
        }
    ).eval()
    loaded_step = load_model_weights(checkpoint, refiner, torch.device("cpu"))
    lr = torch.randn(2, 3, 32, 32)
    domain_id = torch.tensor([0, 1])

    with torch.no_grad():
        expected = base(lr, domain_id)
        actual = refiner(lr, domain_id)

    assert loaded_step == 99
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
