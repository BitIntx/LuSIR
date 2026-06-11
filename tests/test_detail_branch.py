from __future__ import annotations

import torch

from tools.train.train_detail_branch import GatedHighFrequencyDetailBranch, training_loss


def test_detail_branch_zero_init_preserves_base_image() -> None:
    torch.manual_seed(0)
    model = GatedHighFrequencyDetailBranch(
        hidden_channels=16,
        num_blocks=2,
        norm_groups=8,
        residual_scale=0.18,
        gate_bias=-2.0,
        highpass_kernel=5,
    )
    base = torch.rand(2, 3, 32, 32)
    bicubic = torch.rand(2, 3, 32, 32)
    domain_id = torch.tensor([0, 1], dtype=torch.long)

    refined, residual, gate, raw_residual = model(base, bicubic, domain_id=domain_id)

    assert torch.allclose(refined, base, atol=0.0, rtol=0.0)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=0.0, rtol=0.0)
    assert torch.allclose(raw_residual, torch.zeros_like(raw_residual), atol=0.0, rtol=0.0)
    assert torch.allclose(gate.mean(), torch.sigmoid(torch.tensor(-2.0)), atol=1e-6)


def test_detail_branch_accepts_condition_latent() -> None:
    model = GatedHighFrequencyDetailBranch(
        latent_channels=4,
        hidden_channels=16,
        num_blocks=1,
        norm_groups=8,
        highpass_kernel=5,
        use_condition_latent=True,
    )
    base = torch.rand(1, 3, 24, 24)
    bicubic = torch.rand(1, 3, 24, 24)
    condition = torch.rand(1, 4, 6, 6)
    refined, residual, gate, _ = model(base, bicubic, condition_latent=condition)

    assert refined.shape == base.shape
    assert residual.shape == base.shape
    assert gate.shape == (1, 1, 24, 24)


def test_detail_branch_training_loss_backpropagates() -> None:
    torch.manual_seed(1)
    model = GatedHighFrequencyDetailBranch(
        hidden_channels=16,
        num_blocks=2,
        norm_groups=8,
        residual_scale=0.18,
        gate_bias=-2.0,
        highpass_kernel=5,
    )
    base = torch.rand(2, 3, 32, 32) * 0.5 + 0.25
    bicubic = torch.rand(2, 3, 32, 32)
    condition = torch.rand(2, 16, 8, 8)
    hr = (base + 0.05 * torch.randn_like(base)).clamp(0.0, 1.0)
    domain_id = torch.tensor([0, 1], dtype=torch.long)

    loss, parts = training_loss(
        model=model,
        base_sr=base,
        bicubic=bicubic,
        condition=condition,
        hr=hr,
        domain_id=domain_id,
        loss_cfg={"highpass_kernel": 5, "lowpass_kernel": 7},
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(parts["sr"]).all()
    assert model.output.weight.grad is not None
    assert model.output.weight.grad.abs().sum() > 0
