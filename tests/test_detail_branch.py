from __future__ import annotations

import torch

from tools.train.train_detail_branch import GatedHighFrequencyDetailBranch, init_model_from_checkpoint, training_loss


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


def test_detail_branch_external_mask_scales_gate_and_residual() -> None:
    torch.manual_seed(4)
    model = GatedHighFrequencyDetailBranch(
        hidden_channels=16,
        num_blocks=1,
        norm_groups=8,
        highpass_kernel=5,
    )
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)
    base = torch.rand(1, 3, 24, 24)
    bicubic = torch.rand(1, 3, 24, 24)
    full_outputs = model(base, bicubic)
    masked_outputs = model(base, bicubic, detail_mask=torch.zeros(1, 1, 24, 24), detail_mask_floor=0.25)

    assert torch.allclose(masked_outputs[1], full_outputs[1] * 0.25, atol=1e-6)
    assert torch.allclose(masked_outputs[2], full_outputs[2] * 0.25, atol=1e-6)
    assert torch.allclose(masked_outputs[0], (base + full_outputs[1] * 0.25).clamp(0.0, 1.0), atol=1e-6)


def test_detail_branch_model_init_preserves_old_path_with_condition_latent(tmp_path) -> None:
    torch.manual_seed(2)
    old_model = GatedHighFrequencyDetailBranch(
        hidden_channels=16,
        num_blocks=1,
        norm_groups=8,
        residual_scale=0.18,
        gate_bias=-2.0,
        highpass_kernel=5,
        use_condition_latent=False,
    )
    for parameter in old_model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)
    checkpoint = tmp_path / "old.pt"
    torch.save({"step": 123, "model": old_model.state_dict()}, checkpoint)

    new_model = GatedHighFrequencyDetailBranch(
        latent_channels=4,
        hidden_channels=16,
        num_blocks=1,
        norm_groups=8,
        residual_scale=0.18,
        gate_bias=-2.0,
        highpass_kernel=5,
        use_condition_latent=True,
    )
    stats = init_model_from_checkpoint(checkpoint, new_model, torch.device("cpu"))

    base = torch.rand(1, 3, 24, 24)
    bicubic = torch.rand(1, 3, 24, 24)
    condition = torch.rand(1, 4, 6, 6)
    domain_id = torch.tensor([1], dtype=torch.long)
    old_refined, old_residual, old_gate, old_raw = old_model(base, bicubic, domain_id=domain_id)
    new_refined, new_residual, new_gate, new_raw = new_model(base, bicubic, condition, domain_id)

    assert stats["checkpoint_step"] == 123
    assert stats["partial_tensors"] == 1
    assert torch.allclose(new_refined, old_refined, atol=1e-6)
    assert torch.allclose(new_residual, old_residual, atol=1e-6)
    assert torch.allclose(new_gate, old_gate, atol=1e-6)
    assert torch.allclose(new_raw, old_raw, atol=1e-6)


def test_detail_branch_model_init_preserves_old_path_with_new_identity_blocks(tmp_path) -> None:
    torch.manual_seed(3)
    old_model = GatedHighFrequencyDetailBranch(hidden_channels=16, num_blocks=1, norm_groups=8, highpass_kernel=5)
    for parameter in old_model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)
    checkpoint = tmp_path / "old.pt"
    torch.save({"step": 456, "model": old_model.state_dict()}, checkpoint)

    new_model = GatedHighFrequencyDetailBranch(hidden_channels=16, num_blocks=3, norm_groups=8, highpass_kernel=5)
    stats = init_model_from_checkpoint(
        checkpoint,
        new_model,
        torch.device("cpu"),
        identity_init_new_blocks=True,
    )

    base = torch.rand(1, 3, 24, 24)
    bicubic = torch.rand(1, 3, 24, 24)
    domain_id = torch.tensor([1], dtype=torch.long)
    old_outputs = old_model(base, bicubic, domain_id=domain_id)
    new_outputs = new_model(base, bicubic, domain_id=domain_id)

    assert stats["checkpoint_step"] == 456
    assert stats["identity_tensors"] == 4
    for old_output, new_output in zip(old_outputs, new_outputs):
        assert torch.allclose(new_output, old_output, atol=1e-6)


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
