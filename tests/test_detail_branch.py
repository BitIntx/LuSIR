from __future__ import annotations

import torch
from torch import nn

from sr_diffusion.detail_adversarial import MaskedHighpassPatchDiscriminator
from tools.train.train_detail_branch import (
    GatedHighFrequencyDetailBranch,
    apply_detail_mask_policy,
    artifact_negative_residual_loss,
    gt_highpass_hinge_losses,
    init_model_from_checkpoint,
    load_checkpoint,
    make_training_detail_mask,
    save_checkpoint,
    teacher_improvement_mask,
    training_loss,
)


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


def test_detail_mask_policy_keeps_soft_mask_by_default() -> None:
    detail_mask = torch.tensor([[[[0.1, 0.4], [0.8, 0.2]]]])

    masked = apply_detail_mask_policy(detail_mask, {})

    assert torch.allclose(masked, detail_mask)


def test_detail_mask_policy_supports_binary_top_fraction() -> None:
    detail_mask = torch.tensor([[[[0.1, 0.4], [0.8, 0.2]]]])

    masked = apply_detail_mask_policy(detail_mask, {"top_fraction": 0.5})

    assert torch.allclose(masked, torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]))


def test_detail_mask_policy_supports_soft_top_fraction() -> None:
    detail_mask = torch.tensor([[[[0.1, 0.4], [0.8, 0.2]]]])

    masked = apply_detail_mask_policy(detail_mask, {"top_fraction": 0.5, "top_mode": "soft"})

    assert torch.allclose(masked, torch.tensor([[[[0.0, 0.4], [0.8, 0.0]]]]))


def test_training_detail_mask_defaults_to_learned_mask() -> None:
    learned = torch.rand(1, 1, 8, 8)
    base = torch.rand(1, 3, 8, 8)
    hr = torch.rand(1, 3, 8, 8)

    mask = make_training_detail_mask(base, hr, learned, {})

    assert mask is learned


def test_training_detail_mask_can_use_gt_detail_need_top_fraction() -> None:
    base = torch.full((1, 3, 24, 24), 0.5)
    hr = base.clone()
    hr[:, :, 8:16, 8:16] = (hr[:, :, 8:16, 8:16] + 0.08 * torch.randn(1, 3, 8, 8)).clamp(0.0, 1.0)

    mask = make_training_detail_mask(
        base,
        hr,
        learned_mask=None,
        mask_cfg={
            "source": "gt_detail_need",
            "top_fraction": 0.25,
            "top_mode": "binary",
            "highpass_kernel": 5,
            "patch_kernel": 3,
        },
    )

    assert mask is not None
    assert mask.shape == (1, 1, 24, 24)
    assert torch.allclose(mask.mean(), torch.tensor(0.25), atol=1.0 / (24 * 24))


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


def test_artifact_negative_residual_loss_penalizes_flat_targets_more() -> None:
    pattern = torch.tensor([[0.0, 1.0] * 12, [1.0, 0.0] * 12] * 12)
    pattern = pattern.mul(2.0).sub(1.0).view(1, 1, 24, 24)
    residual = pattern.repeat(1, 3, 1, 1) * 0.08
    flat_hr = torch.full((1, 3, 24, 24), 0.5)
    textured_hr = (flat_hr + residual).clamp(0.0, 1.0)
    base = flat_hr.clone()
    detail_mask = torch.ones(1, 1, 24, 24)

    flat_loss, flat_weight = artifact_negative_residual_loss(
        residual=residual,
        base_sr=base,
        hr=flat_hr,
        detail_mask=detail_mask,
        highpass_kernel=5,
        patch_kernel=3,
        flat_threshold=0.02,
    )
    textured_loss, textured_weight = artifact_negative_residual_loss(
        residual=residual,
        base_sr=base,
        hr=textured_hr,
        detail_mask=detail_mask,
        highpass_kernel=5,
        patch_kernel=3,
        flat_threshold=0.02,
    )

    assert flat_weight > textured_weight
    assert flat_loss > textured_loss


def test_teacher_improvement_mask_selects_gt_aligned_teacher_detail() -> None:
    torch.manual_seed(8)
    base = torch.full((1, 3, 24, 24), 0.5)
    pattern = torch.tensor([[0.0, 1.0] * 12, [1.0, 0.0] * 12] * 12).view(1, 1, 24, 24)
    hr = (base + pattern.repeat(1, 3, 1, 1) * 0.04).clamp(0.0, 1.0)
    good_teacher = (base + pattern.repeat(1, 3, 1, 1) * 0.035).clamp(0.0, 1.0)
    bad_teacher = base.clone()
    detail_mask = torch.ones(1, 1, 24, 24)

    good_mask, good_stats = teacher_improvement_mask(
        good_teacher,
        base,
        hr,
        detail_mask,
        highpass_kernel=5,
        patch_kernel=3,
        ratio=0.95,
        margin=0.0,
    )
    bad_mask, bad_stats = teacher_improvement_mask(
        bad_teacher,
        base,
        hr,
        detail_mask,
        highpass_kernel=5,
        patch_kernel=3,
        ratio=0.95,
        margin=0.0,
    )

    assert good_mask.mean() > 0.9
    assert bad_mask.mean() == 0
    assert good_stats["improvement"] > bad_stats["improvement"]


def test_gt_highpass_hinge_losses_penalize_regressed_detail() -> None:
    base = torch.full((1, 3, 24, 24), 0.5)
    pattern = torch.tensor([[0.0, 1.0] * 12, [1.0, 0.0] * 12] * 12).view(1, 1, 24, 24)
    hr = (base + pattern.repeat(1, 3, 1, 1) * 0.04).clamp(0.0, 1.0)
    improved = (base + pattern.repeat(1, 3, 1, 1) * 0.035).clamp(0.0, 1.0)
    regressed = base.clone()
    weight = torch.ones(1, 1, 24, 24)

    improved_loss, _ = gt_highpass_hinge_losses(
        improved,
        base,
        hr,
        positive_weight=weight,
        guard_weight=weight,
        highpass_kernel=5,
        patch_kernel=3,
        positive_ratio=0.98,
    )
    regressed_loss, _ = gt_highpass_hinge_losses(
        regressed,
        base,
        hr,
        positive_weight=weight,
        guard_weight=weight,
        highpass_kernel=5,
        patch_kernel=3,
        positive_ratio=0.98,
    )

    assert regressed_loss > improved_loss


def test_detail_branch_negative_residual_loss_backpropagates() -> None:
    torch.manual_seed(6)
    model = GatedHighFrequencyDetailBranch(
        hidden_channels=16,
        num_blocks=1,
        norm_groups=8,
        highpass_kernel=5,
    )
    base = torch.rand(2, 3, 32, 32) * 0.5 + 0.25
    bicubic = torch.rand(2, 3, 32, 32)
    condition = torch.rand(2, 16, 8, 8)
    hr = (base + 0.02 * torch.randn_like(base)).clamp(0.0, 1.0)
    domain_id = torch.tensor([0, 1], dtype=torch.long)
    detail_mask = torch.ones(2, 1, 32, 32)

    loss, parts = training_loss(
        model=model,
        base_sr=base,
        bicubic=bicubic,
        condition=condition,
        hr=hr,
        domain_id=domain_id,
        loss_cfg={
            "highpass_kernel": 5,
            "lowpass_kernel": 7,
            "negative_residual_weight": 1.0,
            "negative_residual": {"patch_kernel": 3, "flat_threshold": 0.02},
        },
        detail_mask=detail_mask,
    )
    loss.backward()

    assert parts["negative_residual"] >= 0
    assert parts["negative_weight"] > 0
    assert model.output.weight.grad is not None
    assert model.output.weight.grad.abs().sum() > 0


class _MaskedPixelLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        difference = (prediction - target).abs().mean(dim=1, keepdim=True)
        return (difference * mask).sum() / mask.sum().clamp_min(1e-8)


def test_detail_branch_optional_masked_perceptual_and_adversarial_losses_backpropagate() -> None:
    torch.manual_seed(5)
    model = GatedHighFrequencyDetailBranch(
        hidden_channels=16,
        num_blocks=1,
        norm_groups=8,
        highpass_kernel=5,
    )
    discriminator = MaskedHighpassPatchDiscriminator(
        base_channels=8,
        channel_multipliers=(1, 2),
        highpass_kernel=5,
        use_spectral_norm=False,
    )
    base = torch.rand(2, 3, 32, 32) * 0.5 + 0.25
    bicubic = torch.rand(2, 3, 32, 32)
    condition = torch.rand(2, 16, 8, 8)
    hr = (base + 0.05 * torch.randn_like(base)).clamp(0.0, 1.0)
    domain_id = torch.tensor([0, 1], dtype=torch.long)
    detail_mask = torch.rand(2, 1, 32, 32)

    loss, parts = training_loss(
        model=model,
        base_sr=base,
        bicubic=bicubic,
        condition=condition,
        hr=hr,
        domain_id=domain_id,
        loss_cfg={
            "highpass_kernel": 5,
            "lowpass_kernel": 7,
            "masked_perceptual_weight": 0.1,
        },
        detail_mask=detail_mask,
        perceptual_model=_MaskedPixelLoss(),
        discriminator=discriminator,
        adversarial_cfg={"generator_weight": 0.01, "mask_floor": 0.05},
        adversarial_active=True,
    )
    loss.backward()

    assert parts["masked_perceptual"] > 0
    assert parts["adversarial"] > 0
    assert model.output.weight.grad is not None
    assert model.output.weight.grad.abs().sum() > 0


def test_detail_branch_checkpoint_round_trips_optional_discriminator(tmp_path) -> None:
    model = GatedHighFrequencyDetailBranch(hidden_channels=8, num_blocks=1, norm_groups=4, highpass_kernel=5)
    discriminator = MaskedHighpassPatchDiscriminator(
        base_channels=4,
        channel_multipliers=(1,),
        highpass_kernel=5,
        use_spectral_norm=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-5)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in discriminator.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    discriminator_optimizer.step()
    checkpoint = tmp_path / "detail_v3.pt"

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=17,
        config={},
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
    )

    restored_model = GatedHighFrequencyDetailBranch(hidden_channels=8, num_blocks=1, norm_groups=4, highpass_kernel=5)
    restored_discriminator = MaskedHighpassPatchDiscriminator(
        base_channels=4,
        channel_multipliers=(1,),
        highpass_kernel=5,
        use_spectral_norm=False,
    )
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    restored_discriminator_optimizer = torch.optim.AdamW(restored_discriminator.parameters(), lr=2e-5)
    step = load_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
        discriminator=restored_discriminator,
        discriminator_optimizer=restored_discriminator_optimizer,
    )

    assert step == 17
    assert restored_optimizer.state_dict()["state"]
    assert restored_discriminator_optimizer.state_dict()["state"]
    for expected, actual in zip(discriminator.parameters(), restored_discriminator.parameters(), strict=True):
        assert torch.allclose(actual, expected)
