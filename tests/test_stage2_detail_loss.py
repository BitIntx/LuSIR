import torch
from torch import nn

from tools.train.train_latent_pretrain import (
    artifact_excess_loss,
    compute_stage2_loss,
    local_highpass_energy_hinge_losses,
)


class IdentityDecoder(nn.Module):
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class FeatureLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(prediction, target)


def test_stage2_detail_loss_backpropagates_through_decoded_objectives() -> None:
    prediction = torch.zeros(2, 3, 8, 8, requires_grad=True)
    target = torch.rand_like(prediction)
    reference = torch.zeros_like(prediction)
    loss, components = compute_stage2_loss(
        prediction,
        target,
        target,
        reference,
        IdentityDecoder(),
        {
            "latent": "charbonnier",
            "latent_weight": 0.25,
            "decoded_weight": 1.0,
            "edge_weight": 1.0,
            "highpass_weight": 2.0,
            "highpass_magnitude_weight": 1.0,
        },
    )

    loss.backward()

    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
    assert components["decoded_image"].shape == prediction.shape
    assert float(components["highpass_magnitude"].detach()) > 0.0


def test_stage2_perceptual_loss_backpropagates() -> None:
    prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = torch.ones_like(prediction)
    loss, components = compute_stage2_loss(
        prediction,
        target,
        target,
        prediction,
        IdentityDecoder(),
        {"latent_weight": 0.0, "perceptual_weight": 1.0},
        FeatureLoss(),
    )

    loss.backward()

    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
    assert float(components["perceptual"].detach()) > 0.0


def test_stage2_latent_only_loss_skips_decoder() -> None:
    prediction = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = torch.ones_like(prediction)
    loss, components = compute_stage2_loss(
        prediction,
        target,
        target,
        prediction,
        IdentityDecoder(),
        {"latent": "charbonnier"},
    )

    loss.backward()

    assert prediction.grad is not None
    assert components["decoded_image"].numel() == 0
    assert float(components["decoded"]) == 0.0
    assert float(components["detail_decoded"]) == 0.0
    assert float(components["detail_highpass"]) == 0.0


def test_stage2_detail_weighted_loss_backpropagates() -> None:
    prediction = torch.zeros(1, 3, 16, 16, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, :, 4:12, 4:12] = 1.0
    reference = torch.zeros_like(prediction)
    loss, components = compute_stage2_loss(
        prediction,
        target,
        target,
        reference,
        IdentityDecoder(),
        {
            "latent_weight": 0.0,
            "detail_weighted": {
                "source": "prediction_missing",
                "decoded_weight": 1.0,
                "highpass_weight": 1.0,
                "top_fraction": 0.25,
                "top_mode": "binary",
                "mask_floor": 0.05,
                "highpass_kernel": 3,
                "patch_kernel": 3,
                "score_quantile": 0.95,
                "laplacian_kernel": 3,
            },
        },
    )

    loss.backward()

    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
    assert components["detail_mask"].shape == (1, 1, 16, 16)
    assert 0.05 <= float(components["detail_mask_mean"]) <= 0.30
    assert float(components["detail_decoded"].detach()) > 0.0
    assert float(components["detail_highpass"].detach()) > 0.0


def test_stage2_artifact_excess_loss_penalizes_unsupported_texture() -> None:
    target = torch.zeros(1, 3, 16, 16)
    prediction = target.clone()
    prediction[:, :, ::2, ::2] = 0.5
    prediction.requires_grad_()

    loss, active_fraction = artifact_excess_loss(
        prediction,
        target,
        highpass_kernel=3,
        patch_kernel=3,
        margin=0.0,
        temperature=0.01,
    )
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert float(active_fraction.detach()) > 0.0
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0


def test_stage2_artifact_excess_objective_backpropagates() -> None:
    prediction = torch.zeros(1, 3, 16, 16, requires_grad=True)
    prediction.data[:, :, ::2, ::2] = 0.5
    target = torch.zeros_like(prediction)
    loss, components = compute_stage2_loss(
        prediction,
        target,
        target,
        target,
        IdentityDecoder(),
        {
            "latent_weight": 0.0,
            "artifact_excess_weight": 2.0,
            "artifact_excess": {
                "highpass_kernel": 3,
                "patch_kernel": 3,
                "margin": 0.0,
                "temperature": 0.01,
            },
        },
    )
    loss.backward()

    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
    assert float(components["artifact_excess"].detach()) > 0.0
    assert float(components["artifact_excess_active"].detach()) > 0.0


def test_stage2_artifact_missing_objective_penalizes_oversmoothing() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[:, :, ::2, ::2] = 0.5
    prediction = torch.zeros_like(target, requires_grad=True)

    excess, missing, excess_active, missing_active = local_highpass_energy_hinge_losses(
        prediction,
        target,
        highpass_kernel=3,
        patch_kernel=3,
        excess_margin=0.0,
        missing_margin=0.0,
        temperature=0.01,
    )
    missing.backward()

    assert float(missing.detach()) > float(excess.detach())
    assert float(missing_active.detach()) > float(excess_active.detach())
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
