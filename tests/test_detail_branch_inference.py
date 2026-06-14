from __future__ import annotations

from pathlib import Path

import torch

from tools.infer.infer_detail_branch import detail_batch, resolve_checkpoint
from tools.train.train_detail_branch import GatedHighFrequencyDetailBranch


class _ConditionEncoder(torch.nn.Module):
    def forward(self, lr: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        del domain_id
        return lr[:, :2]


class _Decoder(torch.nn.Module):
    def decode(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.cat([condition, condition[:, :1]], dim=1)


class _ZeroMask(torch.nn.Module):
    def forward(
        self,
        base: torch.Tensor,
        bicubic: torch.Tensor,
        condition: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        del bicubic, condition, domain_id
        return torch.zeros_like(base[:, :1])


def test_detail_branch_inference_strength_zero_returns_base() -> None:
    model = GatedHighFrequencyDetailBranch(
        latent_channels=2,
        hidden_channels=8,
        num_blocks=1,
        norm_groups=4,
        use_condition_latent=True,
        highpass_kernel=3,
    )
    lr = torch.rand(1, 3, 16, 16)
    domain_id = torch.zeros(1, dtype=torch.long)

    base, detail = detail_batch(
        vae=_Decoder(),
        condition_encoder=_ConditionEncoder(),
        detail_branch=model,
        lr=lr,
        domain_id=domain_id,
        dtype_name="fp32",
        detail_strength=0.0,
    )

    assert torch.allclose(detail, base)


def test_detail_branch_inference_applies_learned_mask_floor() -> None:
    torch.manual_seed(7)
    model = GatedHighFrequencyDetailBranch(
        latent_channels=2,
        hidden_channels=8,
        num_blocks=1,
        norm_groups=4,
        use_condition_latent=True,
        highpass_kernel=3,
    )
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)
    lr = torch.rand(1, 3, 16, 16)
    domain_id = torch.zeros(1, dtype=torch.long)

    base, unmasked = detail_batch(
        vae=_Decoder(),
        condition_encoder=_ConditionEncoder(),
        detail_branch=model,
        lr=lr,
        domain_id=domain_id,
        dtype_name="fp32",
        detail_strength=1.0,
    )
    _, masked = detail_batch(
        vae=_Decoder(),
        condition_encoder=_ConditionEncoder(),
        detail_branch=model,
        lr=lr,
        domain_id=domain_id,
        dtype_name="fp32",
        detail_strength=1.0,
        detail_mask_predictor=_ZeroMask(),
        detail_mask_floor=0.25,
    )

    assert torch.allclose(masked, (base + (unmasked - base) * 0.25).clamp(0.0, 1.0), atol=1e-6)


def test_detail_branch_inference_resolves_config_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "detail.pt"
    checkpoint.touch()
    config = {
        "_config_path": str(tmp_path / "config.yaml"),
        "inference": {"checkpoint": "detail.pt"},
    }

    assert resolve_checkpoint(config, None) == checkpoint.resolve()
