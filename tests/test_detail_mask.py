from __future__ import annotations

import torch

from sr_diffusion.detail_mask import (
    DetailMaskPredictor,
    detail_need_components,
    highpass,
    masked_highpass_oracle,
    observable_detail_proxies,
    top_fraction_mask,
)


def test_detail_mask_predictor_output_is_bounded_and_full_resolution() -> None:
    model = DetailMaskPredictor(latent_channels=4, hidden_channels=16, num_blocks=2, norm_groups=8)
    base = torch.rand(2, 3, 32, 32)
    bicubic = torch.rand(2, 3, 32, 32)
    condition = torch.rand(2, 4, 8, 8)
    domain_id = torch.tensor([0, 1], dtype=torch.long)
    prediction = model(base, bicubic, condition, domain_id)
    assert prediction.shape == (2, 1, 32, 32)
    assert prediction.min() >= 0.0
    assert prediction.max() <= 1.0


def test_detail_need_prefers_missing_texture_over_excess_texture() -> None:
    target = torch.full((1, 3, 32, 32), 0.5)
    target[:, :, 4:12, 4:12] += torch.tensor([[-0.15, 0.15] * 4] * 8)
    base = target.clone()
    base[:, :, 4:12, 4:12] = 0.5
    base[:, :, 20:28, 20:28] += torch.tensor([[-0.2, 0.2] * 4] * 8)

    components = detail_need_components(base, target, highpass_kernel=5, patch_kernel=3)

    missing_score = components["score"][:, :, 4:12, 4:12].mean()
    excess_score = components["score"][:, :, 20:28, 20:28].mean()
    assert missing_score > excess_score + 0.25
    assert components["missing"][:, :, 4:12, 4:12].mean() > 0
    assert components["excess"][:, :, 20:28, 20:28].mean() > 0


def test_top_fraction_mask_has_requested_coverage() -> None:
    score = torch.arange(100, dtype=torch.float32).view(1, 1, 10, 10)
    mask = top_fraction_mask(score, 0.2)
    assert mask.sum() == 20
    assert mask.flatten()[80:].sum() == 20


def test_highpass_does_not_create_false_image_border() -> None:
    constant = torch.ones(1, 3, 24, 24)
    assert torch.allclose(highpass(constant, 15), torch.zeros_like(constant), atol=1e-6)


def test_observable_proxies_and_oracle_keep_expected_shapes() -> None:
    torch.manual_seed(0)
    target = torch.rand(2, 3, 24, 24)
    base = target * 0.8 + 0.1
    bicubic = torch.nn.functional.avg_pool2d(target, 3, stride=1, padding=1)
    components = detail_need_components(base, target, highpass_kernel=5, patch_kernel=3)
    proxies = observable_detail_proxies(base, bicubic, highpass_kernel=5, patch_kernel=3)
    mask = top_fraction_mask(components["score"], 0.25)
    oracle = masked_highpass_oracle(base, target, mask, highpass_kernel=5)

    assert components["score"].shape == (2, 1, 24, 24)
    assert set(proxies) == {"base_detail", "bicubic_detail", "base_bicubic_gap", "highpass_disagreement"}
    assert all(proxy.shape == (2, 1, 24, 24) for proxy in proxies.values())
    assert oracle.shape == target.shape
    assert torch.isfinite(oracle).all()
