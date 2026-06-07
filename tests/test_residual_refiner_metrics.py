from __future__ import annotations

import torch

from train_residual_refiner import laplacian_response, metric_highpass, ssim_per_image


def test_ssim_is_one_for_identical_images() -> None:
    image = torch.rand(2, 3, 32, 32)
    assert torch.allclose(ssim_per_image(image, image), torch.ones(2), atol=1e-5)


def test_detail_responses_are_zero_for_constant_image() -> None:
    image = torch.full((1, 3, 32, 32), 0.5)
    assert metric_highpass(image).abs().max() < 1e-6
    assert laplacian_response(image).abs().max() < 1e-6
