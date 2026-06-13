from __future__ import annotations

import torch
import torch.nn.functional as F

from sr_diffusion.wavelet import haar_dwt2, haar_high_bands, haar_idwt2, image_from_haar_high


def test_haar_round_trip_is_exact() -> None:
    image = torch.randn(2, 3, 16, 20)
    reconstructed = haar_idwt2(haar_dwt2(image), channels=3)
    assert torch.allclose(reconstructed, image, atol=1e-6)


def test_high_only_reconstruction_has_zero_two_by_two_mean() -> None:
    image = torch.randn(2, 3, 16, 20)
    residual = image_from_haar_high(haar_high_bands(image), channels=3)
    block_mean = F.avg_pool2d(residual, kernel_size=2, stride=2)
    assert torch.allclose(block_mean, torch.zeros_like(block_mean), atol=1e-6)


def test_haar_rejects_odd_spatial_size() -> None:
    try:
        haar_dwt2(torch.randn(1, 3, 15, 16))
    except ValueError as exc:
        assert "even spatial dimensions" in str(exc)
    else:
        raise AssertionError("Expected odd spatial size to be rejected")
