from __future__ import annotations

import torch


def haar_dwt2(image: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D Haar transform with bands ordered LL, LH, HL, HH."""
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(image.shape)}")
    if image.shape[-2] % 2 or image.shape[-1] % 2:
        raise ValueError(f"Haar DWT requires even spatial dimensions, got {tuple(image.shape[-2:])}")
    top_left = image[..., 0::2, 0::2]
    top_right = image[..., 0::2, 1::2]
    bottom_left = image[..., 1::2, 0::2]
    bottom_right = image[..., 1::2, 1::2]
    ll = (top_left + top_right + bottom_left + bottom_right) * 0.5
    lh = (-top_left - top_right + bottom_left + bottom_right) * 0.5
    hl = (-top_left + top_right - bottom_left + bottom_right) * 0.5
    hh = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)


def haar_idwt2(coefficients: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    """Inverse of :func:`haar_dwt2`."""
    if coefficients.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(coefficients.shape)}")
    channels = int(channels or coefficients.shape[1] // 4)
    if coefficients.shape[1] != channels * 4:
        raise ValueError(f"Expected {channels * 4} coefficients, got {coefficients.shape[1]}")
    ll, lh, hl, hh = torch.split(coefficients, channels, dim=1)
    top_left = (ll - lh - hl + hh) * 0.5
    top_right = (ll - lh + hl - hh) * 0.5
    bottom_left = (ll + lh - hl - hh) * 0.5
    bottom_right = (ll + lh + hl + hh) * 0.5
    output = coefficients.new_empty(
        coefficients.shape[0],
        channels,
        coefficients.shape[-2] * 2,
        coefficients.shape[-1] * 2,
    )
    output[..., 0::2, 0::2] = top_left
    output[..., 0::2, 1::2] = top_right
    output[..., 1::2, 0::2] = bottom_left
    output[..., 1::2, 1::2] = bottom_right
    return output


def haar_high_bands(image: torch.Tensor) -> torch.Tensor:
    channels = int(image.shape[1])
    return haar_dwt2(image)[:, channels:]


def image_from_haar_high(high_bands: torch.Tensor, channels: int = 3) -> torch.Tensor:
    if high_bands.shape[1] != channels * 3:
        raise ValueError(f"Expected {channels * 3} high-frequency channels, got {high_bands.shape[1]}")
    ll = high_bands.new_zeros(high_bands.shape[0], channels, *high_bands.shape[-2:])
    return haar_idwt2(torch.cat([ll, high_bands], dim=1), channels=channels)
