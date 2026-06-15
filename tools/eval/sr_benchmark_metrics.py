from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def crop_border(image: np.ndarray, border: int) -> np.ndarray:
    if border < 0:
        raise ValueError(f"border must be non-negative: {border}")
    if border == 0:
        return image
    if image.shape[0] <= border * 2 or image.shape[1] <= border * 2:
        raise ValueError(f"image is too small for border={border}: {image.shape}")
    return image[border:-border, border:-border, ...]


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    """Convert uint8/float RGB in [0, 255] to MATLAB-compatible BT.601 Y."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    rgb = image.astype(np.float64) / 255.0
    return np.dot(rgb, [65.481, 128.553, 24.966]) + 16.0


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    center = window_size // 2
    values = torch.arange(window_size, dtype=torch.float64) - center
    kernel = torch.exp(-(values**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return torch.outer(kernel, kernel)


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Calculate MATLAB-style SSIM using an 11x11 valid Gaussian window."""
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")
    if prediction.ndim == 2:
        prediction = prediction[..., None]
        target = target[..., None]
    if prediction.shape[0] < 11 or prediction.shape[1] < 11:
        raise ValueError(f"SSIM requires images at least 11x11, got {prediction.shape}")
    try:
        import cv2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        values = []
        for channel in range(prediction.shape[2]):
            pred = prediction[..., channel].astype(np.float64)
            truth = target[..., channel].astype(np.float64)
            mu_pred = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
            mu_truth = cv2.filter2D(truth, -1, window)[5:-5, 5:-5]
            mu_pred_sq = mu_pred**2
            mu_truth_sq = mu_truth**2
            mu_both = mu_pred * mu_truth
            sigma_pred = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu_pred_sq
            sigma_truth = cv2.filter2D(truth**2, -1, window)[5:-5, 5:-5] - mu_truth_sq
            sigma_both = cv2.filter2D(pred * truth, -1, window)[5:-5, 5:-5] - mu_both
            c1 = (0.01 * 255.0) ** 2
            c2 = (0.03 * 255.0) ** 2
            values.append(
                np.mean(
                    ((2.0 * mu_both + c1) * (2.0 * sigma_both + c2))
                    / ((mu_pred_sq + mu_truth_sq + c1) * (sigma_pred + sigma_truth + c2))
                )
            )
        return float(np.mean(values))
    except ImportError:
        pass

    pred = torch.from_numpy(prediction.astype(np.float64)).permute(2, 0, 1).unsqueeze(0)
    truth = torch.from_numpy(target.astype(np.float64)).permute(2, 0, 1).unsqueeze(0)
    channels = pred.shape[1]
    window = _gaussian_window().view(1, 1, 11, 11).expand(channels, 1, 11, 11)
    mu_pred = F.conv2d(pred, window, groups=channels)
    mu_truth = F.conv2d(truth, window, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_truth_sq = mu_truth.square()
    mu_both = mu_pred * mu_truth
    sigma_pred = F.conv2d(pred.square(), window, groups=channels) - mu_pred_sq
    sigma_truth = F.conv2d(truth.square(), window, groups=channels) - mu_truth_sq
    sigma_both = F.conv2d(pred * truth, window, groups=channels) - mu_both
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    result = ((2.0 * mu_both + c1) * (2.0 * sigma_both + c2)) / (
        (mu_pred_sq + mu_truth_sq + c1) * (sigma_pred + sigma_truth + c2)
    )
    return float(result.mean())


def benchmark_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    border: int,
    *,
    include_ssim: bool = True,
    include_rgb_ssim: bool = False,
) -> dict[str, float]:
    prediction = crop_border(prediction, border)
    target = crop_border(target, border)
    pred_y = rgb_to_y(prediction)
    target_y = rgb_to_y(target)
    metrics = {
        "y_psnr": psnr(pred_y, target_y),
        "rgb_psnr": psnr(prediction, target),
    }
    if include_ssim:
        metrics["y_ssim"] = ssim(pred_y, target_y)
    if include_rgb_ssim:
        if not include_ssim:
            raise ValueError("include_rgb_ssim requires include_ssim=True")
        metrics["rgb_ssim"] = ssim(prediction, target)
    return metrics


# The bicubic resize implementation below is adapted from BasicSR's
# Apache-2.0-licensed basicsr/utils/matlab_functions.py.
def _cubic(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.abs(x)
    abs_x2 = abs_x**2
    abs_x3 = abs_x**3
    return (1.5 * abs_x3 - 2.5 * abs_x2 + 1.0) * (abs_x <= 1).type_as(abs_x) + (
        -0.5 * abs_x3 + 2.5 * abs_x2 - 4.0 * abs_x + 2.0
    ) * (((abs_x > 1) * (abs_x <= 2)).type_as(abs_x))


def _resize_weights(in_length: int, out_length: int, scale: float) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    kernel_width = 4.0 / scale if scale < 1.0 else 4.0
    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = torch.floor(u - kernel_width / 2.0)
    width = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1) + torch.arange(width).view(1, width)
    distances = u.view(out_length, 1) - indices
    weights = scale * _cubic(distances * scale) if scale < 1.0 else _cubic(distances)
    weights /= weights.sum(dim=1, keepdim=True)
    if torch.all(weights[:, 0] == 0):
        weights = weights[:, 1:]
        indices = indices[:, 1:]
    if torch.all(weights[:, -1] == 0):
        weights = weights[:, :-1]
        indices = indices[:, :-1]
    symmetric_start = int(-indices.min() + 1)
    symmetric_end = int(indices.max() - in_length)
    indices = indices + symmetric_start - 1
    return weights.contiguous(), indices.long().contiguous(), symmetric_start, symmetric_end


@torch.no_grad()
def matlab_bicubic_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize HWC RGB in [0, 255] with MATLAB-compatible antialiased bicubic."""
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got {image.shape}")
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    channels, in_height, in_width = tensor.shape
    out_height = math.ceil(in_height * scale)
    out_width = math.ceil(in_width * scale)
    weights_h, indices_h, start_h, end_h = _resize_weights(in_height, out_height, scale)
    weights_w, indices_w, start_w, end_w = _resize_weights(in_width, out_width, scale)

    augmented_h = torch.empty(channels, in_height + start_h + end_h, in_width)
    augmented_h[:, start_h : start_h + in_height] = tensor
    augmented_h[:, :start_h] = tensor[:, :start_h].flip(1)
    augmented_h[:, start_h + in_height :] = tensor[:, -end_h:].flip(1)
    resized_h = (augmented_h[:, indices_h, :] * weights_h.view(1, out_height, -1, 1)).sum(dim=2)

    augmented_w = torch.empty(channels, out_height, in_width + start_w + end_w)
    augmented_w[:, :, start_w : start_w + in_width] = resized_h
    augmented_w[:, :, :start_w] = resized_h[:, :, :start_w].flip(2)
    augmented_w[:, :, start_w + in_width :] = resized_h[:, :, -end_w:].flip(2)
    output = (augmented_w[:, :, indices_w] * weights_w.view(1, 1, out_width, -1)).sum(dim=3)
    return np.round(output.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
