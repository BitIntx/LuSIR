from __future__ import annotations

import numpy as np

from tools.eval.eval_sr_benchmark import summarize
from tools.eval.sr_benchmark_metrics import benchmark_metrics, crop_border, matlab_bicubic_resize, rgb_to_y, ssim


def test_benchmark_metrics_are_exact_for_identical_images() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    metrics = benchmark_metrics(image, image, border=4)

    assert np.isinf(metrics["y_psnr"])
    assert np.isinf(metrics["rgb_psnr"])
    assert abs(metrics["y_ssim"] - 1.0) < 1e-12
    assert "rgb_ssim" not in metrics


def test_rgb_to_y_preserves_shape_and_luminance_order() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = [255, 255, 255]
    y = rgb_to_y(image)

    assert y.shape == (2, 2)
    assert y[0, 0] > y[1, 1]


def test_crop_border_removes_scale_pixels() -> None:
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    assert crop_border(image, 4).shape == (12, 16, 3)


def test_matlab_bicubic_resize_has_expected_shape_and_range() -> None:
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    resized = matlab_bicubic_resize(image, 4.0)

    assert resized.shape == (32, 36, 3)
    assert resized.dtype == np.uint8
    assert ssim(resized, resized) == 1.0


def test_benchmark_summary_reports_delta_and_wins_vs_bicubic() -> None:
    records = [
        {"dataset": "Set5", "id": "a", "candidate": "bicubic", "y_psnr": 20.0, "y_ssim": 0.7, "rgb_psnr": 19.0},
        {"dataset": "Set5", "id": "a", "candidate": "model", "y_psnr": 21.0, "y_ssim": 0.8, "rgb_psnr": 20.0},
    ]

    summary = summarize(records)

    assert summary["candidates"]["model"]["mean_y_psnr_delta_vs_bicubic"] == 1.0
    assert summary["candidates"]["model"]["wins_y_psnr_vs_bicubic"] == 1
