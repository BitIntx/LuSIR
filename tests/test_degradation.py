from __future__ import annotations

import random

import numpy as np
from PIL import Image

from sr_diffusion.degradations import DegradationPipeline


def test_degradation_outputs_expected_size() -> None:
    image = Image.new("RGB", (128, 128), (200, 120, 80))
    pipeline = DegradationPipeline.from_preset("mild", scale=4)
    lr = pipeline.apply(image, rng=random.Random(0), out_size=32)
    assert lr.size == (32, 32)
    assert lr.mode == "RGB"


def test_photo_v2_degradation_outputs_expected_size() -> None:
    image = Image.new("RGB", (128, 128), (200, 120, 80))
    pipeline = DegradationPipeline.from_preset("photo_v2", scale=4)
    lr = pipeline.apply(image, rng=random.Random(0), out_size=32)
    assert lr.size == (32, 32)
    assert lr.mode == "RGB"


def test_photo_v3_noise_mix_outputs_expected_size() -> None:
    image = Image.new("RGB", (128, 128), (200, 120, 80))
    pipeline = DegradationPipeline.from_preset("photo_v3_noise_mix", scale=4)
    lr = pipeline.apply(image, rng=random.Random(0), out_size=32)
    assert lr.size == (32, 32)
    assert lr.mode == "RGB"


def test_photo_detail_mix_outputs_expected_size() -> None:
    image = Image.new("RGB", (128, 128), (200, 120, 80))
    pipeline = DegradationPipeline.from_preset("photo_detail_mix", scale=4)
    lr = pipeline.apply(image, rng=random.Random(0), out_size=32)
    assert lr.size == (32, 32)
    assert lr.mode == "RGB"


def test_benchmark_bicubic_matches_pil_bicubic_downsample() -> None:
    gradient = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    image = Image.fromarray(np.stack([gradient, gradient.T, np.flipud(gradient)], axis=-1), mode="RGB")
    pipeline = DegradationPipeline.from_preset("benchmark_bicubic", scale=4)

    lr = pipeline.apply(image, rng=random.Random(0), out_size=32)
    expected = image.resize((32, 32), resample=Image.Resampling.BICUBIC)

    assert np.array_equal(np.asarray(lr), np.asarray(expected))


def test_forced_artifact_degradation_stays_rgb_uint8() -> None:
    gradient = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    image = Image.fromarray(np.stack([gradient, np.flipud(gradient), gradient.T], axis=-1), mode="RGB")
    pipeline = DegradationPipeline(
        {
            "blur_prob": 1.0,
            "blur_radius": 1.5,
            "lr_blur_prob": 1.0,
            "lr_blur_radius": 0.4,
            "sensor_noise_prob": 1.0,
            "sensor_read_sigma": 3.0,
            "sensor_shot_scale": 8.0,
            "gaussian_noise_prob": 1.0,
            "gaussian_sigma": 3.0,
            "chroma_noise_prob": 1.0,
            "chroma_noise_sigma": 8.0,
            "chroma_noise_blur_radius": 0.5,
            "poisson_noise_prob": 1.0,
            "jpeg_prob": 1.0,
            "jpeg_quality": 50,
            "webp_prob": 1.0,
            "webp_quality": 50,
            "ringing_prob": 1.0,
            "ringing_radius": 3,
            "ringing_strength": 0.08,
            "color_prob": 1.0,
            "color_jitter": [0.9, 1.1],
            "color_shift_prob": 1.0,
            "color_shift_gain": [0.95, 1.05],
            "color_shift_bias": [-4.0, 4.0],
            "banding_prob": 1.0,
            "banding_levels": 32,
            "sharpen_prob": 1.0,
            "sharpen_factor": 1.4,
            "oversharpen_prob": 1.0,
            "oversharpen_radius": 1.0,
            "oversharpen_percent": 180,
            "oversharpen_threshold": 1,
        },
        scale=4,
    )
    lr = pipeline.apply(image, rng=random.Random(123), out_size=32)
    array = np.asarray(lr)
    assert lr.size == (32, 32)
    assert lr.mode == "RGB"
    assert array.dtype == np.uint8
