from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_degradation_preset(
    name: str,
    preset_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(preset_path) if preset_path else _repo_root() / "configs" / "degradation_presets.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    presets = data.get("presets", {})
    if name not in presets:
        available = ", ".join(sorted(presets))
        raise KeyError(f"Unknown degradation preset '{name}'. Available: {available}")
    return presets[name]


def _chance(rng: random.Random, probability: float) -> bool:
    return rng.random() < float(probability)


def _range_value(rng: random.Random, value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if len(value) != 2:
        raise ValueError(f"Expected scalar or [min, max], got {value}")
    return rng.uniform(float(value[0]), float(value[1]))


def _range_int(rng: random.Random, value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if len(value) != 2:
        raise ValueError(f"Expected int or [min, max], got {value}")
    return rng.randint(int(value[0]), int(value[1]))


def _to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0, 255).round().astype(np.uint8)


def _resolve_downsample_mode(name: str) -> Image.Resampling:
    modes = {
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "lanczos": Image.Resampling.LANCZOS,
    }
    normalized = str(name).lower()
    if normalized not in modes:
        raise ValueError(f"Unsupported downsample_mode {name!r}. Available: {sorted(modes)}")
    return modes[normalized]


class DegradationPipeline:
    def __init__(self, config: dict[str, Any], scale: int = 4):
        self.config = config
        self.scale = scale
        self._mix_pipelines: list[tuple[float, DegradationPipeline]] = []
        for item in config.get("preset_mix", []):
            weight = float(item.get("weight", 1.0))
            if weight <= 0.0:
                continue
            self._mix_pipelines.append((weight, DegradationPipeline.from_preset(str(item["preset"]), scale=scale)))
        self._mix_total = sum(weight for weight, _ in self._mix_pipelines)

    @classmethod
    def from_preset(cls, name: str, scale: int = 4) -> "DegradationPipeline":
        return cls(load_degradation_preset(name), scale=scale)

    def apply(
        self,
        image: Image.Image,
        rng: random.Random | None = None,
        out_size: int | None = None,
    ) -> Image.Image:
        rng = rng or random.Random()
        if self._mix_pipelines:
            threshold = rng.random() * self._mix_total
            cumulative = 0.0
            for weight, pipeline in self._mix_pipelines:
                cumulative += weight
                if threshold <= cumulative:
                    return pipeline.apply(image, rng=rng, out_size=out_size)
            return self._mix_pipelines[-1][1].apply(image, rng=rng, out_size=out_size)

        cfg = self.config
        hr = image.convert("RGB")

        if _chance(rng, cfg.get("blur_prob", 0.0)):
            radius = _range_value(rng, cfg.get("blur_radius"), 0.6)
            hr = hr.filter(ImageFilter.GaussianBlur(radius=radius))

        target_size = out_size or max(1, min(hr.size) // self.scale)
        downsample_mode = cfg.get("downsample_mode")
        if downsample_mode is None:
            resample = rng.choice(
                [
                    Image.Resampling.BICUBIC,
                    Image.Resampling.BILINEAR,
                    Image.Resampling.LANCZOS,
                ]
            )
        else:
            resample = _resolve_downsample_mode(str(downsample_mode))
        lr = hr.resize((target_size, target_size), resample=resample)

        if float(cfg.get("lr_blur_prob", 0.0)) > 0.0 and _chance(rng, cfg.get("lr_blur_prob", 0.0)):
            radius = _range_value(rng, cfg.get("lr_blur_radius"), 0.4)
            lr = lr.filter(ImageFilter.GaussianBlur(radius=radius))

        if float(cfg.get("sensor_noise_prob", 0.0)) > 0.0 and _chance(rng, cfg.get("sensor_noise_prob", 0.0)):
            read_sigma = _range_value(rng, cfg.get("sensor_read_sigma"), 2.0)
            shot_scale = _range_value(rng, cfg.get("sensor_shot_scale"), 8.0)
            lr = self._sensor_noise(lr, read_sigma=read_sigma, shot_scale=shot_scale, rng=rng)

        if _chance(rng, cfg.get("gaussian_noise_prob", 0.0)):
            sigma = _range_value(rng, cfg.get("gaussian_sigma"), 2.0)
            lr = self._gaussian_noise(lr, sigma=sigma, rng=rng)

        if float(cfg.get("chroma_noise_prob", 0.0)) > 0.0 and _chance(rng, cfg.get("chroma_noise_prob", 0.0)):
            sigma = _range_value(rng, cfg.get("chroma_noise_sigma"), 6.0)
            blur_radius = _range_value(rng, cfg.get("chroma_noise_blur_radius"), 0.0)
            lr = self._chroma_noise(lr, sigma=sigma, blur_radius=blur_radius, rng=rng)

        if _chance(rng, cfg.get("poisson_noise_prob", 0.0)):
            lr = self._poisson_noise(lr, rng=rng)

        if _chance(rng, cfg.get("jpeg_prob", 0.0)):
            quality = _range_int(rng, cfg.get("jpeg_quality"), 90)
            lr = self._compress(lr, fmt="JPEG", quality=quality)

        if _chance(rng, cfg.get("webp_prob", 0.0)):
            quality = _range_int(rng, cfg.get("webp_quality"), 90)
            lr = self._compress(lr, fmt="WEBP", quality=quality)

        if float(cfg.get("ringing_prob", 0.0)) > 0.0 and _chance(rng, cfg.get("ringing_prob", 0.0)):
            radius = _range_int(rng, cfg.get("ringing_radius"), 2)
            strength = _range_value(rng, cfg.get("ringing_strength"), 0.08)
            lr = self._ringing(lr, radius=radius, strength=strength)

        if _chance(rng, cfg.get("color_prob", 0.0)):
            jitter = cfg.get("color_jitter", [0.95, 1.05])
            lr = ImageEnhance.Color(lr).enhance(_range_value(rng, jitter, 1.0))
            lr = ImageEnhance.Contrast(lr).enhance(_range_value(rng, jitter, 1.0))
            lr = ImageEnhance.Brightness(lr).enhance(_range_value(rng, jitter, 1.0))

        if float(cfg.get("color_shift_prob", 0.0)) > 0.0 and _chance(rng, cfg.get("color_shift_prob", 0.0)):
            gain = cfg.get("color_shift_gain", [0.96, 1.04])
            bias = cfg.get("color_shift_bias", [-4.0, 4.0])
            lr = self._color_shift(lr, gain=gain, bias=bias, rng=rng)

        if _chance(rng, cfg.get("anime_line_prob", 0.0)):
            lr = self._anime_line_change(lr, rng=rng)

        if _chance(rng, cfg.get("banding_prob", 0.0)):
            levels = _range_int(rng, cfg.get("banding_levels"), 64)
            lr = self._banding(lr, levels=levels)

        if _chance(rng, cfg.get("sharpen_prob", 0.0)):
            factor = _range_value(rng, cfg.get("sharpen_factor"), 1.2)
            lr = ImageEnhance.Sharpness(lr).enhance(factor)

        if float(cfg.get("oversharpen_prob", 0.0)) > 0.0 and _chance(rng, cfg.get("oversharpen_prob", 0.0)):
            radius = _range_value(rng, cfg.get("oversharpen_radius"), 1.0)
            percent = _range_int(rng, cfg.get("oversharpen_percent"), 180)
            threshold = _range_int(rng, cfg.get("oversharpen_threshold"), 2)
            lr = lr.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

        return lr.convert("RGB")

    @staticmethod
    def _gaussian_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
        array = np.asarray(image).astype(np.float32)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        array = array + np_rng.normal(0.0, sigma, size=array.shape)
        return Image.fromarray(_to_uint8(array), mode="RGB")

    @staticmethod
    def _chroma_noise(image: Image.Image, sigma: float, blur_radius: float, rng: random.Random) -> Image.Image:
        array = np.asarray(image.convert("YCbCr")).astype(np.float32)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        noise = np_rng.normal(0.0, sigma, size=array.shape[:2] + (2,)).astype(np.float32)
        if blur_radius > 0.0:
            channels = []
            for channel in range(2):
                noise_image = Image.fromarray(_to_uint8(noise[..., channel] + 128.0), mode="L")
                noise_image = noise_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                channels.append(np.asarray(noise_image).astype(np.float32) - 128.0)
            noise = np.stack(channels, axis=-1)
        array[..., 1:] += noise
        return Image.fromarray(_to_uint8(array), mode="YCbCr").convert("RGB")

    @staticmethod
    def _sensor_noise(image: Image.Image, read_sigma: float, shot_scale: float, rng: random.Random) -> Image.Image:
        array = np.asarray(image).astype(np.float32) / 255.0
        np_rng = np.random.default_rng(rng.randrange(2**32))
        read = np_rng.normal(0.0, read_sigma / 255.0, size=array.shape)
        shot_sigma = np.sqrt(np.clip(array, 0.0, 1.0)) * (shot_scale / 255.0)
        shot = np_rng.normal(0.0, shot_sigma, size=array.shape)
        return Image.fromarray(_to_uint8((array + read + shot) * 255.0), mode="RGB")

    @staticmethod
    def _poisson_noise(image: Image.Image, rng: random.Random) -> Image.Image:
        array = np.asarray(image).astype(np.float32) / 255.0
        np_rng = np.random.default_rng(rng.randrange(2**32))
        peak = max(8.0, 2.0 ** math.ceil(math.log2(255.0)))
        noised = np_rng.poisson(array * peak) / peak
        return Image.fromarray(_to_uint8(noised * 255.0), mode="RGB")

    @staticmethod
    def _compress(image: Image.Image, fmt: str, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        try:
            image.save(buffer, format=fmt, quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        except OSError:
            return image

    @staticmethod
    def _ringing(image: Image.Image, radius: int, strength: float) -> Image.Image:
        radius = max(1, int(radius))
        gray = np.asarray(image.convert("L")).astype(np.float32)
        blurred = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(radius=max(0.6, radius * 0.5))))
        edge = gray - blurred
        halo = np.zeros_like(edge)

        def shifted(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
            height, width = array.shape
            pad_x = abs(dx)
            pad_y = abs(dy)
            padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
            x0 = pad_x - dx
            y0 = pad_y - dy
            return padded[y0 : y0 + height, x0 : x0 + width]

        for offset in range(1, radius + 1):
            sign = -1.0 if offset % 2 else 0.65
            weight = sign / float(offset)
            halo += weight * (
                shifted(edge, offset, 0)
                + shifted(edge, -offset, 0)
                + shifted(edge, 0, offset)
                + shifted(edge, 0, -offset)
            )
        halo *= 0.25 * float(strength)
        array = np.asarray(image).astype(np.float32) + halo[..., None]
        return Image.fromarray(_to_uint8(array), mode="RGB")

    @staticmethod
    def _color_shift(image: Image.Image, gain: Any, bias: Any, rng: random.Random) -> Image.Image:
        array = np.asarray(image).astype(np.float32)
        gains = np.array([_range_value(rng, gain, 1.0) for _ in range(3)], dtype=np.float32)
        biases = np.array([_range_value(rng, bias, 0.0) for _ in range(3)], dtype=np.float32)
        array = array * gains.reshape(1, 1, 3) + biases.reshape(1, 1, 3)
        return Image.fromarray(_to_uint8(array), mode="RGB")

    @staticmethod
    def _anime_line_change(image: Image.Image, rng: random.Random) -> Image.Image:
        if rng.random() < 0.5:
            filtered = image.filter(ImageFilter.MinFilter(3))
        else:
            filtered = image.filter(ImageFilter.MaxFilter(3))
        return Image.blend(image, filtered, alpha=rng.uniform(0.08, 0.22))

    @staticmethod
    def _banding(image: Image.Image, levels: int) -> Image.Image:
        levels = max(2, int(levels))
        array = np.asarray(image).astype(np.float32)
        step = 255.0 / float(levels - 1)
        array = np.round(array / step) * step
        return Image.fromarray(_to_uint8(array), mode="RGB")
