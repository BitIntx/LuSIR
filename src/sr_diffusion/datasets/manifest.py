from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from sr_diffusion.degradations import DegradationPipeline


@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    domain: str
    split: str


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _prepare_crop_image(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) < size:
        scale = size / float(min(width, height))
        new_size = (max(size, round(width * scale)), max(size, round(height * scale)))
        image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
    return image


def _random_crop_box(width: int, height: int, size: int, rng: random.Random, random_crop: bool) -> tuple[int, int, int, int]:
    max_x = width - size
    max_y = height - size
    if random_crop and (max_x > 0 or max_y > 0):
        left = rng.randint(0, max_x) if max_x > 0 else 0
        top = rng.randint(0, max_y) if max_y > 0 else 0
    else:
        left = max_x // 2
        top = max_y // 2
    return left, top, left + size, top + size


def _texture_score(image: Image.Image, downsample: int) -> float:
    if downsample > 0 and max(image.size) > downsample:
        image = image.resize((downsample, downsample), resample=Image.Resampling.BILINEAR)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    return float(grad_x + grad_y)


def crop_square(
    image: Image.Image,
    size: int,
    rng: random.Random,
    random_crop: bool,
    texture_crop_retries: int = 1,
    texture_crop_downsample: int = 128,
) -> Image.Image:
    image = _prepare_crop_image(image, size)
    width, height = image.size
    retries = max(1, int(texture_crop_retries))
    if not random_crop or retries == 1:
        return image.crop(_random_crop_box(width, height, size, rng, random_crop=random_crop))

    best_crop: Image.Image | None = None
    best_score = float("-inf")
    for _ in range(retries):
        crop = image.crop(_random_crop_box(width, height, size, rng, random_crop=True))
        score = _texture_score(crop, downsample=int(texture_crop_downsample))
        if score > best_score:
            best_score = score
            best_crop = crop
    if best_crop is None:
        return image.crop(_random_crop_box(width, height, size, rng, random_crop=random_crop))
    return best_crop


def _range_value(rng: random.Random, value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if len(value) != 2:
        raise ValueError(f"Expected scalar or [min, max], got {value}")
    return rng.uniform(float(value[0]), float(value[1]))


def augment_hr_image(
    image: Image.Image,
    rng: random.Random,
    hflip_prob: float,
    color_jitter_prob: float,
    color_jitter: Any,
) -> Image.Image:
    if hflip_prob > 0.0 and rng.random() < hflip_prob:
        image = ImageOps.mirror(image)
    if color_jitter_prob > 0.0 and rng.random() < color_jitter_prob:
        color_factor = _range_value(rng, color_jitter, 1.0)
        contrast_factor = _range_value(rng, color_jitter, 1.0)
        brightness_factor = _range_value(rng, color_jitter, 1.0)
        image = ImageEnhance.Color(image).enhance(color_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    return image


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        hr_size: int,
        scale: int,
        domains: dict[str, int],
        degradation_preset: str = "mild",
        seed: int = 0,
        deterministic: bool | None = None,
        hflip_prob: float = 0.0,
        texture_crop_retries: int = 1,
        texture_crop_downsample: int = 128,
        hr_color_jitter_prob: float = 0.0,
        hr_color_jitter: Any = (0.97, 1.03),
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.hr_size = int(hr_size)
        self.scale = int(scale)
        self.lr_size = self.hr_size // self.scale
        self.domains = domains
        self.seed = int(seed)
        self.deterministic = split != "train" if deterministic is None else deterministic
        self.random_crop = split == "train"
        self.hflip_prob = float(hflip_prob)
        self.texture_crop_retries = max(1, int(texture_crop_retries))
        self.texture_crop_downsample = int(texture_crop_downsample)
        self.hr_color_jitter_prob = float(hr_color_jitter_prob)
        self.hr_color_jitter = hr_color_jitter
        self.entries = self._load_entries()
        self.degradation_preset = degradation_preset
        self.default_pipeline = DegradationPipeline.from_preset(degradation_preset, scale=scale)
        self.anime_pipeline = DegradationPipeline.from_preset("anime", scale=scale)

        if self.hr_size % self.scale != 0:
            raise ValueError(f"hr_size must be divisible by scale: {self.hr_size}, {self.scale}")
        if not self.entries:
            raise ValueError(f"No entries for split '{split}' in {self.manifest_path}")

    @classmethod
    def from_config(cls, data_config: dict[str, Any], seed: int = 0) -> "ManifestImageDataset":
        return cls(
            manifest_path=data_config["manifest"],
            split=data_config.get("split", "train"),
            hr_size=data_config.get("hr_size", 512),
            scale=data_config.get("scale", 4),
            domains=data_config.get("domains", {"photo": 0, "anime": 1}),
            degradation_preset=data_config.get("degradation_preset", "mild"),
            seed=seed,
            hflip_prob=data_config.get("hflip_prob", 0.0),
            texture_crop_retries=data_config.get("texture_crop_retries", 1),
            texture_crop_downsample=data_config.get("texture_crop_downsample", 128),
            hr_color_jitter_prob=data_config.get("hr_color_jitter_prob", 0.0),
            hr_color_jitter=data_config.get("hr_color_jitter", (0.97, 1.03)),
        )

    def _load_entries(self) -> list[ManifestEntry]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        base_dir = self.manifest_path.parent
        entries: list[ManifestEntry] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"path", "domain", "split"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest must contain columns: {sorted(required)}")
            for row in reader:
                if row["split"] != self.split:
                    continue
                image_path = Path(row["path"])
                if not image_path.is_absolute():
                    image_path = base_dir / image_path
                domain = row["domain"]
                if domain not in self.domains:
                    raise ValueError(f"Unknown domain '{domain}' for {image_path}")
                entries.append(ManifestEntry(path=image_path, domain=domain, split=row["split"]))
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        rng = random.Random(self.seed + index) if self.deterministic else random
        image = Image.open(entry.path).convert("RGB")
        use_train_augment = self.split == "train"
        hr = crop_square(
            image,
            self.hr_size,
            rng=rng,
            random_crop=self.random_crop,
            texture_crop_retries=self.texture_crop_retries if use_train_augment else 1,
            texture_crop_downsample=self.texture_crop_downsample,
        )
        if use_train_augment:
            hr = augment_hr_image(
                hr,
                rng=rng,
                hflip_prob=self.hflip_prob,
                color_jitter_prob=self.hr_color_jitter_prob,
                color_jitter=self.hr_color_jitter,
            )
        pipeline = self.anime_pipeline if self.degradation_preset == "domain" and entry.domain == "anime" else self.default_pipeline
        lr = pipeline.apply(hr, rng=rng, out_size=self.lr_size)

        return {
            "hr": pil_to_tensor(hr),
            "lr": pil_to_tensor(lr),
            "domain_id": torch.tensor(self.domains[entry.domain], dtype=torch.long),
            "domain": entry.domain,
            "path": str(entry.path),
            "index": torch.tensor(index, dtype=torch.long),
        }
