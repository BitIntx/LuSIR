from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from sr_diffusion.datasets import ManifestImageDataset
from tools.train.train_latent_pretrain import make_dataset


def test_manifest_dataset_returns_hr_lr(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (96, 96), (255, 0, 0)).save(images / "a.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerow({"path": "images/a.png", "domain": "photo", "split": "train"})

    dataset = ManifestImageDataset(
        manifest_path=manifest,
        split="train",
        hr_size=64,
        scale=4,
        domains={"photo": 0, "anime": 1},
        degradation_preset="clean",
        seed=0,
    )
    item = dataset[0]
    assert item["hr"].shape == (3, 64, 64)
    assert item["lr"].shape == (3, 16, 16)
    assert int(item["domain_id"]) == 0


def test_manifest_dataset_applies_train_hflip(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    for x in range(32, 64):
        for y in range(64):
            image.putpixel((x, y), (0, 0, 255))
    image.save(images / "asymmetric.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerow({"path": "images/asymmetric.png", "domain": "photo", "split": "train"})

    dataset = ManifestImageDataset(
        manifest_path=manifest,
        split="train",
        hr_size=64,
        scale=4,
        domains={"photo": 0, "anime": 1},
        degradation_preset="clean",
        seed=0,
        deterministic=True,
        hflip_prob=1.0,
    )

    item = dataset[0]
    assert float(item["hr"][2, 0, 0]) == 1.0
    assert float(item["hr"][0, 0, -1]) == 1.0


def test_stage2_make_dataset_forwards_augmentation_options(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    for x in range(32, 64):
        for y in range(64):
            image.putpixel((x, y), (0, 0, 255))
    image.save(images / "asymmetric.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerow({"path": "images/asymmetric.png", "domain": "photo", "split": "train"})

    dataset = make_dataset(
        {
            "data": {
                "manifest": str(manifest),
                "hr_size": 64,
                "scale": 4,
                "domains": {"photo": 0, "anime": 1},
                "degradation_preset": "clean",
                "hflip_prob": 1.0,
                "texture_crop_retries": 3,
            }
        },
        split="train",
        seed=0,
        deterministic=True,
    )

    item = dataset[0]
    assert dataset.texture_crop_retries == 3
    assert float(item["hr"][2, 0, 0]) == 1.0
    assert float(item["hr"][0, 0, -1]) == 1.0


def test_stage2_make_dataset_supports_fixed_subset_train_probe(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        image = Image.new("RGB", (96, 96), (index * 40, 0, 0))
        for x in range(96):
            for y in range(96):
                image.putpixel((x, y), ((x + index * 40) % 256, y % 256, (x + y) % 256))
        image.save(images / f"sample_{index}.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        for index in range(3):
            writer.writerow({"path": f"images/sample_{index}.png", "domain": "photo", "split": "train"})

    dataset = make_dataset(
        {
            "data": {
                "manifest": str(manifest),
                "hr_size": 64,
                "scale": 4,
                "domains": {"photo": 0, "anime": 1},
                "degradation_preset": "clean",
                "deterministic_train": True,
                "max_items": 2,
                "hflip_prob": 0.5,
                "texture_crop_retries": 3,
            }
        },
        split="train",
        seed=123,
    )

    assert len(dataset) == 2
    assert dataset.deterministic is True
    first = dataset[0]
    second = dataset[0]
    assert first["hr"].equal(second["hr"])
    assert first["lr"].equal(second["lr"])


def test_stage2_make_dataset_applies_data_overrides(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), (120, 80, 40)).save(images / "sample.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        writer.writerow({"path": "images/sample.png", "domain": "photo", "split": "val"})

    dataset = make_dataset(
        {
            "data": {
                "manifest": str(manifest),
                "hr_size": 64,
                "scale": 4,
                "domains": {"photo": 0, "anime": 1},
                "degradation_preset": "mild",
            }
        },
        split="val",
        seed=0,
        deterministic=True,
        data_overrides={"degradation_preset": "benchmark_bicubic"},
    )

    assert dataset.degradation_preset == "benchmark_bicubic"
