from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_fixed_review_set_creates_frozen_pairs(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        Image.new("RGB", (96, 96), color).save(images / f"{index}.png")
    source_manifest = tmp_path / "source.csv"
    write_manifest(
        source_manifest,
        [
            {"path": f"images/{index}.png", "domain": "photo", "split": "val"}
            for index in range(3)
        ],
    )
    output_dir = tmp_path / "review"
    subprocess.run(
        [
            sys.executable,
            "tools/eval/build_fixed_review_set.py",
            "--manifest",
            str(source_manifest),
            "--output-dir",
            str(output_dir),
            "--split",
            "val",
            "--count",
            "2",
            "--hr-size",
            "64",
            "--scale",
            "4",
            "--presets",
            "clean",
        ],
        check=True,
    )

    review_manifest = output_dir / "review_manifest.csv"
    assert review_manifest.exists()
    with review_manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for row in rows:
        assert (output_dir / row["hr_path"]).exists()
        assert (output_dir / row["lr_path"]).exists()
        assert (output_dir / row["bicubic_path"]).exists()


def test_eval_fixed_review_outputs_scores_candidate(tmp_path: Path) -> None:
    review_root = tmp_path / "review"
    sample_dir = review_root / "samples" / "0000_texture" / "clean"
    sample_dir.mkdir(parents=True)
    gt = Image.new("RGB", (32, 32), (180, 120, 60))
    lr = Image.new("RGB", (8, 8), (180, 120, 60))
    bad = Image.new("RGB", (32, 32), (0, 0, 0))
    gt.save(review_root / "samples" / "0000_texture" / "gt.png")
    lr.save(sample_dir / "lr.png")
    bad.save(sample_dir / "bicubic.png")

    outputs = tmp_path / "outputs" / "0000_texture_clean"
    outputs.mkdir(parents=True)
    gt.save(outputs / "refined.png")

    review_manifest = review_root / "review_manifest.csv"
    write_manifest(
        review_manifest,
        [
            {
                "id": "0000_texture_clean",
                "source_id": "0000_texture",
                "source_index": "0",
                "source_path": "source.png",
                "domain": "photo",
                "split": "val",
                "bucket": "texture",
                "preset": "clean",
                "seed": "1",
                "hr_path": "samples/0000_texture/gt.png",
                "lr_path": "samples/0000_texture/clean/lr.png",
                "bicubic_path": "samples/0000_texture/clean/bicubic.png",
                "notes": "",
                "laplacian_energy": "0.0",
                "edge_density": "0.0",
                "colorfulness": "0.0",
                "local_contrast": "0.0",
            }
        ],
    )
    output_dir = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            "tools/eval/eval_fixed_review_outputs.py",
            "--review-manifest",
            str(review_manifest),
            "--output-dir",
            str(output_dir),
            "--candidate",
            f"refined={outputs.parent}/{{id}}/refined.png",
            "--batch-size",
            "1",
            "--sheet-count",
            "1",
        ],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"]["refined"]["mean_psnr"] > summary["candidates"]["bicubic"]["mean_psnr"]
    assert (output_dir / "contact_sheet.png").exists()
    assert (output_dir / "report.html").exists()
