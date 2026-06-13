from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


DIV2K_URLS = {
    "hr": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "lr": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
}
SELFEXSR_URL = "https://github.com/jbhuang0604/SelfExSR/archive/refs/heads/master.zip"
CLASSIC_DATASETS = ("Set5", "Set14", "Urban100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download standard x4 SR benchmark pairs and build one manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/scratch/sr-diffusion/benchmarks"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv"),
    )
    parser.add_argument("--datasets", nargs="+", default=["DIV2K", *CLASSIC_DATASETS])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force and zipfile.is_zipfile(destination):
        print(f"exists: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    last_report = [0.0]

    def report(blocks: int, block_size: int, total_size: int) -> None:
        now = time.monotonic()
        downloaded = blocks * block_size
        if downloaded < total_size and now - last_report[0] < 1.0:
            return
        last_report[0] = now
        total = f"/{total_size / 1e9:.2f} GB" if total_size > 0 else ""
        sys.stdout.write(f"\r{destination.name}: {downloaded / 1e9:.2f}{total}")
        sys.stdout.flush()

    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, partial, reporthook=report)
    print()
    if not zipfile.is_zipfile(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid zip: {url}")
    shutil.move(str(partial), destination)


def extract_all(zip_path: Path, destination: Path, marker_name: str) -> None:
    marker = destination / marker_name
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    marker.touch()


def extract_classic(zip_path: Path, destination: Path) -> None:
    marker = destination / ".extracted_selfexsr_x4"
    if marker.exists():
        return
    wanted = tuple(f"SelfExSR-master/data/{name}/image_SRF_4/" for name in CLASSIC_DATASETS)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if member.filename.startswith(wanted) and member.filename.lower().endswith(".png")
        ]
        for member in members:
            archive.extract(member, destination)
    marker.touch()


def build_manifest(output_dir: Path, manifest: Path, datasets: list[str]) -> None:
    rows: list[dict[str, str]] = []
    if "DIV2K" in datasets:
        hr_root = output_dir / "div2k" / "DIV2K_valid_HR"
        lr_root = output_dir / "div2k" / "DIV2K_valid_LR_bicubic" / "X4"
        for hr_path in sorted(hr_root.glob("*.png")):
            sample_id = hr_path.stem
            lr_path = lr_root / f"{sample_id}x4.png"
            if not lr_path.exists():
                raise FileNotFoundError(lr_path)
            rows.append(
                {
                    "dataset": "DIV2K",
                    "id": sample_id,
                    "scale": "4",
                    "hr_path": str(hr_path.resolve()),
                    "lr_path": str(lr_path.resolve()),
                    "source": "official_DIV2K_valid_bicubic_x4",
                }
            )
    classic_root = output_dir / "selfexsr" / "SelfExSR-master" / "data"
    for dataset in CLASSIC_DATASETS:
        if dataset not in datasets:
            continue
        pair_root = classic_root / dataset / "image_SRF_4"
        for hr_path in sorted(pair_root.glob("*_HR.png")):
            sample_id = hr_path.stem.removesuffix("_SRF_4_HR")
            lr_path = pair_root / f"{sample_id}_SRF_4_LR.png"
            if not lr_path.exists():
                raise FileNotFoundError(lr_path)
            rows.append(
                {
                    "dataset": dataset,
                    "id": sample_id,
                    "scale": "4",
                    "hr_path": str(hr_path.resolve()),
                    "lr_path": str(lr_path.resolve()),
                    "source": "SelfExSR_benchmark_pairs",
                }
            )
    if not rows:
        raise ValueError(f"No benchmark rows found under {output_dir}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = {dataset: sum(row["dataset"] == dataset for row in rows) for dataset in datasets}
    print(f"wrote {manifest} rows={len(rows)} counts={counts}", flush=True)


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.datasets) - {"DIV2K", *CLASSIC_DATASETS})
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    downloads = args.output_dir / "downloads"
    if "DIV2K" in args.datasets:
        for kind, url in DIV2K_URLS.items():
            archive = downloads / Path(url).name
            if not args.skip_download:
                download(url, archive, args.force)
            extract_all(archive, args.output_dir / "div2k", f".extracted_{kind}")
    if any(dataset in args.datasets for dataset in CLASSIC_DATASETS):
        archive = downloads / "SelfExSR-master.zip"
        if not args.skip_download:
            download(SELFEXSR_URL, archive, args.force)
        extract_classic(archive, args.output_dir / "selfexsr")
    build_manifest(args.output_dir, args.manifest, args.datasets)


if __name__ == "__main__":
    main()
