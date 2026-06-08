from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


DEFAULT_REPO_ID = "danjacobellis/LSDIR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract an LSDIR subset from Hugging Face parquet shards.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=30000)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=195)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--keep-parquet", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def image_bytes(value: dict[str, object]) -> bytes:
    data = value.get("bytes")
    if not isinstance(data, bytes):
        raise ValueError("LSDIR parquet image column did not contain encoded bytes")
    return data


def existing_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.glob("*.jpg") if path.is_file())


def write_manifest(paths: list[Path], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "domain", "split"])
        writer.writeheader()
        for path in paths:
            writer.writerow({"path": str(path.resolve()), "domain": "photo", "split": "train"})
    print(f"wrote={manifest} rows={len(paths)}", flush=True)


def main() -> None:
    args = parse_args()
    image_dir = args.output_dir / "images"
    shard_dir = args.output_dir / "parquet"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = existing_paths(image_dir)
    count = len(paths)
    print(f"existing={count} target={args.target_count}", flush=True)

    for shard_index in range(args.start_shard, min(195, args.start_shard + args.num_shards)):
        if count >= args.target_count:
            break
        shard_name = f"train-{shard_index:05d}-of-00195.parquet"
        shard_path = shard_dir / shard_name
        if not shard_path.exists():
            url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main/data/{shard_name}"
            print(f"download shard={shard_index}", flush=True)
            download(url, shard_path)

        table = pq.read_table(shard_path, columns=["path", "image", "w", "h"])
        rows = table.to_pylist()
        extracted = 0
        for row_index, row in enumerate(rows):
            if count >= args.target_count:
                break
            if min(int(row["w"]), int(row["h"])) < args.min_size:
                continue
            output_path = image_dir / f"lsdir_{shard_index:05d}_{row_index:05d}.jpg"
            if output_path.exists():
                continue
            with Image.open(io.BytesIO(image_bytes(row["image"]))) as image:
                image.convert("RGB").save(output_path, quality=args.jpeg_quality, subsampling=0)
            paths.append(output_path)
            count += 1
            extracted += 1
        print(f"shard={shard_index} extracted={extracted} total={count}", flush=True)
        if not args.keep_parquet:
            shard_path.unlink(missing_ok=True)
        write_manifest(paths, args.manifest)

    if count < args.target_count:
        raise RuntimeError(f"Only extracted {count} images; target was {args.target_count}")
    shutil.rmtree(shard_dir, ignore_errors=True)
    write_manifest(existing_paths(image_dir)[: args.target_count], args.manifest)


if __name__ == "__main__":
    main()
