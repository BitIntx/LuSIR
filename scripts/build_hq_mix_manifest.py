from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat high-quality training rows in an image manifest.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hq-repeat", type=int, default=30)
    parser.add_argument("--hq-markers", nargs="+", default=["div2k", "flickr2k"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hq_repeat < 1:
        raise ValueError("--hq-repeat must be at least 1")

    markers = tuple(marker.lower() for marker in args.hq_markers)
    counts: Counter[str] = Counter()
    output_rows: list[dict[str, str]] = []
    with args.input.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Manifest has no header: {args.input}")
        for row in reader:
            is_hq = row.get("split") == "train" and any(marker in row.get("path", "").lower() for marker in markers)
            repeats = args.hq_repeat if is_hq else 1
            output_rows.extend(dict(row) for _ in range(repeats))
            counts["hq_source_rows" if is_hq else "other_source_rows"] += 1
            counts["hq_output_rows" if is_hq else "other_output_rows"] += repeats

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    train_rows = sum(1 for row in output_rows if row.get("split") == "train")
    hq_train_rows = counts["hq_output_rows"]
    print(f"wrote={args.output}")
    print(f"rows={len(output_rows)} train_rows={train_rows}")
    print(f"hq_train_rows={hq_train_rows} hq_train_fraction={hq_train_rows / max(1, train_rows):.4f}")


if __name__ == "__main__":
    main()
