from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.train.train_residual_refiner import laplacian_response, metric_highpass, ssim_per_image
from sr_diffusion.utils import get_device


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    pattern: str


class OptionalMetric:
    def __init__(self, name: str, device: torch.device) -> None:
        self.name = name
        self.device = device
        self.kind = "none"
        self.metric: Any = None
        self.available = False
        self.error: str | None = None
        try:
            if name == "lpips":
                import lpips  # type: ignore[import-not-found]

                self.metric = lpips.LPIPS(net="alex").to(device).eval()
                self.kind = "full_reference_minus1_1"
            elif name == "dists":
                try:
                    from DISTS_pytorch import DISTS  # type: ignore[import-not-found]
                except Exception:
                    from dists_pytorch import DISTS  # type: ignore[import-not-found]

                self.metric = DISTS().to(device).eval()
                self.kind = "full_reference_0_1"
            elif name in {"maniqa", "musiq"} or name.startswith("pyiqa:"):
                import pyiqa  # type: ignore[import-not-found]

                metric_name = name.split(":", 1)[1] if name.startswith("pyiqa:") else name
                self.metric = pyiqa.create_metric(metric_name, device=device, as_loss=False)
                self.kind = "no_reference_0_1"
                self.name = metric_name
            else:
                raise ValueError(f"Unknown optional metric: {name}")
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available:
            raise RuntimeError(f"Optional metric {self.name} is unavailable: {self.error}")
        assert self.metric is not None
        prediction = prediction.to(self.device)
        target = target.to(self.device)
        if self.kind == "full_reference_minus1_1":
            return self.metric(prediction.mul(2.0).sub(1.0), target.mul(2.0).sub(1.0)).flatten().float()
        if self.kind == "full_reference_0_1":
            return self.metric(prediction, target).flatten().float()
        if self.kind == "no_reference_0_1":
            return self.metric(prediction).flatten().float()
        raise RuntimeError(f"Unsupported optional metric kind: {self.kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed review-set outputs with distortion/detail/perceptual metrics.")
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate image pattern as name=path. Pattern may include {id}, {source_id}, {preset}, {sample_dir}.",
    )
    parser.add_argument("--include-bicubic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optional-metric", action="append", default=[], help="Optional: lpips, dists, maniqa, musiq, pyiqa:name.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sheet-count", type=int, default=24)
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_candidate_specs(values: list[str]) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be name=pattern, got: {value}")
        name, pattern = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Candidate name is empty in: {value}")
        specs.append(CandidateSpec(name=name, pattern=pattern.strip()))
    return specs


def read_manifest(path: Path, limit: int = 0) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"id", "source_id", "preset", "hr_path", "lr_path", "bicubic_path"}
    if not rows:
        raise ValueError(f"Empty review manifest: {path}")
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"Review manifest must include columns: {sorted(required)}")
    return rows[:limit] if limit > 0 else rows


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def candidate_path(spec: CandidateSpec, manifest_path: Path, row: dict[str, str]) -> Path:
    sample_dir = resolve_manifest_path(manifest_path, row["lr_path"]).parent
    formatted = spec.pattern.format(
        id=row["id"],
        source_id=row["source_id"],
        preset=row["preset"],
        sample_dir=str(sample_dir),
    )
    return resolve_manifest_path(manifest_path, formatted)


def image_to_tensor(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def full_reference_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    mae = (prediction - target).abs().flatten(1).mean(dim=1)
    psnr = psnr_from_mse(mse)
    ssim = ssim_per_image(prediction, target)
    pred_lap = laplacian_response(prediction).abs().flatten(1).mean(dim=1)
    target_lap = laplacian_response(target).abs().flatten(1).mean(dim=1)
    pred_high = metric_highpass(prediction).abs().flatten(1).mean(dim=1)
    target_high = metric_highpass(target).abs().flatten(1).mean(dim=1)
    high_l1 = (metric_highpass(prediction) - metric_highpass(target)).abs().flatten(1).mean(dim=1)
    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "laplacian_abs": pred_lap,
        "laplacian_abs_gt": target_lap,
        "laplacian_ratio": pred_lap / target_lap.clamp_min(1e-8),
        "highpass_abs": pred_high,
        "highpass_abs_gt": target_high,
        "highpass_ratio": pred_high / target_high.clamp_min(1e-8),
        "highpass_l1": high_l1,
    }


def add_label(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 18
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image.convert("RGB"), (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill="black", font=font)
    return canvas


def make_contact_sheet(rows: list[list[tuple[str, Image.Image]]], output_path: Path, gap: int = 6) -> None:
    if not rows:
        return
    labeled_rows = [[add_label(image, label) for label, image in row] for row in rows]
    columns = max(len(row) for row in labeled_rows)
    widths = [0 for _ in range(columns)]
    for row in labeled_rows:
        for col, image in enumerate(row):
            widths[col] = max(widths[col], image.width)
    heights = [max(image.height for image in row) for row in labeled_rows]
    sheet = Image.new("RGB", (sum(widths) + gap * (columns + 1), sum(heights) + gap * (len(rows) + 1)), "white")
    y = gap
    for row, height in zip(labeled_rows, heights, strict=True):
        x = gap
        for col, image in enumerate(row):
            sheet.paste(image, (x, y))
            x += widths[col] + gap
        y += height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def summarize(rows: list[dict[str, Any]], candidate_names: list[str], optional_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_samples": len({row["id"] for row in rows}), "candidates": {}}
    metric_names = [
        "psnr",
        "ssim",
        "mae",
        "laplacian_ratio",
        "highpass_ratio",
        "highpass_l1",
        *optional_names,
    ]
    for candidate in candidate_names:
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        if not candidate_rows:
            continue
        candidate_summary: dict[str, Any] = {"count": len(candidate_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in candidate_rows if row.get(metric) not in (None, "")]
            if values:
                candidate_summary[f"mean_{metric}"] = float(np.mean(values))
                candidate_summary[f"median_{metric}"] = float(np.median(values))
        by_preset: dict[str, Any] = {}
        for preset in sorted({row["preset"] for row in candidate_rows}):
            preset_rows = [row for row in candidate_rows if row["preset"] == preset]
            preset_summary: dict[str, Any] = {"count": len(preset_rows)}
            for metric in metric_names:
                values = [float(row[metric]) for row in preset_rows if row.get(metric) not in (None, "")]
                if values:
                    preset_summary[f"mean_{metric}"] = float(np.mean(values))
                    preset_summary[f"median_{metric}"] = float(np.median(values))
            by_preset[preset] = preset_summary
        candidate_summary["by_preset"] = by_preset
        summary["candidates"][candidate] = candidate_summary
    baseline = "bicubic" if "bicubic" in candidate_names else None
    if baseline:
        by_id_candidate = {(row["id"], row["candidate"]): row for row in rows}
        for candidate in candidate_names:
            if candidate == baseline:
                continue
            deltas = []
            wins = 0
            total = 0
            for row in rows:
                if row["candidate"] != candidate:
                    continue
                base = by_id_candidate.get((row["id"], baseline))
                if base is None:
                    continue
                delta = float(row["psnr"]) - float(base["psnr"])
                deltas.append(delta)
                wins += int(delta > 0.0)
                total += 1
            if deltas:
                summary["candidates"][candidate]["mean_psnr_delta_vs_bicubic"] = float(np.mean(deltas))
                summary["candidates"][candidate]["wins_vs_bicubic"] = wins
                summary["candidates"][candidate]["comparisons_vs_bicubic"] = total
    return summary


def write_html_report(
    output_path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    candidate_names: list[str],
    image_root: Path,
) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_id.setdefault(row["id"], {})[row["candidate"]] = row
        by_id[row["id"]]["_meta"] = row
    blocks: list[str] = []
    for sample_id, grouped in list(by_id.items())[:200]:
        meta = grouped["_meta"]
        cells = [
            f'<td><img src="{html.escape(meta["lr_report_path"])}"><br>LR</td>',
            f'<td><img src="{html.escape(meta["gt_report_path"])}"><br>GT</td>',
        ]
        for candidate in candidate_names:
            row = grouped.get(candidate)
            if row is None:
                continue
            metric = f'PSNR {float(row["psnr"]):.2f}, SSIM {float(row["ssim"]):.3f}, H {float(row["highpass_ratio"]):.2f}x'
            cells.append(
                f'<td><img src="{html.escape(row["candidate_report_path"])}"><br>'
                f'{html.escape(candidate)}<br><small>{html.escape(metric)}</small></td>'
            )
        blocks.append(
            "<section>"
            f"<h3>{html.escape(sample_id)} <span>{html.escape(meta['preset'])} / {html.escape(meta['bucket'])}</span></h3>"
            f"<table><tr>{''.join(cells)}</tr></table>"
            "</section>"
        )
    summary_html = html.escape(json.dumps(summary, indent=2, sort_keys=True))
    output_path.write_text(
        """
<!doctype html>
<meta charset="utf-8">
<title>Fixed Review Set Report</title>
<style>
body { font-family: system-ui, sans-serif; margin: 24px; color: #202124; }
pre { background: #f5f6f7; padding: 12px; overflow-x: auto; }
section { margin: 24px 0; border-top: 1px solid #ddd; padding-top: 16px; }
h3 span { color: #666; font-weight: 400; font-size: 0.9em; }
table { border-collapse: collapse; }
td { vertical-align: top; padding: 6px; text-align: center; font-size: 13px; }
img { width: 192px; height: 192px; object-fit: contain; background: #eee; }
small { color: #555; }
</style>
<h1>Fixed Review Set Report</h1>
<pre>"""
        + summary_html
        + "</pre>\n"
        + "\n".join(blocks)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    review_manifest = args.review_manifest
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(review_manifest, limit=int(args.limit))
    candidate_specs = parse_candidate_specs(args.candidate)
    if args.include_bicubic:
        candidate_specs = [CandidateSpec("bicubic", "{sample_dir}/bicubic.png"), *candidate_specs]
    if not candidate_specs:
        raise ValueError("No candidates to evaluate. Pass --candidate or keep --include-bicubic.")

    optional_metrics = [OptionalMetric(name, device=device) for name in args.optional_metric]
    available_optional = [metric for metric in optional_metrics if metric.available]
    unavailable_optional = {metric.name: metric.error for metric in optional_metrics if not metric.available}

    eval_rows: list[dict[str, Any]] = []
    sheet_rows: list[list[tuple[str, Image.Image]]] = []
    report_image_dir = output_dir / "images"

    for batch_start in range(0, len(rows), int(args.batch_size)):
        batch = rows[batch_start : batch_start + int(args.batch_size)]
        for row in batch:
            gt_path = resolve_manifest_path(review_manifest, row["hr_path"])
            lr_path = resolve_manifest_path(review_manifest, row["lr_path"])
            gt_image = Image.open(gt_path).convert("RGB")
            gt_size = gt_image.size
            gt = image_to_tensor(gt_path).unsqueeze(0).to(device)
            lr_image = Image.open(lr_path).convert("RGB")
            if args.copy_images:
                sample_report_dir = report_image_dir / row["id"]
                sample_report_dir.mkdir(parents=True, exist_ok=True)
                lr_report = sample_report_dir / "lr.png"
                gt_report = sample_report_dir / "gt.png"
                shutil.copyfile(lr_path, lr_report)
                shutil.copyfile(gt_path, gt_report)
            else:
                lr_report = lr_path
                gt_report = gt_path

            sheet_items: list[tuple[str, Image.Image]] = []
            if len(sheet_rows) < int(args.sheet_count):
                sheet_items.append(("LR", lr_image.resize(gt_size, Image.Resampling.NEAREST)))
            for spec in candidate_specs:
                path = candidate_path(spec, review_manifest, row)
                if not path.exists():
                    print(f"missing candidate={spec.name} id={row['id']} path={path}", flush=True)
                    continue
                prediction = image_to_tensor(path, size=gt_size).unsqueeze(0).to(device)
                metrics = full_reference_metrics(prediction, gt)
                record: dict[str, Any] = {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "source_path": row["source_path"],
                    "domain": row["domain"],
                    "bucket": row["bucket"],
                    "preset": row["preset"],
                    "candidate": spec.name,
                    "candidate_path": str(path),
                    "lr_path": str(lr_path),
                    "gt_path": str(gt_path),
                    "lr_report_path": str(Path("images") / row["id"] / "lr.png"),
                    "gt_report_path": str(Path("images") / row["id"] / "gt.png"),
                }
                for key, value in metrics.items():
                    record[key] = float(value[0].detach().cpu())
                for optional in available_optional:
                    with torch.no_grad():
                        record[optional.name] = float(optional(prediction, gt)[0].detach().cpu())
                if args.copy_images:
                    candidate_report = report_image_dir / row["id"] / f"{spec.name}.png"
                    shutil.copyfile(path, candidate_report)
                else:
                    candidate_report = path
                record["candidate_report_path"] = str(Path("images") / row["id"] / f"{spec.name}.png")
                eval_rows.append(record)
                if len(sheet_rows) < int(args.sheet_count):
                    label = f"{spec.name} {record['psnr']:.2f}"
                    sheet_items.append((label, Image.open(path).convert("RGB").resize(gt_size, Image.Resampling.BICUBIC)))
            if len(sheet_rows) < int(args.sheet_count):
                sheet_items.append(("GT", gt_image))
                sheet_rows.append(sheet_items)
        print(f"processed {min(batch_start + len(batch), len(rows))}/{len(rows)}", flush=True)

    if not eval_rows:
        raise ValueError("No candidate images were evaluated.")

    metric_keys = list(eval_rows[0].keys())
    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(eval_rows)

    candidate_names = [spec.name for spec in candidate_specs]
    optional_names = [metric.name for metric in available_optional]
    summary = summarize(eval_rows, candidate_names=candidate_names, optional_names=optional_names)
    summary["optional_metrics"] = {
        "requested": list(args.optional_metric),
        "available": optional_names,
        "unavailable": unavailable_optional,
    }
    summary["review_manifest"] = str(review_manifest)
    summary["metrics_csv"] = str(metrics_csv)
    summary["contact_sheet"] = str(output_dir / "contact_sheet.png")
    summary["html_report"] = str(output_dir / "report.html")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_contact_sheet(sheet_rows, output_dir / "contact_sheet.png")
    write_html_report(output_dir / "report.html", summary, eval_rows, candidate_names, report_image_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
