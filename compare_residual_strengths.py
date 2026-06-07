from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eval_residual_refiner import resolve_checkpoint
from train_residual_refiner import (
    BoundedResidualRefiner,
    apply_residual_strength,
    denormalize,
    load_autoencoder,
    load_condition_encoder,
    make_eval_loader,
    make_grid,
    normalize_image,
    tensor_to_pil,
)
from sr_diffusion.utils import autocast_context, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual residual-strength comparison report.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["photo_detail_mix", "mild", "photo_v2", "photo_v3_noise_mix"],
    )
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    return parser.parse_args()


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).pow(2).flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def mode_name(strength: float) -> str:
    names = {0.5: "Conservative", 0.75: "Balanced", 1.0: "Full"}
    return names.get(float(strength), f"Strength {strength:.2f}")


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


@torch.no_grad()
def compare_preset(
    *,
    config: dict[str, Any],
    preset: str,
    strengths: list[float],
    sample_count: int,
    model: BoundedResidualRefiner,
    vae: torch.nn.Module,
    condition_encoder: torch.nn.Module,
    device: torch.device,
    dtype_name: str,
    output_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    preset_config = {**config, "data": {**config["data"], "degradation_preset": preset}}
    preset_config["eval"] = {
        **config.get("eval", {}),
        "split": "val",
        "limit": sample_count,
        "batch_size": min(sample_count, int(config.get("eval", {}).get("batch_size", sample_count))),
        "num_workers": int(config.get("eval", {}).get("num_workers", 4)),
        "sample_count": sample_count,
    }
    dataloader = make_eval_loader(preset_config, seed=seed, device=device)
    preset_dir = output_dir / preset
    preset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[list[tuple[str, Any]]] = []
    records: list[dict[str, Any]] = []

    for batch in dataloader:
        hr = batch["hr"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        domain_id = batch["domain_id"].to(device, non_blocking=True)
        paths = batch.get("path", [""] * int(hr.shape[0]))
        lr_input = normalize_image(lr)
        with autocast_context(device, dtype_name):
            condition = condition_encoder(lr_input, domain_id)
            _, residual, _ = model(condition, lr_input, domain_id)
            decoded_condition = denormalize(vae.decode(condition)).float()
            decoded_by_strength = {
                strength: denormalize(vae.decode(apply_residual_strength(condition, residual, strength))).float()
                for strength in strengths
            }
        bicubic = F.interpolate(lr.float(), size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        lr_nearest = F.interpolate(lr.float(), size=hr.shape[-2:], mode="nearest").clamp(0.0, 1.0)
        condition_psnr = psnr_per_image(decoded_condition, hr)
        psnr_by_strength = {strength: psnr_per_image(decoded, hr) for strength, decoded in decoded_by_strength.items()}

        for item_index in range(int(hr.shape[0])):
            sample_index = len(records)
            if sample_index >= sample_count:
                break
            sample_dir = preset_dir / f"sample_{sample_index + 1:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            images = {
                "lr": tensor_to_pil(lr_nearest[item_index]),
                "bicubic": tensor_to_pil(bicubic[item_index]),
                "condition": tensor_to_pil(decoded_condition[item_index]),
                "gt": tensor_to_pil(hr[item_index]),
            }
            for strength, decoded in decoded_by_strength.items():
                images[f"strength_{strength:.2f}"] = tensor_to_pil(decoded[item_index])
            for name, image in images.items():
                image.save(sample_dir / f"{name}.png")

            condition_value = float(condition_psnr[item_index].cpu())
            deltas = {
                strength: float(psnr_by_strength[strength][item_index].cpu()) - condition_value for strength in strengths
            }
            row: list[tuple[str, Any]] = [
                ("LR", images["lr"]),
                ("Bicubic", images["bicubic"]),
                (f"Condition {condition_value:.2f}", images["condition"]),
            ]
            row.extend(
                (
                    f"{mode_name(strength)} {deltas[strength]:+.3f}",
                    images[f"strength_{strength:.2f}"],
                )
                for strength in strengths
            )
            row.append(("GT", images["gt"]))
            rows.append(row)
            records.append(
                {
                    "sample": sample_index + 1,
                    "source": str(paths[item_index]),
                    "condition_psnr": condition_value,
                    "deltas": {f"{strength:.2f}": deltas[strength] for strength in strengths},
                }
            )
        if len(records) >= sample_count:
            break

    make_grid(rows, preset_dir / "comparison_grid.png", gap=8)
    (preset_dir / "samples.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def write_html(
    output_dir: Path,
    preset_records: dict[str, list[dict[str, Any]]],
    strengths: list[float],
) -> None:
    columns = [
        ("lr", "LR"),
        ("bicubic", "Bicubic"),
        ("condition", "Condition"),
        *[(f"strength_{strength:.2f}", mode_name(strength)) for strength in strengths],
        ("gt", "GT"),
    ]
    summary_rows = []
    for preset, records in preset_records.items():
        cells = []
        for strength in strengths:
            deltas = [float(record["deltas"][f"{strength:.2f}"]) for record in records]
            mean_delta = sum(deltas) / max(len(deltas), 1)
            wins = sum(delta > 0.0 for delta in deltas)
            cells.append(f"<td>{mean_delta:+.3f} dB<br><small>{wins}/{len(deltas)} wins</small></td>")
        summary_rows.append(f"<tr><th>{html.escape(preset)}</th>{''.join(cells)}</tr>")
    summary_header = "".join(f"<th>{html.escape(mode_name(strength))}</th>" for strength in strengths)
    summary_table = (
        '<div class="summary"><table><thead><tr><th>Preset</th>'
        f"{summary_header}</tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>"
    )
    sections = []
    for preset, records in preset_records.items():
        rows = []
        for record in records:
            sample = int(record["sample"])
            cells = []
            for key, label in columns:
                caption = label
                if key == "condition":
                    caption += f"<small>PSNR {record['condition_psnr']:.2f}</small>"
                elif key.startswith("strength_"):
                    strength = key.removeprefix("strength_")
                    caption += f"<small>vs condition {record['deltas'][strength]:+.3f} dB</small>"
                image_path = f"{preset}/sample_{sample:02d}/{key}.png"
                cells.append(
                    f'<figure><a href="{html.escape(image_path)}" target="_blank">'
                    f'<img src="{html.escape(image_path)}" loading="lazy"></a><figcaption>{caption}</figcaption></figure>'
                )
            rows.append(
                f'<article><div class="sample-title">Sample {sample:02d} '
                f'<span>{html.escape(Path(record["source"]).name)}</span></div>'
                f'<div class="comparison">{"".join(cells)}</div></article>'
            )
        sections.append(
            f'<section><h2>{html.escape(preset)}</h2>'
            f'<p><a href="{html.escape(preset)}/comparison_grid.png" target="_blank">Open one-page grid</a></p>'
            f'{"".join(rows)}</section>'
        )

    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Residual Refiner Strength Comparison</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #151515; background: #f3f4f6; }}
header {{ padding: 24px 28px; background: white; border-bottom: 1px solid #d7d9dd; position: sticky; top: 0; z-index: 2; }}
h1 {{ margin: 0 0 8px; font-size: 24px; }} h2 {{ margin: 28px 0 4px; }}
p {{ margin: 6px 0; color: #4d5157; }}
main {{ padding: 0 24px 40px; }}
.summary {{ margin: 18px 0 26px; overflow-x: auto; }}
table {{ border-collapse: collapse; min-width: 640px; background: white; }}
th, td {{ padding: 10px 14px; border: 1px solid #d7d9dd; text-align: left; white-space: nowrap; }}
thead th {{ background: #e9edf2; }}
article {{ margin: 14px 0; background: white; border: 1px solid #d7d9dd; border-radius: 6px; overflow: hidden; }}
.sample-title {{ padding: 9px 12px; font-weight: 700; border-bottom: 1px solid #e2e4e8; }}
.sample-title span {{ margin-left: 8px; color: #696e76; font-weight: 400; }}
.comparison {{ display: grid; grid-template-columns: repeat({len(columns)}, minmax(180px, 1fr)); gap: 1px; background: #d7d9dd; overflow-x: auto; }}
figure {{ margin: 0; padding: 8px; min-width: 180px; background: white; }}
img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: contain; background: #eee; }}
figcaption {{ margin-top: 6px; font-size: 13px; font-weight: 700; }}
small {{ display: block; margin-top: 2px; color: #5a6068; font-weight: 400; }}
a {{ color: #1769aa; }}
</style>
</head>
<body>
<header>
<h1>Residual Refiner Strength Comparison</h1>
<p>같은 샘플을 왼쪽에서 오른쪽으로 비교하세요. 이미지를 클릭하면 원본 크기로 열립니다.</p>
<p>GT와 실제로 가까워지는지, 가짜 질감·halo·흰 점·형태 변화가 생기는지에 집중하세요.</p>
</header>
<main>
<h2>8-sample quick summary</h2>
<p>아래 수치는 빠른 시각 검토 샘플의 PSNR 변화이며 전체 val100 결과를 대체하지 않습니다.</p>
{summary_table}
{"".join(sections)}
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["eval"] = {
        **config.get("eval", {}),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
    }
    seed_everything(args.seed)
    device = get_device(args.device)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_autoencoder(config, device)
    condition_encoder = load_condition_encoder(config, device)
    model = BoundedResidualRefiner.from_config(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    strengths = sorted({float(strength) for strength in args.strengths})
    preset_records = {
        preset: compare_preset(
            config=config,
            preset=preset,
            strengths=strengths,
            sample_count=int(args.sample_count),
            model=model,
            vae=vae,
            condition_encoder=condition_encoder,
            device=device,
            dtype_name=dtype_name,
            output_dir=args.output_dir,
            seed=int(args.seed),
        )
        for preset in args.presets
    }
    write_html(args.output_dir, preset_records, strengths)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "presets": args.presets,
        "strengths": strengths,
        "sample_count": int(args.sample_count),
        "report": str(args.output_dir / "index.html"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
