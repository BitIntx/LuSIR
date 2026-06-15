from __future__ import annotations

import argparse
import base64
import html
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "jwheo/LuSIR"
OUTPUT_ROOT = ROOT / "outputs" / "colab_webui"


MODEL_OPTIONS = {
    "Best T4 default - Stage 2 Guarded Detail v2": "stage2_guarded_detail_v2",
    "Conservative quality - Residual Refiner v2": "residual_refiner_v2",
    "Latest masked detail research - Detail Branch v2": "detail_branch_v2_masked",
    "Previous detail research - Detail Branch v1d": "detail_branch_v1d",
    "Sharper diffusion comparison - XL Edge": "photo100k_xl_edge_b16",
    "Smaller diffusion comparison - Stage 4 v2": "photo100k_v2_stage4",
    "Mild diffusion comparison - Stage 4": "photo100k_stage4",
}

TTA_OPTIONS = {
    "Off": ("identity",),
    "Horizontal flip x2": ("identity", "hflip"),
    "Full x8 self-ensemble": ("identity", "hflip", "vflip", "rot180", "rot90", "rot270", "transpose", "transverse"),
}

COMMON_FILES = [
    "LICENSE",
    "CHECKPOINT_LICENSE.md",
    "checkpoints/stage1_autoencoder_best_eval_recon.pt",
]

VARIANTS: dict[str, dict[str, Any]] = {
    "stage2_guarded_detail_v2": {
        "runner": "stage2",
        "config": "configs/hf/latent_pretrain_photo130k_lsdir_dual_detail_guarded_v2.yaml",
        "files": [
            "checkpoints/stage2_photo130k_lsdir_dual_detail_guarded_v2_best10000.pt",
        ],
        "note": "T4-friendly default: guarded-detail Stage 2 step 10000 -> Stage 1 decoder.",
    },
    "residual_refiner_v2": {
        "runner": "residual_refiner",
        "config": "configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml",
        "files": [
            "checkpoints/stage2_photo100k_v3_noise_xl_b64_step_0072000.pt",
            "checkpoints/residual_refiner_stage2_xl_photo_detail_v2_best39000.pt",
        ],
        "note": "Deterministic public default: Stage 2 XL -> residual refiner v2 -> Stage 1 decoder.",
    },
    "detail_branch_v1d": {
        "runner": "detail_branch",
        "config": "configs/hf/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml",
        "files": [
            "checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt",
            "checkpoints/detail_branch_v1d_deep3m_photo130k_lsdir_best99500.pt",
        ],
        "note": "Latest deterministic detail research path: dual-context Stage 2 -> Stage 1 decoder -> detail branch v1d.",
    },
    "detail_branch_v2_masked": {
        "runner": "detail_branch",
        "config": "configs/hf/detail_branch_v2_masked_photo130k_lsdir.yaml",
        "files": [
            "checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt",
            "checkpoints/detail_mask_predictor_v1_best3250.pt",
            "checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt",
        ],
        "note": "Latest research path: learned detail-need mask gates the deterministic v2 detail branch.",
    },
    "photo100k_xl_edge_b16": {
        "runner": "diffusion",
        "config": "configs/hf/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml",
        "files": [
            "checkpoints/stage2_photo100k_v3_noise_xl_b64_step_0072000.pt",
            "checkpoints/stage4_photo100k_xl_edge_b16_best_eval_condition_decoded.pt",
        ],
        "note": "Aggressive XL Stage 4 diffusion comparison. Slower than the default deterministic path.",
    },
    "photo100k_v2_stage4": {
        "runner": "diffusion",
        "config": "configs/hf/diffusion_photo100k_stage4_condition_v2.yaml",
        "files": [
            "checkpoints/stage2_photo100k_v2_b64_best_eval_latent.pt",
            "checkpoints/stage4_photo100k_condition_v2_b32_best_eval_condition_decoded.pt",
        ],
        "note": "Smaller Stage 4 condition-start diffusion comparison.",
    },
    "photo100k_stage4": {
        "runner": "diffusion",
        "config": "configs/hf/diffusion_photo100k_stage4_condition.yaml",
        "files": [
            "checkpoints/stage2_photo100k_b64_best_eval_latent.pt",
            "checkpoints/stage4_photo100k_condition_b32_best_eval_condition_decoded.pt",
        ],
        "note": "Milder Stage 4 condition-start diffusion comparison.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the LuSIR Colab Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--repo-id", default=REPO_ID)
    return parser.parse_args()


def gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def ensure_gpu() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU runtime is required. In Colab, use Runtime -> Change runtime type -> GPU.")
    return f"{torch.cuda.get_device_name(0)} ({gpu_total_gb():.1f} GB)"


def ensure_model(variant: str, repo_id: str) -> None:
    selected = VARIANTS[variant]
    files_to_download = [*COMMON_FILES, *selected["files"]]
    missing = [filename for filename in files_to_download if not (ROOT / filename).exists()]
    if missing:
        cmd = [sys.executable, "scripts/download_hf_checkpoints.py", "--repo-id", repo_id]
        for filename in files_to_download:
            cmd += ["--file", filename]
        subprocess.run(cmd, cwd=ROOT, check=True)
    still_missing = [filename for filename in files_to_download if not (ROOT / filename).exists()]
    if still_missing:
        raise FileNotFoundError(f"Missing downloaded files: {still_missing}")


def save_input(image: Image.Image, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "uploaded_input.png"
    image.convert("RGB").save(input_path)
    return input_path


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def image_path(output_dir: Path, filename: str) -> Path | None:
    path = output_dir / filename
    return path if path.exists() else None


def image_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def make_lr_nearest(input_lr: Path, target: Path, output_dir: Path) -> Path:
    target_image = Image.open(target).convert("RGB")
    lr_image = Image.open(input_lr).convert("RGB")
    nearest = lr_image.resize(target_image.size, Image.Resampling.NEAREST)
    output_path = output_dir / "input_lr_nearest.png"
    nearest.save(output_path)
    return output_path


def make_compare_slider(before_path: Path, after_path: Path, before_label: str, after_label: str) -> str:
    before = image_data_url(before_path)
    after = image_data_url(after_path)
    slider_id = f"compare_{uuid.uuid4().hex}"
    before_label = html.escape(before_label)
    after_label = html.escape(after_label)
    return f"""
<style>
  #{slider_id} {{
    --pos: 50%;
    width: 100%;
    max-width: 960px;
    margin: 0 auto;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #{slider_id} .stage {{
    position: relative;
    width: 100%;
    overflow: hidden;
    background: #111;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
  }}
  #{slider_id} img {{
    display: block;
    width: 100%;
    height: auto;
    user-select: none;
    pointer-events: none;
  }}
  #{slider_id} .after {{
    position: absolute;
    inset: 0;
    clip-path: inset(0 calc(100% - var(--pos)) 0 0);
  }}
  #{slider_id} .handle {{
    position: absolute;
    top: 0;
    bottom: 0;
    left: var(--pos);
    width: 2px;
    background: white;
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35);
  }}
  #{slider_id} .labels {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin: 8px 0 6px;
    color: #222;
    font-size: 13px;
  }}
  #{slider_id} input[type="range"] {{
    width: 100%;
  }}
</style>
<div id="{slider_id}">
  <div class="labels"><span>{before_label}</span><span>{after_label}</span></div>
  <div class="stage">
    <img src="{before}" alt="{before_label}">
    <div class="after"><img src="{after}" alt="{after_label}"></div>
    <div class="handle"></div>
  </div>
  <input type="range" min="0" max="100" value="50"
    oninput="document.getElementById('{slider_id}').style.setProperty('--pos', this.value + '%')">
</div>
"""


def collect_gallery(output_dir: Path, result_path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for filename, label in [
        ("input_lr.png", "Input LR"),
        ("bicubic.png", "Bicubic x4"),
        ("condition.png", "Stage 2 condition"),
        (result_path.name, "SR output"),
        ("gt_hr.png", "GT HR"),
    ]:
        path = output_dir / filename
        if path.exists():
            entries.append((str(path), label))
    return entries


def apply_tta_transform(image: Image.Image, transform: str) -> Image.Image:
    if transform == "identity":
        return image.copy()
    if transform == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if transform == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if transform == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)
    if transform == "rot90":
        return image.transpose(Image.Transpose.ROTATE_90)
    if transform == "rot270":
        return image.transpose(Image.Transpose.ROTATE_270)
    if transform == "transpose":
        return image.transpose(Image.Transpose.TRANSPOSE)
    if transform == "transverse":
        return image.transpose(Image.Transpose.TRANSVERSE)
    raise ValueError(f"Unknown TTA transform: {transform}")


def invert_tta_transform(image: Image.Image, transform: str) -> Image.Image:
    if transform == "rot90":
        return image.transpose(Image.Transpose.ROTATE_270)
    if transform == "rot270":
        return image.transpose(Image.Transpose.ROTATE_90)
    return apply_tta_transform(image, transform)


def average_tta_images(images: list[Image.Image]) -> Image.Image:
    arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) for image in images]
    reference_shape = arrays[0].shape
    if any(array.shape != reference_shape for array in arrays):
        shapes = [array.shape for array in arrays]
        raise ValueError(f"TTA outputs have mismatched shapes: {shapes}")
    merged = np.clip(np.round(np.stack(arrays, axis=0).mean(axis=0)), 0, 255).astype(np.uint8)
    return Image.fromarray(merged, mode="RGB")


def copy_identity_outputs(identity_dir: Path, output_dir: Path) -> None:
    for filename in [
        "input_lr.png",
        "bicubic.png",
        "condition.png",
        "base.png",
        "detail.png",
        "refined.png",
        "sr.png",
        "sr_00.png",
        "gt_hr.png",
    ]:
        source = identity_dir / filename
        if source.exists():
            shutil.copyfile(source, output_dir / filename)


def build_command(
    *,
    variant: str,
    input_mode: str,
    input_path: Path,
    output_dir: Path,
    residual_strength: float,
    use_tiling: bool,
    tile_overlap: int,
    tile_batch_size: int,
    steps: int,
    seed: int,
) -> tuple[list[str], str, bool]:
    selected = VARIANTS[variant]
    runner = selected["runner"]
    uses_strength = runner in {"residual_refiner", "detail_branch"}
    runner_script = {
        "stage2": "tools/infer/infer_stage2.py",
        "residual_refiner": "tools/infer/infer_residual_refiner.py",
        "detail_branch": "tools/infer/infer_detail_branch.py",
        "diffusion": "tools/infer/infer_diffusion.py",
    }[runner]
    result_file = {
        "stage2": "sr.png",
        "residual_refiner": "refined.png",
        "detail_branch": "detail.png",
        "diffusion": "sr_00.png",
    }[runner]
    input_flag = "--input-lr" if input_mode == "Low-resolution image to upscale" else "--input-hr"
    cmd = [
        sys.executable,
        "-u",
        runner_script,
        "--config",
        selected["config"],
        input_flag,
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(int(seed)),
    ]
    if runner == "residual_refiner":
        cmd += ["--residual-strength", f"{float(residual_strength):.3f}"]
    elif runner == "detail_branch":
        cmd += ["--detail-strength", f"{float(residual_strength):.3f}"]
    elif runner == "diffusion":
        cmd += ["--steps", str(int(steps)), "--progress-every", "4"]
    if input_flag == "--input-lr" and use_tiling:
        cmd += [
            "--tile",
            "--tile-overlap",
            str(int(tile_overlap)),
            "--tile-batch-size",
            str(int(tile_batch_size)),
        ]
    return cmd, result_file, uses_strength


def run_tta_sr(
    *,
    variant: str,
    input_path: Path,
    output_dir: Path,
    tta_mode: str,
    residual_strength: float,
    use_tiling: bool,
    tile_overlap: int,
    tile_batch_size: int,
    steps: int,
    seed: int,
) -> tuple[Path, bool]:
    transforms = TTA_OPTIONS[tta_mode]
    input_image = Image.open(input_path).convert("RGB")
    restored_outputs: list[Image.Image] = []
    uses_strength = False
    identity_dir: Path | None = None
    result_file = ""
    for index, transform in enumerate(transforms):
        tta_dir = output_dir / f"tta_{index:02d}_{transform}"
        tta_dir.mkdir(parents=True, exist_ok=True)
        tta_input = output_dir / f"tta_input_{index:02d}_{transform}.png"
        apply_tta_transform(input_image, transform).save(tta_input)
        cmd, result_file, uses_strength = build_command(
            variant=variant,
            input_mode="Low-resolution image to upscale",
            input_path=tta_input,
            output_dir=tta_dir,
            residual_strength=residual_strength,
            use_tiling=use_tiling,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            steps=steps,
            seed=seed,
        )
        run_command(cmd)
        transformed_result = tta_dir / result_file
        if not transformed_result.exists():
            raise FileNotFoundError(f"Expected TTA output was not created: {transformed_result}")
        restored_outputs.append(invert_tta_transform(Image.open(transformed_result).convert("RGB"), transform))
        if transform == "identity":
            identity_dir = tta_dir
    if identity_dir is not None:
        copy_identity_outputs(identity_dir, output_dir)
    result_path = output_dir / "tta_sr_output.png"
    average_tta_images(restored_outputs).save(result_path)
    return result_path, uses_strength


def run_sr(
    image: Image.Image | None,
    model_preset: str,
    input_mode: str,
    residual_strength: float,
    comparison_left: str,
    tta_mode: str,
    use_tiling: bool,
    tile_overlap: int,
    tile_batch_size: int,
    steps: int,
    seed: int,
) -> tuple[str, str, list[tuple[str, str]], str | None]:
    if image is None:
        raise ValueError("Upload an input image first.")
    gpu_name = ensure_gpu()
    variant = MODEL_OPTIONS[model_preset]
    selected = VARIANTS[variant]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{variant}_{timestamp}"
    input_path = save_input(image, output_dir)
    ensure_model(variant, REPO_ID)
    if tta_mode != "Off":
        if input_mode != "Low-resolution image to upscale":
            raise ValueError("TTA is available for low-resolution image uploads only.")
        result_path, uses_strength = run_tta_sr(
            variant=variant,
            input_path=input_path,
            output_dir=output_dir,
            tta_mode=tta_mode,
            residual_strength=residual_strength,
            use_tiling=use_tiling,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            steps=steps,
            seed=seed,
        )
    else:
        cmd, result_file, uses_strength = build_command(
            variant=variant,
            input_mode=input_mode,
            input_path=input_path,
            output_dir=output_dir,
            residual_strength=residual_strength,
            use_tiling=use_tiling,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            steps=steps,
            seed=seed,
        )
        run_command(cmd)
        result_path = output_dir / result_file
        if not result_path.exists():
            raise FileNotFoundError(f"Expected output was not created: {result_path}")
    before_path: Path
    before_label: str
    if comparison_left == "Stage 2 condition" and (output_dir / "condition.png").exists():
        before_path = output_dir / "condition.png"
        before_label = "Before: Stage 2 condition"
    elif comparison_left == "Input LR nearest":
        before_path = make_lr_nearest(output_dir / "input_lr.png", result_path, output_dir)
        before_label = "Before: input LR nearest"
    else:
        before_path = image_path(output_dir, "bicubic.png") or make_lr_nearest(output_dir / "input_lr.png", result_path, output_dir)
        before_label = "Before: bicubic x4"
    slider_html = make_compare_slider(before_path, result_path, before_label, "After: SR output")
    shutil.copyfile(result_path, output_dir / "sr_output.png")
    details = [
        f"GPU: {gpu_name}",
        f"model: {model_preset}",
        f"path: {selected['note']}",
        f"steps: {int(steps) if selected['runner'] == 'diffusion' else 'deterministic'}",
        f"correction strength: {float(residual_strength):.2f}" if uses_strength else "correction strength: n/a",
        f"TTA: {tta_mode}",
        f"tiling: {bool(use_tiling)}",
        f"tile batch size: {int(tile_batch_size)}",
        f"output: {output_dir}",
    ]
    return "\n".join(details), slider_html, collect_gallery(output_dir, result_path), str(output_dir / "sr_output.png")


def build_app(repo_id: str) -> Any:
    import gradio as gr

    global REPO_ID
    REPO_ID = repo_id
    with gr.Blocks(title="LuSIR x4 WebUI", css="footer {visibility: hidden}") as demo:
        gr.Markdown(
            """
# LuSIR x4 WebUI
Upload an image, run x4 SR, then use the slider to compare before and after.

Default path: **LR -> guarded-detail Stage 2 -> Stage 1 decoder**.
It is deterministic, T4-friendly, and uses tile batch size 1 by default.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input image")
                model_preset = gr.Dropdown(
                    choices=list(MODEL_OPTIONS),
                    value="Best T4 default - Stage 2 Guarded Detail v2",
                    label="Model",
                )
                input_mode = gr.Radio(
                    choices=["Low-resolution image to upscale", "High-resolution image for controlled test"],
                    value="Low-resolution image to upscale",
                    label="Input type",
                )
                residual_strength = gr.Slider(
                    minimum=0.25,
                    maximum=1.25,
                    value=1.0,
                    step=0.05,
                    label="Correction strength (refiner/detail only)",
                )
                comparison_left = gr.Radio(
                    choices=["Bicubic x4", "Stage 2 condition", "Input LR nearest"],
                    value="Bicubic x4",
                    label="Slider left side",
                )
                tta_mode = gr.Dropdown(
                    choices=list(TTA_OPTIONS),
                    value="Off",
                    label="TTA inference",
                )
                use_tiling = gr.Checkbox(value=True, label="Tile large LR images")
                with gr.Row():
                    tile_overlap = gr.Slider(0, 96, value=32, step=8, label="Tile overlap")
                    tile_batch_size = gr.Slider(1, 16, value=1, step=1, label="Tile batch size")
                steps = gr.Slider(8, 64, value=32, step=4, label="Diffusion steps (ignored by refiner)")
                seed = gr.Number(value=123, precision=0, label="Seed")
                run_button = gr.Button("Run x4 SR", variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", lines=8)
                slider = gr.HTML(label="Before/after slider")
                gallery = gr.Gallery(label="Outputs", columns=2, object_fit="contain", height=520)
                download = gr.File(label="Download SR output")
        run_button.click(
            fn=run_sr,
            inputs=[
                input_image,
                model_preset,
                input_mode,
                residual_strength,
                comparison_left,
                tta_mode,
                use_tiling,
                tile_overlap,
                tile_batch_size,
                steps,
                seed,
            ],
            outputs=[status, slider, gallery, download],
        )
    return demo


def main() -> None:
    args = parse_args()
    app = build_app(args.repo_id)
    app.queue(default_concurrency_limit=1)
    app.launch(share=args.share, server_name="0.0.0.0", server_port=args.server_port)


if __name__ == "__main__":
    main()
