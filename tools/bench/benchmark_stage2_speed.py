from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue.yaml"
STEP_RE = re.compile(r"step=(\d+).*steps_per_sec=([0-9.]+)")

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LuSIR Stage 2 quick throughput benchmark. By default it "
            "uses torchrun over every visible CUDA GPU, matching the real DDP "
            "training path when multiple GPUs are present."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup-step", type=int, default=50)
    parser.add_argument("--python", default=sys.executable, help="Python executable for the training subprocess.")
    parser.add_argument(
        "--nproc-per-node",
        default="auto",
        help="GPU processes for torchrun. 'auto' uses every visible CUDA GPU; use 1 for single-GPU.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Colorize the final result block.",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep the temporary run directory. By default it is removed because it contains checkpoints.",
    )
    return parser.parse_args()


def load_train_shape(config_path: Path) -> tuple[int | None, int | None]:
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    train_cfg = config.get("train", {})
    batch_size = train_cfg.get("batch_size")
    grad_accum = train_cfg.get("grad_accum_steps")
    return (int(batch_size) if batch_size is not None else None, int(grad_accum) if grad_accum is not None else None)


def detect_cuda_device_count(python_executable: str) -> int:
    code = "import torch; print(torch.cuda.device_count())"
    completed = subprocess.run(
        [python_executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def resolve_nproc(value: str, python_executable: str) -> int:
    if value == "auto":
        count = detect_cuda_device_count(python_executable)
        if count <= 0:
            raise RuntimeError("No CUDA GPUs are visible to PyTorch.")
        return count
    count = int(value)
    if count <= 0:
        raise ValueError("--nproc-per-node must be positive")
    return count


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    host = os.uname().nodename.split(".")[0]
    return Path.home() / "scratch" / "sr-diffusion" / "runs" / f"speed_stage2_ddp_{host}_{stamp}"


def use_color(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()


def paint(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def summary_metrics(
    values: list[float],
    batch_size: int | None,
    grad_accum: int | None,
    world_size: int,
) -> dict[str, float]:
    mean = statistics.mean(values)
    metrics = {
        "stable_points": float(len(values)),
        "world_size": float(world_size),
        "mean_steps_per_sec": mean,
        "median_steps_per_sec": statistics.median(values),
        "min_steps_per_sec": min(values),
        "max_steps_per_sec": max(values),
    }
    if batch_size is not None:
        metrics["global_images_per_sec"] = mean * batch_size * world_size
        metrics["local_images_per_sec"] = mean * batch_size
    if grad_accum is not None:
        metrics["optimizer_updates_per_sec"] = mean / grad_accum
    if batch_size is not None and grad_accum is not None:
        metrics["effective_batch_size"] = float(batch_size * grad_accum * world_size)
    return metrics


def format_plain_summary(metrics: dict[str, float]) -> str:
    lines = [
        "throughput_summary:",
        f"  world_size: {int(metrics['world_size'])}",
        f"  stable_points: {int(metrics['stable_points'])}",
        f"  mean_steps_per_sec: {metrics['mean_steps_per_sec']:.4f}",
        f"  median_steps_per_sec: {metrics['median_steps_per_sec']:.4f}",
        f"  min_steps_per_sec: {metrics['min_steps_per_sec']:.4f}",
        f"  max_steps_per_sec: {metrics['max_steps_per_sec']:.4f}",
    ]
    if "global_images_per_sec" in metrics:
        lines.append(f"  global_images_per_sec: {metrics['global_images_per_sec']:.4f}")
        lines.append(f"  local_images_per_sec: {metrics['local_images_per_sec']:.4f}")
    if "optimizer_updates_per_sec" in metrics:
        lines.append(f"  optimizer_updates_per_sec: {metrics['optimizer_updates_per_sec']:.4f}")
    if "effective_batch_size" in metrics:
        lines.append(f"  effective_batch_size: {int(metrics['effective_batch_size'])}")
    return "\n".join(lines)


def format_color_result(metrics: dict[str, float], log_path: Path, output_dir: Path, removed_output: bool, enabled: bool) -> str:
    mode = "DDP" if int(metrics["world_size"]) > 1 else "single GPU"
    mean_text = paint(f"{metrics['mean_steps_per_sec']:.4f} step/s", "green", enabled)
    median_text = paint(f"{metrics['median_steps_per_sec']:.4f} step/s", "green", enabled)
    images_text = (
        paint(f"{metrics['global_images_per_sec']:.4f} img/s global", "green", enabled)
        if "global_images_per_sec" in metrics
        else None
    )
    lines = [
        "",
        paint("=" * 72, "cyan", enabled),
        f"{paint('RESULT', 'bold', enabled)} {paint('LuSIR Stage 2 Quick Benchmark', 'cyan', enabled)}",
        f"  mode        {mode}",
        f"  world_size  {int(metrics['world_size'])}",
        f"  mean        {mean_text}",
        f"  median      {median_text}",
        f"  range       {metrics['min_steps_per_sec']:.4f} - {metrics['max_steps_per_sec']:.4f} step/s",
    ]
    if images_text is not None:
        lines.append(f"  images      {images_text}")
        lines.append(f"  local       {metrics['local_images_per_sec']:.4f} img/s per GPU")
    if "optimizer_updates_per_sec" in metrics:
        lines.append(f"  updates     {metrics['optimizer_updates_per_sec']:.4f} optimizer updates/s")
    if "effective_batch_size" in metrics:
        lines.append(f"  eff_batch   {int(metrics['effective_batch_size'])}")
    lines.extend(
        [
            f"  log         {log_path}",
            f"  temp        {output_dir} {'(removed)' if removed_output else '(kept)'}",
            paint("=" * 72, "cyan", enabled),
        ]
    )
    return "\n".join(lines)


def build_command(args: argparse.Namespace, output_dir: Path, nproc: int) -> list[str]:
    train_script = REPO_ROOT / "tools" / "train" / "train_latent_pretrain.py"
    train_args = [
        str(train_script),
        "--config",
        str(args.config.resolve()),
        "--limit-steps",
        str(args.steps),
        "--disable-wandb",
        "--output-dir",
        str(output_dir),
    ]
    if nproc > 1:
        return [
            args.python,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            *train_args,
        ]
    return [args.python, "-u", *train_args]


def main() -> None:
    args = parse_args()
    nproc = resolve_nproc(str(args.nproc_per_node), args.python)
    config_path = args.config.resolve()
    output_dir = args.output_dir or default_output_dir()
    log_path = args.log_path or output_dir.with_suffix(".log")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size, grad_accum = load_train_shape(config_path)
    cmd = build_command(args, output_dir=output_dir, nproc=nproc)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    stable_values: list[float] = []
    print("command:", " ".join(cmd), flush=True)
    print(f"log_path: {log_path}", flush=True)
    print(f"temporary_output_dir: {output_dir}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            match = STEP_RE.search(line)
            if match and int(match.group(1)) >= args.warmup_step:
                stable_values.append(float(match.group(2)))
        return_code = process.wait()

    removed_output = False
    if not args.keep_output:
        shutil.rmtree(output_dir, ignore_errors=True)
        removed_output = True
        print(f"removed_temporary_output_dir: {output_dir}", flush=True)

    if return_code != 0:
        raise SystemExit(return_code)
    if not stable_values:
        raise RuntimeError(f"No stable benchmark values found after warmup step {args.warmup_step}.")

    metrics = summary_metrics(stable_values, batch_size=batch_size, grad_accum=grad_accum, world_size=nproc)
    plain_summary = format_plain_summary(metrics)
    color_enabled = use_color(args.color)
    color_result = format_color_result(metrics, log_path, output_dir, removed_output, color_enabled)
    print(plain_summary, flush=True)
    print(color_result, flush=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(plain_summary + "\n")
        log_handle.write(color_result + "\n")


if __name__ == "__main__":
    main()
