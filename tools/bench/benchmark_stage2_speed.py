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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the short Stage 2 clean-bicubic training throughput benchmark. "
            "This is the benchmark used to compare the current L40S reference "
            "speed of about 1.15 micro-steps/s."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup-step", type=int, default=50)
    parser.add_argument("--python", default=sys.executable, help="Python executable for the training subprocess.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
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


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    host = os.uname().nodename.split(".")[0]
    return Path.home() / "scratch" / "sr-diffusion" / "runs" / f"speed_stage2_bicubic_{host}_{stamp}"


def summarize(values: list[float], batch_size: int | None, grad_accum: int | None) -> str:
    lines = [
        "throughput_summary:",
        f"  stable_points: {len(values)}",
        f"  mean_steps_per_sec: {statistics.mean(values):.4f}",
        f"  median_steps_per_sec: {statistics.median(values):.4f}",
        f"  min_steps_per_sec: {min(values):.4f}",
        f"  max_steps_per_sec: {max(values):.4f}",
    ]
    if batch_size is not None:
        lines.append(f"  images_per_sec_microbatch: {statistics.mean(values) * batch_size:.4f}")
    if grad_accum is not None:
        lines.append(f"  optimizer_updates_per_sec: {statistics.mean(values) / grad_accum:.4f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir or default_output_dir()
    log_path = args.log_path or output_dir.with_suffix(".log")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size, grad_accum = load_train_shape(config_path)
    cmd = [
        args.python,
        "-u",
        str(REPO_ROOT / "tools" / "train" / "train_latent_pretrain.py"),
        "--config",
        str(config_path),
        "--limit-steps",
        str(args.steps),
        "--disable-wandb",
        "--output-dir",
        str(output_dir),
    ]

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
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            match = STEP_RE.search(line)
            if match and int(match.group(1)) >= args.warmup_step:
                stable_values.append(float(match.group(2)))
        return_code = process.wait()

    if stable_values:
        summary = summarize(stable_values, batch_size=batch_size, grad_accum=grad_accum)
        print(summary, flush=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(summary + "\n")
    else:
        print(f"No stable steps found at warmup_step >= {args.warmup_step}.", flush=True)

    if not args.keep_output:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"removed_temporary_output_dir: {output_dir}", flush=True)

    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
