from __future__ import annotations

import argparse
from typing import Any


STAGES = ("stage1", "stage2", "stage3", "stage4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize existing W&B runs by training stage.")
    parser.add_argument("--project", default="jwheo/LuSIR")
    parser.add_argument("--apply", action="store_true", help="Apply updates. The default is a dry run.")
    return parser.parse_args()


def classify_run(run: Any) -> tuple[str, str]:
    tags = set(run.tags or [])
    stage = next((candidate for candidate in STAGES if candidate in tags), None)
    name = str(run.name or "")
    if stage is None and ("autoencoder" in tags or name.startswith("autoencoder_")):
        stage = "stage1"
    if stage is None:
        raise ValueError(f"cannot classify run {run.id}: {name}")

    if "autoencoder" in tags or name.startswith("autoencoder_"):
        job_type = "autoencoder"
    elif "residual-refiner" in tags or name.startswith("residual_refiner_"):
        job_type = "residual-refiner"
    elif "latent-pretrain" in tags or name.startswith("latent_pretrain_"):
        job_type = "latent-pretrain"
    elif "diffusion" in tags or name.startswith("diffusion_"):
        job_type = "diffusion"
    else:
        job_type = "training"
    return stage, job_type


def main() -> None:
    args = parse_args()
    import wandb

    api = wandb.Api()
    runs = list(api.runs(args.project))
    changed = 0
    for run in runs:
        stage, job_type = classify_run(run)
        tags = list(run.tags or [])
        if stage not in tags:
            tags.append(stage)
        needs_update = run.group != stage or run.job_type != job_type or tags != list(run.tags or [])
        print(
            f"{run.id} {run.name}: group={run.group or '-'}->{stage} "
            f"job_type={run.job_type or '-'}->{job_type}"
        )
        if args.apply and needs_update:
            run.group = stage
            run.job_type = job_type
            run.tags = tags
            run.update()
            changed += 1
    print(f"{'updated' if args.apply else 'would_update'}={changed if args.apply else len(runs)} total={len(runs)}")


if __name__ == "__main__":
    main()
