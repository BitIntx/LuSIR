from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.train.train_latent_pretrain import make_dataset, seed_worker  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LuSIR Stage 2 DataLoader throughput.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, workers: int) -> None:
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config.get("seed", 0))
    train_cfg = config.get("train", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 1))
    dataset = make_dataset(config, split=config["data"].get("split", "train"), seed=seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": bool(args.pin_memory),
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
        "drop_last": True,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

    loader = DataLoader(dataset, **loader_kwargs)
    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        next(iterator)

    start = time.perf_counter()
    count = 0
    for _ in range(args.batches):
        next(iterator)
        count += 1
    elapsed = time.perf_counter() - start
    batches_per_sec = count / max(elapsed, 1e-9)
    print(
        f"workers={workers} "
        f"persistent_workers={bool(args.persistent_workers) if workers > 0 else False} "
        f"prefetch_factor={args.prefetch_factor if workers > 0 else None} "
        f"pin_memory={bool(args.pin_memory)} "
        f"batches={count} "
        f"seconds={elapsed:.2f} "
        f"batches_per_sec={batches_per_sec:.2f} "
        f"images_per_sec={batches_per_sec * batch_size:.2f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    for workers in args.workers:
        run_one(args, workers=workers)


if __name__ == "__main__":
    main()
