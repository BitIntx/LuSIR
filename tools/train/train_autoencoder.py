from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets import ManifestImageDataset
from sr_diffusion.eval import evaluate_autoencoder, make_eval_loader, save_eval_metrics
from sr_diffusion.losses import vae_loss
from sr_diffusion.models import AutoencoderKL
from sr_diffusion.utils import autocast_context, get_device, load_config, save_config, seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 1 autoencoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    array = np.round(array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def make_dataset_for_split(config: dict[str, Any], split: str, seed: int) -> ManifestImageDataset:
    data_config = config["data"]
    return ManifestImageDataset(
        manifest_path=data_config["manifest"],
        split=split,
        hr_size=data_config.get("hr_size", 512),
        scale=data_config.get("scale", 4),
        domains=data_config.get("domains", {"photo": 0, "anime": 1}),
        degradation_preset=data_config.get("degradation_preset", "mild"),
        seed=seed,
        deterministic=True,
    )


def make_fixed_sample_batch(config: dict[str, Any], seed: int) -> dict[str, Any] | None:
    sample_config = config.get("logging", {}).get("samples", {})
    if not bool(sample_config.get("enabled", True)):
        return None

    count = int(sample_config.get("count", 4))
    if count <= 0:
        return None

    split = str(sample_config.get("split", "val"))
    fallback_split = str(sample_config.get("fallback_split", "train"))
    try:
        dataset = make_dataset_for_split(config, split=split, seed=seed)
    except ValueError:
        if split == fallback_split:
            raise
        print(f"sample split '{split}' is empty; falling back to '{fallback_split}'")
        split = fallback_split
        dataset = make_dataset_for_split(config, split=split, seed=seed)

    configured_indices = sample_config.get("indices")
    if configured_indices is None:
        indices = list(range(min(count, len(dataset))))
    else:
        indices = [int(index) % len(dataset) for index in configured_indices[:count]]

    items = [dataset[index] for index in indices]
    return {
        "hr": torch.stack([item["hr"] for item in items], dim=0),
        "lr": torch.stack([item["lr"] for item in items], dim=0),
        "path": [item["path"] for item in items],
        "split": split,
        "indices": indices,
    }


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def init_wandb(config: dict[str, Any], output_dir: Path, model: torch.nn.Module) -> Any | None:
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging is enabled, but wandb is not installed") from exc

    wandb_dir = Path(wandb_cfg.get("dir", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = wandb_cfg.get("mode", "offline")
    os.environ.setdefault("WANDB_MODE", str(mode))
    tags = list(wandb_cfg.get("tags") or [])
    if "stage1" not in tags:
        tags.append("stage1")

    run = wandb.init(
        project=wandb_cfg.get("project", "LuSIR"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name", config.get("project", {}).get("name")),
        dir=str(wandb_dir),
        mode=mode,
        tags=tags,
        group=wandb_cfg.get("group", "stage1"),
        job_type=wandb_cfg.get("job_type", "autoencoder"),
        config=clean_config(config),
    )
    if bool(wandb_cfg.get("watch", False)):
        wandb.watch(model, log="gradients", log_freq=int(wandb_cfg.get("watch_log_freq", 100)))
    return run


def wandb_log(run: Any | None, data: dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": clean_config(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    output_dir = Path(config["project"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples"
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    train_cfg = config["train"]
    device = get_device(train_cfg.get("device", "auto"))
    dtype_name = train_cfg.get("dtype", "bf16")
    print(f"device={device} dtype={dtype_name}")

    dataset = ManifestImageDataset.from_config(config["data"], seed=seed)
    fixed_sample_batch = make_fixed_sample_batch(config, seed=seed)
    if fixed_sample_batch is not None:
        print(
            "sample_logging="
            f"split={fixed_sample_batch['split']} "
            f"indices={fixed_sample_batch['indices']} "
            f"count={len(fixed_sample_batch['path'])}"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )
    eval_cfg = config.get("eval", {})
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_loader = None
    eval_every = int(eval_cfg.get("every", 1000))
    eval_run_at_start = bool(eval_cfg.get("run_at_start", True))
    if eval_enabled:
        eval_loader = make_eval_loader(
            config,
            split=str(eval_cfg.get("split", "val")),
            seed=seed,
            batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 1))),
            limit=int(eval_cfg.get("limit", 0)),
            num_workers=int(eval_cfg.get("num_workers", config["data"].get("num_workers", 0))),
        )
        print(
            "eval="
            f"split={eval_cfg.get('split', 'val')} "
            f"limit={eval_cfg.get('limit', 0)} "
            f"batch_size={eval_cfg.get('batch_size', train_cfg.get('batch_size', 1))}"
        )

    model = AutoencoderKL.from_config(config["model"]).to(device)
    if bool(train_cfg.get("compile", False)):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed step={start_step}")

    max_steps = int(args.limit_steps or train_cfg.get("max_steps", 1000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 1000))
    sample_every = int(train_cfg.get("sample_every", 500))

    run = init_wandb(config, output_dir, model)
    if run is not None:
        wandb_log(
            run,
            {
                "dataset/num_images": len(dataset),
                "train/batch_size": int(train_cfg.get("batch_size", 1)),
                "train/grad_accum_steps": grad_accum_steps,
            },
            step=start_step,
        )

    model.train()
    step = start_step
    best_eval_recon = float("inf")
    last_log = time.time()
    last_log_step = step
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        for batch in dataloader:
            step += 1
            hr = batch["hr"].to(device, non_blocking=True)
            target = hr.mul(2.0).sub(1.0)

            with autocast_context(device, dtype_name):
                output = model(target, sample_posterior=True)
                loss, metrics = vae_loss(
                    output.reconstruction,
                    target,
                    output.mean,
                    output.logvar,
                    config["loss"],
                )
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()

            if step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0 or step == 1:
                elapsed = max(1e-6, time.time() - last_log)
                interval_steps = max(1, step - last_log_step)
                last_log = time.time()
                last_log_step = step
                print(
                    f"step={step} loss={metrics['loss']:.5f} "
                    f"recon={metrics['recon']:.5f} kl={metrics['kl']:.5f} "
                    f"steps_per_sec={interval_steps / elapsed:.2f}"
                )
                wandb_log(
                    run,
                    {
                        "train/loss": metrics["loss"],
                        "train/recon": metrics["recon"],
                        "train/kl": metrics["kl"],
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "system/steps_per_sec": interval_steps / elapsed,
                    },
                    step=step,
                )

            should_eval = (
                eval_enabled
                and eval_loader is not None
                and eval_every > 0
                and (step % eval_every == 0 or (step == 1 and eval_run_at_start))
            )
            if should_eval:
                eval_metrics = evaluate_autoencoder(
                    model,
                    eval_loader,
                    device,
                    dtype_name,
                    config["loss"],
                )
                save_eval_metrics(eval_dir / f"step_{step:07d}_metrics.json", step=step, metrics=eval_metrics)
                print(
                    f"eval step={step} "
                    f"recon={eval_metrics['eval/recon']:.5f} "
                    f"kl={eval_metrics['eval/kl']:.5f} "
                    f"psnr={eval_metrics['eval/psnr']:.2f}"
                )
                wandb_log(run, eval_metrics, step=step)
                if eval_metrics["eval/recon"] < best_eval_recon:
                    best_eval_recon = eval_metrics["eval/recon"]
                    save_checkpoint(checkpoints_dir / "best_eval_recon.pt", model, optimizer, step, config)

            if step % sample_every == 0 or step == 1:
                with torch.no_grad():
                    sample_source = fixed_sample_batch if fixed_sample_batch is not None else batch
                    sample_hr = sample_source["hr"].to(device, non_blocking=True)
                    sample_target = sample_hr.mul(2.0).sub(1.0)
                    with autocast_context(device, dtype_name):
                        sample_output = model(sample_target, sample_posterior=False)

                    sample_count = sample_hr.shape[0]
                    lr = sample_source["lr"][:sample_count].float().cpu()
                    gt = denormalize(sample_target).float().cpu()
                    hr = denormalize(sample_output.reconstruction).float().cpu()
                    lr_display = F.interpolate(lr, size=gt.shape[-2:], mode="nearest")

                    lr_path = samples_dir / f"step_{step:07d}_lr.png"
                    gt_path = samples_dir / f"step_{step:07d}_gt.png"
                    hr_path = samples_dir / f"step_{step:07d}_hr.png"
                    save_image(lr_display, lr_path, nrow=sample_count)
                    save_image(gt, gt_path, nrow=sample_count)
                    save_image(hr, hr_path, nrow=sample_count)

                    if run is not None:
                        import wandb

                        paths = sample_source.get("path", [""] * sample_count)
                        captions = [Path(str(path)).name or f"sample_{idx}" for idx, path in enumerate(paths[:sample_count])]
                        wandb_log(
                            run,
                            {
                                "samples/LR": [
                                    wandb.Image(tensor_to_pil(image), caption=caption)
                                    for image, caption in zip(lr_display, captions, strict=True)
                                ],
                                "samples/GT": [
                                    wandb.Image(tensor_to_pil(image), caption=caption)
                                    for image, caption in zip(gt, captions, strict=True)
                                ],
                                "samples/HR": [
                                    wandb.Image(tensor_to_pil(image), caption=caption)
                                    for image, caption in zip(hr, captions, strict=True)
                                ],
                            },
                            step=step,
                        )

            if step % save_every == 0 or step == max_steps:
                save_checkpoint(checkpoints_dir / f"step_{step:07d}.pt", model, optimizer, step, config)
                save_checkpoint(checkpoints_dir / "latest.pt", model, optimizer, step, config)

            if step >= max_steps:
                break

    print(f"finished step={step}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
