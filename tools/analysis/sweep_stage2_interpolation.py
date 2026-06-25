from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.utils import get_device, load_config
from tools.train.train_latent_pretrain import (
    add_eval_selection_metric,
    build_stage2_model,
    evaluate,
    load_autoencoder,
    make_dataset,
    make_perceptual_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate weight interpolation between two compatible Stage 2 checkpoints."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument(
        "--alpha",
        type=float,
        action="append",
        default=None,
        help="Interpolation alpha where 0.0 is checkpoint A and 1.0 is checkpoint B. Can be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_model_state(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    checkpoint = torch.load(path, map_location="cpu")
    model_state = checkpoint["model"]
    if not isinstance(model_state, dict):
        raise TypeError(f"checkpoint model state must be a dict: {path}")
    return model_state, int(checkpoint.get("step", 0))


def validate_compatible_states(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> None:
    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)[:10]
        only_b = sorted(keys_b - keys_a)[:10]
        raise ValueError(f"checkpoint keys differ; only_a={only_a} only_b={only_b}")
    mismatches = [
        (key, tuple(state_a[key].shape), tuple(state_b[key].shape))
        for key in sorted(keys_a)
        if tuple(state_a[key].shape) != tuple(state_b[key].shape)
    ]
    if mismatches:
        raise ValueError(f"checkpoint tensor shapes differ; first mismatches={mismatches[:10]}")


def interpolate_state_dicts(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    mixed: dict[str, torch.Tensor] = {}
    for key, tensor_a in state_a.items():
        tensor_b = state_b[key]
        if tensor_a.is_floating_point():
            mixed[key] = tensor_a.detach().float().lerp(tensor_b.detach().float(), float(alpha))
            if tensor_a.dtype != torch.float32:
                mixed[key] = mixed[key].to(dtype=tensor_a.dtype)
        elif torch.equal(tensor_a, tensor_b):
            mixed[key] = tensor_a.detach().clone()
        else:
            mixed[key] = tensor_a.detach().clone() if alpha < 0.5 else tensor_b.detach().clone()
    return mixed


def make_eval_loader(
    config: dict[str, Any],
    eval_config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    limit_override: int | None = None,
    batch_size_override: int | None = None,
    num_workers_override: int | None = None,
) -> DataLoader:
    dataset = make_dataset(
        config,
        split=str(eval_config.get("split", "val")),
        seed=seed,
        deterministic=bool(eval_config.get("deterministic", True)),
        data_overrides=eval_config.get("data", {}),
    )
    limit = int(eval_config.get("limit", 0) if limit_override is None else limit_override)
    if limit > 0 and limit < len(dataset):
        dataset = Subset(dataset, list(range(limit)))
    batch_size = int(
        eval_config.get("batch_size", config.get("train", {}).get("batch_size", 1))
        if batch_size_override is None
        else batch_size_override
    )
    num_workers = int(
        eval_config.get("num_workers", config.get("data", {}).get("num_workers", 0))
        if num_workers_override is None
        else num_workers_override
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def evaluate_current_model(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    dtype_name: str,
    *,
    eval_limit: int | None,
    eval_batch_size: int | None,
    num_workers: int | None,
) -> dict[str, float]:
    eval_config = config.get("eval", {})
    loss_config = config.get("loss", {})
    perceptual_model = make_perceptual_model(loss_config, device=device)
    primary_loader = make_eval_loader(
        config,
        eval_config,
        seed=int(config.get("seed", 0)),
        device=device,
        limit_override=eval_limit,
        batch_size_override=eval_batch_size,
        num_workers_override=num_workers,
    )
    combined = evaluate(model, vae, primary_loader, device, dtype_name, loss_config, perceptual_model)
    for additional_config in eval_config.get("additional", []) or []:
        additional_loader = make_eval_loader(
            config,
            additional_config,
            seed=int(config.get("seed", 0)),
            device=device,
            limit_override=eval_limit,
            batch_size_override=eval_batch_size,
            num_workers_override=num_workers,
        )
        additional_metrics = evaluate(model, vae, additional_loader, device, dtype_name, loss_config, perceptual_model)
        name = str(additional_config["name"]).strip()
        combined.update({key.replace("eval/", f"eval_{name}/", 1): value for key, value in additional_metrics.items()})
    add_eval_selection_metric(combined, eval_config)
    return combined


def compact_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "eval/decoded_mean_psnr",
        "eval/decoded_ssim",
        "eval/excess_energy",
        "eval_mild/decoded_mean_psnr",
        "eval_photo_detail_mix/decoded_mean_psnr",
        "eval_photo_v2/decoded_mean_psnr",
        "eval_photo_v3_noise_mix/decoded_mean_psnr",
        "eval/robustness_score_raw",
        "eval/selection_valid",
        "eval/robustness_score",
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}


def alpha_label(alpha: float) -> str:
    return f"{alpha:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dtype_name = str(args.dtype or config.get("train", {}).get("dtype", "bf16"))
    device = get_device(args.device)
    alphas = sorted(set(args.alpha if args.alpha is not None else [0.0, 0.25, 0.5, 0.75, 1.0]))

    state_a, step_a = load_model_state(args.checkpoint_a)
    state_b, step_b = load_model_state(args.checkpoint_b)
    validate_compatible_states(state_a, state_b)

    eval_config = deepcopy(config)
    eval_config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    vae = load_autoencoder(eval_config, device=device)
    model = build_stage2_model(eval_config["model"]).to(device)

    results = []
    for alpha in alphas:
        mixed_state = interpolate_state_dicts(state_a, state_b, alpha)
        model.load_state_dict(mixed_state)
        del mixed_state
        metrics = evaluate_current_model(
            model,
            vae,
            eval_config,
            device,
            dtype_name,
            eval_limit=args.eval_limit,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        row = {
            "alpha": float(alpha),
            "label": f"{args.label_a}:{1.0 - float(alpha):.3f}_{args.label_b}:{float(alpha):.3f}",
            "checkpoint_a": str(args.checkpoint_a),
            "checkpoint_b": str(args.checkpoint_b),
            "checkpoint_a_step": step_a,
            "checkpoint_b_step": step_b,
            "metrics": metrics,
            "compact": compact_metrics(metrics),
        }
        results.append(row)
        compact = row["compact"]
        print(
            "alpha="
            f"{alpha_label(float(alpha)):>5} "
            f"clean={compact.get('eval/decoded_mean_psnr', float('nan')):.4f} "
            f"ssim={compact.get('eval/decoded_ssim', float('nan')):.6f} "
            f"excess={compact.get('eval/excess_energy', float('nan')):.6f} "
            f"mild={compact.get('eval_mild/decoded_mean_psnr', float('nan')):.4f} "
            f"detail={compact.get('eval_photo_detail_mix/decoded_mean_psnr', float('nan')):.4f} "
            f"v2={compact.get('eval_photo_v2/decoded_mean_psnr', float('nan')):.4f} "
            f"v3={compact.get('eval_photo_v3_noise_mix/decoded_mean_psnr', float('nan')):.4f} "
            f"score={compact.get('eval/robustness_score_raw', float('nan')):.4f} "
            f"valid={int(compact.get('eval/selection_valid', 0.0))}"
        )

    best = max(
        results,
        key=lambda row: (
            float(row["compact"].get("eval/selection_valid", 0.0)),
            float(row["compact"].get("eval/robustness_score_raw", float("-inf"))),
        ),
    )
    payload = {
        "config": str(args.config),
        "checkpoint_a": str(args.checkpoint_a),
        "checkpoint_b": str(args.checkpoint_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "checkpoint_a_step": step_a,
        "checkpoint_b_step": step_b,
        "dtype": dtype_name,
        "eval_limit": args.eval_limit,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "results": results,
        "best_alpha": float(best["alpha"]),
        "best_compact": best["compact"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
