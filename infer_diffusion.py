from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_diffusion.datasets.manifest import crop_square, pil_to_tensor
from sr_diffusion.degradations import DegradationPipeline
from sr_diffusion.models import AutoencoderKL, ConditionalUNet, LRToLatentPredictor, NoiseScheduler, predict_x0_and_noise
from sr_diffusion.utils import autocast_context, cuda_autocast_dtype, get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 conditional diffusion SR sampling.")
    parser.add_argument("--config", type=Path, default=Path("configs/hf/diffusion_stage4_condition.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-lr", type=Path, help="Low-resolution RGB input image.")
    input_group.add_argument("--input-hr", type=Path, help="HR image to center-crop and degrade for controlled eval.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="photo")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--init", choices=("noise", "condition"), default=None)
    parser.add_argument("--start-timestep", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resize-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile", action="store_true", help="Run tiled inference for arbitrary-size LR images.")
    parser.add_argument("--tile-overlap", type=int, default=32, help="LR-pixel overlap between 128x128 tiles.")
    parser.add_argument("--tile-batch-size", type=int, default=1, help="Number of LR tiles to sample at once.")
    parser.add_argument("--save-every", type=int, default=0, help="Save intermediate samples every N sampler steps.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=4,
        help="Print sampler progress every N DDIM steps. Set to 0 to disable.",
    )
    return parser.parse_args()


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    array = np.round(array * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def float_array_to_pil(array: np.ndarray) -> Image.Image:
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")


def resolve_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    config_path = config.get("_config_path")
    if config_path:
        config_relative = Path(config_path).parent / path
        if config_relative.exists():
            return config_relative
        if path.parts and path.parts[0] in (".", ".."):
            return config_relative
        if len(path.parts) == 1:
            return config_relative
    return cwd_path


def resolve_checkpoint_arg(args: argparse.Namespace, config: dict) -> Path:
    if args.checkpoint is not None:
        return resolve_path(config, args.checkpoint)
    checkpoint = config.get("inference", {}).get("checkpoint") or config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("No checkpoint provided. Pass --checkpoint or set inference.checkpoint in the config.")
    return resolve_path(config, checkpoint)


def inference_module_dtype(device: torch.device, dtype_name: str | None) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return cuda_autocast_dtype(dtype_name)


def load_autoencoder(config: dict, device: torch.device, dtype_name: str | None) -> AutoencoderKL:
    auto_cfg = config["autoencoder"]
    vae_config = load_config(resolve_path(config, auto_cfg["config"]))
    vae = AutoencoderKL.from_config(vae_config["model"])
    checkpoint = torch.load(resolve_path(config, auto_cfg["checkpoint"]), map_location="cpu")
    vae.load_state_dict(checkpoint["model"])
    dtype = inference_module_dtype(device, dtype_name)
    if dtype is None:
        vae = vae.to(device)
    else:
        vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae


def load_condition_encoder(config: dict, device: torch.device, dtype_name: str | None) -> LRToLatentPredictor:
    cond_cfg = config["condition_encoder"]
    cond_config = load_config(resolve_path(config, cond_cfg["config"]))
    encoder = LRToLatentPredictor.from_config(cond_config["model"])
    checkpoint = torch.load(resolve_path(config, cond_cfg["checkpoint"]), map_location="cpu")
    encoder.load_state_dict(checkpoint["model"])
    dtype = inference_module_dtype(device, dtype_name)
    if dtype is None:
        encoder = encoder.to(device)
    else:
        encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()
    return encoder


def load_unet(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
    dtype_name: str | None,
) -> tuple[ConditionalUNet, int]:
    model = ConditionalUNet.from_config(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    dtype = inference_module_dtype(device, dtype_name)
    if dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, int(checkpoint.get("step", 0))


def prepare_inputs(args: argparse.Namespace, config: dict) -> tuple[Image.Image, Image.Image | None]:
    data_config = config["data"]
    hr_size = int(data_config["hr_size"])
    scale = int(data_config.get("scale", 4))
    lr_size = hr_size // scale
    rng = random.Random(args.seed)

    if args.input_hr:
        hr = Image.open(args.input_hr).convert("RGB")
        hr = crop_square(hr, hr_size, rng=rng, random_crop=False)
        pipeline = DegradationPipeline.from_preset(data_config.get("degradation_preset", "mild"), scale=scale)
        lr = pipeline.apply(hr, rng=rng, out_size=lr_size)
        return lr, hr

    lr = Image.open(args.input_lr).convert("RGB")
    if args.resize_lr:
        lr = crop_square(lr, lr_size, rng=rng, random_crop=False)
    elif lr.size != (lr_size, lr_size):
        raise ValueError(f"Expected LR size {(lr_size, lr_size)}, got {lr.size}. Use --resize-lr to resize/crop.")
    return lr, None


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive: {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"tile_overlap must be in [0, {tile_size - 1}], got {overlap}")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def edge_pad_image(image: Image.Image, min_width: int, min_height: int) -> Image.Image:
    width, height = image.size
    padded_width = max(width, min_width)
    padded_height = max(height, min_height)
    if (padded_width, padded_height) == (width, height):
        return image
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pad_width = padded_width - width
    pad_height = padded_height - height
    padded = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    return Image.fromarray(padded, mode="RGB")


def tile_blend_mask(
    tile_size: int,
    overlap: int,
    *,
    left_edge: bool,
    right_edge: bool,
    top_edge: bool,
    bottom_edge: bool,
) -> np.ndarray:
    mask_size = int(tile_size)
    overlap = max(0, min(int(overlap), mask_size // 2))
    weights_x = np.ones(mask_size, dtype=np.float32)
    weights_y = np.ones(mask_size, dtype=np.float32)
    if overlap > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)) ** 2
        if not left_edge:
            weights_x[:overlap] = ramp
        if not right_edge:
            weights_x[-overlap:] = ramp[::-1]
        if not top_edge:
            weights_y[:overlap] = ramp
        if not bottom_edge:
            weights_y[-overlap:] = ramp[::-1]
    return weights_y[:, None, None] * weights_x[None, :, None]


def make_timesteps(
    num_train_timesteps: int,
    num_steps: int,
    device: torch.device,
    start_timestep: int | None = None,
) -> torch.Tensor:
    steps = max(1, min(int(num_steps), int(num_train_timesteps)))
    start = num_train_timesteps - 1 if start_timestep is None else max(0, min(int(start_timestep), num_train_timesteps - 1))
    timesteps = torch.linspace(start, 0, steps, device=device).round().long()
    return torch.unique_consecutive(timesteps)


def resolve_start_timestep(config: dict, requested: int | None) -> int | None:
    if requested is not None:
        return requested
    value = config.get("sampling", {}).get("start_timestep")
    return None if value is None else int(value)


def ddim_sample(
    model: ConditionalUNet,
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    scheduler: NoiseScheduler,
    lr: torch.Tensor,
    domain_id: torch.Tensor,
    steps: int,
    eta: float,
    init: str,
    start_timestep: int | None,
    dtype_name: str,
    seed: int,
    output_dir: Path,
    save_every: int = 0,
    progress_every: int = 4,
    progress_label: str = "sample",
    diffusion_config: dict | None = None,
) -> torch.Tensor:
    device = lr.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if start_timestep is None and init == "condition":
        start_timestep = min(int(scheduler.num_train_timesteps) - 1, 50)
    timesteps = make_timesteps(scheduler.num_train_timesteps, steps, device=device, start_timestep=start_timestep)
    lr_input = lr.mul(2.0).sub(1.0)
    with torch.no_grad(), autocast_context(device, dtype_name):
        condition = condition_encoder(lr_input, domain_id)

    noise = torch.randn(
        condition.shape[0],
        model.latent_channels,
        condition.shape[-2],
        condition.shape[-1],
        device=device,
        generator=generator,
    )
    if init == "noise":
        latent = noise
    elif init == "condition":
        first_timestep = torch.full((condition.shape[0],), int(timesteps[0].item()), device=device, dtype=torch.long)
        latent = scheduler.add_noise(condition, noise, first_timestep)
    else:
        raise ValueError(f"Unsupported init: {init}")
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=latent.dtype)

    total_steps = len(timesteps)
    if progress_every > 0:
        print(
            f"{progress_label} sampler_start steps={total_steps} batch={lr.shape[0]} "
            f"init={init} start_timestep={int(timesteps[0].item())}",
            flush=True,
        )

    with torch.no_grad():
        for index, timestep in enumerate(timesteps):
            timestep_batch = torch.full((latent.shape[0],), int(timestep.item()), device=device, dtype=torch.long)
            with autocast_context(device, dtype_name):
                model_output = model(latent, timestep_batch, condition, domain_id)
            model_output = model_output.to(dtype=latent.dtype)
            pred_x0, predicted_noise = predict_x0_and_noise(
                scheduler,
                latent,
                timestep_batch,
                model_output,
                condition.to(dtype=latent.dtype),
                diffusion_config,
            )

            prev_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            alpha_t = alphas_cumprod[int(timestep.item())]
            alpha_prev = torch.ones((), device=device, dtype=latent.dtype) if prev_timestep < 0 else alphas_cumprod[prev_timestep]

            if prev_timestep < 0:
                latent = pred_x0
            else:
                sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t)) * torch.sqrt(
                    torch.clamp(1.0 - alpha_t / alpha_prev, min=0.0)
                )
                direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma.pow(2), min=0.0))
                latent = torch.sqrt(alpha_prev) * pred_x0 + direction_scale * predicted_noise
                if eta > 0.0:
                    latent = latent + sigma * torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=generator)

            if save_every > 0 and ((index + 1) % save_every == 0 or index + 1 == len(timesteps)):
                with autocast_context(device, dtype_name):
                    preview = vae.decode(latent)
                for sample_idx, image in enumerate(denormalize(preview)):
                    tensor_to_pil(image).save(output_dir / f"sample_{sample_idx:02d}_step_{index + 1:03d}.png")
            step_index = index + 1
            if progress_every > 0 and (
                step_index == 1 or step_index % progress_every == 0 or step_index == total_steps
            ):
                print(
                    f"{progress_label} ddim_step={step_index}/{total_steps} "
                    f"timestep={int(timestep.item())}",
                    flush=True,
                )

    with torch.no_grad(), autocast_context(device, dtype_name):
        decoded = vae.decode(latent)
    return denormalize(decoded)


def tiled_sample(
    model: ConditionalUNet,
    vae: AutoencoderKL,
    condition_encoder: LRToLatentPredictor,
    scheduler: NoiseScheduler,
    lr_image: Image.Image,
    domain_id_value: int,
    scale: int,
    tile_lr_size: int,
    overlap_lr: int,
    tile_batch_size: int,
    steps: int,
    eta: float,
    init: str,
    start_timestep: int | None,
    dtype_name: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    progress_every: int = 4,
    diffusion_config: dict | None = None,
) -> Image.Image:
    if tile_batch_size <= 0:
        raise ValueError(f"tile_batch_size must be positive: {tile_batch_size}")
    if overlap_lr < 0 or overlap_lr >= tile_lr_size:
        raise ValueError(f"tile_overlap must be in [0, {tile_lr_size - 1}], got {overlap_lr}")

    original_width, original_height = lr_image.size
    padded = edge_pad_image(lr_image, tile_lr_size, tile_lr_size)
    padded_width, padded_height = padded.size
    x_positions = tile_positions(padded_width, tile_lr_size, overlap_lr)
    y_positions = tile_positions(padded_height, tile_lr_size, overlap_lr)
    tile_hr_size = tile_lr_size * scale
    overlap_hr = overlap_lr * scale
    canvas = np.zeros((padded_height * scale, padded_width * scale, 3), dtype=np.float32)
    weights = np.zeros((padded_height * scale, padded_width * scale, 1), dtype=np.float32)

    tiles: list[tuple[int, int]] = [(x, y) for y in y_positions for x in x_positions]
    print(
        f"tile_inference lr_size={original_width}x{original_height} padded={padded_width}x{padded_height} "
        f"tiles={len(tiles)} tile={tile_lr_size} overlap={overlap_lr} batch={tile_batch_size}",
        flush=True,
    )
    num_batches = (len(tiles) + tile_batch_size - 1) // tile_batch_size
    for batch_start in range(0, len(tiles), tile_batch_size):
        batch_index = batch_start // tile_batch_size + 1
        batch_coords = tiles[batch_start : batch_start + tile_batch_size]
        print(
            f"tile_batch_start={batch_index}/{num_batches} "
            f"tiles={batch_start + 1}-{batch_start + len(batch_coords)}/{len(tiles)}",
            flush=True,
        )
        batch_images = [
            pil_to_tensor(padded.crop((x, y, x + tile_lr_size, y + tile_lr_size))) for x, y in batch_coords
        ]
        lr_tensor = torch.stack(batch_images, dim=0).to(device)
        domain_ids = torch.full((len(batch_coords),), domain_id_value, device=device, dtype=torch.long)
        with torch.no_grad():
            output = ddim_sample(
                model=model,
                vae=vae,
                condition_encoder=condition_encoder,
                scheduler=scheduler,
                lr=lr_tensor,
                domain_id=domain_ids,
                steps=steps,
                eta=eta,
                init=init,
                start_timestep=start_timestep,
                dtype_name=dtype_name,
                seed=seed + batch_start,
                output_dir=output_dir,
                save_every=0,
                progress_every=progress_every,
                progress_label=f"tile_batch={batch_index}/{num_batches}",
                diffusion_config=diffusion_config,
            )

        for tile_output, (x, y) in zip(output, batch_coords, strict=True):
            tile_array = tile_output.detach().float().cpu().permute(1, 2, 0).numpy()
            left_edge = x == 0
            top_edge = y == 0
            right_edge = x + tile_lr_size >= padded_width
            bottom_edge = y + tile_lr_size >= padded_height
            mask = tile_blend_mask(
                tile_hr_size,
                overlap_hr,
                left_edge=left_edge,
                right_edge=right_edge,
                top_edge=top_edge,
                bottom_edge=bottom_edge,
            )
            x0 = x * scale
            y0 = y * scale
            canvas[y0 : y0 + tile_hr_size, x0 : x0 + tile_hr_size] += tile_array * mask
            weights[y0 : y0 + tile_hr_size, x0 : x0 + tile_hr_size] += mask
        print(f"tiles_done={min(batch_start + len(batch_coords), len(tiles))}/{len(tiles)}", flush=True)

    stitched = canvas / np.maximum(weights, 1e-6)
    stitched = stitched[: original_height * scale, : original_width * scale]
    return float_array_to_pil(stitched)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = resolve_checkpoint_arg(args, config)
    inference_config = config.get("inference", {})
    steps = int(args.steps if args.steps is not None else inference_config.get("steps", 50))
    init = str(args.init or inference_config.get("init", "condition"))
    device = get_device(args.device)
    dtype_name = config["train"].get("dtype", "bf16")
    data_config = config["data"]
    domains = data_config.get("domains", {"photo": 0, "anime": 1})
    if args.domain not in domains:
        raise ValueError(f"Unknown domain '{args.domain}'. Available: {sorted(domains)}")
    if args.tile and args.input_hr:
        raise ValueError("--tile currently supports --input-lr only. Use --input-lr with an arbitrary-size LR image.")
    if args.tile and args.num_samples != 1:
        raise ValueError("--tile currently supports --num-samples 1.")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_autoencoder(config, device, dtype_name)
    condition_encoder = load_condition_encoder(config, device, dtype_name)
    model, checkpoint_step = load_unet(config, checkpoint_path, device, dtype_name)
    scheduler = NoiseScheduler.from_config(config.get("diffusion", {}))
    start_timestep = resolve_start_timestep(config, args.start_timestep)

    if args.tile:
        data_scale = int(data_config.get("scale", 4))
        tile_lr_size = int(data_config["hr_size"]) // data_scale
        lr_image = Image.open(args.input_lr).convert("RGB")
        lr_image.save(args.output_dir / "input_lr.png")
        print(
            f"checkpoint_step={checkpoint_step} lr_size={lr_image.size} steps={steps} "
            f"eta={args.eta} init={init} start_timestep={start_timestep} device={device} "
            f"progress_every={args.progress_every}",
            flush=True,
        )
        output = tiled_sample(
            model=model,
            vae=vae,
            condition_encoder=condition_encoder,
            scheduler=scheduler,
            lr_image=lr_image,
            domain_id_value=int(domains[args.domain]),
            scale=data_scale,
            tile_lr_size=tile_lr_size,
            overlap_lr=int(args.tile_overlap),
            tile_batch_size=int(args.tile_batch_size),
            steps=steps,
            eta=args.eta,
            init=init,
            start_timestep=start_timestep,
            dtype_name=dtype_name,
            seed=args.seed,
            output_dir=args.output_dir,
            device=device,
            progress_every=args.progress_every,
            diffusion_config=config.get("diffusion", {}),
        )
        output.save(args.output_dir / "sr_00.png")
        print(f"saved {args.output_dir}", flush=True)
        return

    lr_image, gt_image = prepare_inputs(args, config)
    lr_image.save(args.output_dir / "input_lr.png")
    if gt_image is not None:
        gt_image.save(args.output_dir / "gt_hr.png")

    lr_tensor = pil_to_tensor(lr_image).unsqueeze(0).repeat(args.num_samples, 1, 1, 1).to(device)
    domain_id = torch.full((args.num_samples,), int(domains[args.domain]), device=device, dtype=torch.long)
    print(
        f"checkpoint_step={checkpoint_step} lr_size={lr_image.size} steps={steps} "
        f"eta={args.eta} init={init} start_timestep={start_timestep} "
        f"samples={args.num_samples} device={device} progress_every={args.progress_every}",
        flush=True,
    )

    output = ddim_sample(
        model=model,
        vae=vae,
        condition_encoder=condition_encoder,
        scheduler=scheduler,
        lr=lr_tensor,
        domain_id=domain_id,
        steps=steps,
        eta=args.eta,
        init=init,
        start_timestep=start_timestep,
        dtype_name=dtype_name,
        seed=args.seed,
        output_dir=args.output_dir,
        save_every=args.save_every,
        progress_every=args.progress_every,
        progress_label="sample",
        diffusion_config=config.get("diffusion", {}),
    )
    for index, image in enumerate(output):
        tensor_to_pil(image).save(args.output_dir / f"sr_{index:02d}.png")
    print(f"saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
