# Vision-Only Latent Diffusion Super-Resolution without T2I Pretraining

Snapshot: Stage 4 XL gated-residual probe complete.

## Objective

This project trains a vision-only x4 latent diffusion super-resolution model
without using a pretrained text-to-image backbone. The active task is:

```text
LR 128x128 -> HR 512x512
```

The model path is:

```text
HR image -> factor-4 VAE -> HR latent
LR image -> condition encoder -> condition latent
noisy HR latent + condition latent + timestep + domain id -> conditional U-Net
denoised latent -> VAE decoder -> SR output
```

## Data

The current large photo split has 103,450 training images and 100 fixed
validation images. LR inputs are generated on the fly from HR crops. The latest
XL edge-loss work uses `photo_v3_noise_mix`, a stronger denoise-focused
degradation curriculum with mixed mild/v2/v3 noise cases. The latest
residual-refinement probes also evaluate a mild val100 setting to isolate
whether Stage 4 adds detail beyond the Stage 2 condition-only output.

## Completed Baselines

| Stage | Run | Result |
| --- | --- | --- |
| Stage 1 VAE | `autoencoder_photo10k_b16_eval_online` | `eval/psnr 40.19` |
| Stage 2 photo100k | `latent_pretrain_photo100k_b64` | best latent loss `0.21230` |
| Stage 3 photo100k | `diffusion_photo100k_b32` | sampled val100 `25.3745` PSNR |
| Stage 4 photo100k | `diffusion_photo100k_b32_stage4_condition` | sampled val100 `25.4072` PSNR |
| Stage 4 photo100k v2 | `diffusion_photo100k_b32_stage4_condition_v2` | sampled val100 `22.8426` PSNR, `+0.4323` over bicubic |

The v2 task has a stronger degradation distribution, so its absolute PSNR is
not directly comparable with the earlier mild photo100k run.

## Stage 2 XL Candidate Selection

The XL condition encoder uses:

```text
config: configs/latent_pretrain_photo100k_v3_noise_xl.yaml
base_channels: 256
num_blocks: 16
degradation: photo_v3_noise_mix
finished step: 80000
```

Candidate comparison on the same validation images:

| Candidate | Step | Latent loss | Latent MSE | Decoded PSNR |
| --- | ---: | ---: | ---: | ---: |
| `best_eval_latent` | 66000 | `0.27230` | `0.91770` | `21.3828` |
| `step_0072000` | 72000 | `0.27940` | `0.97295` | `21.5241` |
| `latest` | 80000 | `0.27593` | `0.89609` | `21.5062` |

The 72k checkpoint was selected for Stage 4 XL because it had the best decoded
condition-only PSNR on the fixed validation set.

## Stage 4 XL Edge-Loss Result

The first XL diffusion run used the selected Stage 2 XL condition encoder and
partial initialization from the smaller Stage 4 v2 checkpoint.

```text
config: configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml
run: diffusion_photo100k_xl_stage4_condition_v3_edge_b16
U-Net path: 469.6M parameters
full inference path: 509.658M parameters
train batch size: 16
GPUs: 2x A100-SXM4-80GB through PyTorch DDP
finished step: 5000
selected checkpoint: step 4250, best eval/decoded_mse
```

Training-time best proxy:

```text
step 4250
eval/decoded_psnr: 21.9872
eval/decoded_mse: 0.025313
eval/noise_mse: 27.66008
eval/x0_mse: 0.86226
```

Sampled validation evaluation:

```text
checkpoint: best_eval_condition_decoded.pt
checkpoint step: 4250
split: val
limit: 100
init: condition
start_timestep: 50
steps: 32
mean_bicubic_psnr: 22.3599
mean_sr_psnr: 23.0793
mean_psnr_delta: +0.7195
```

The latest XL Stage 4 checkpoint is therefore better than bicubic on the
current v3 validation setup and better aligned with the desired denoise/color
cleanup behavior than the Stage 2 condition-only path. It is still not a final
restoration model: outputs remain softer than GT on fine textures.

## Stage 4 XL Role-Split and Gated-Residual Probes

Follow-up probes tested whether Stage 4 could act as a bounded detail refiner
instead of freely overwriting the deterministic Stage 2 condition latent.

The role-split mild probe added a lowpass anchor and separated low/high-frequency
decoded losses:

```text
config: configs/diffusion_photo100k_xl_stage4_condition_v3_rolesplit_mild_b8_probe.yaml
run: diffusion_photo100k_xl_stage4_condition_v3_rolesplit_mild_b8_probe
W&B: https://wandb.ai/jwheo/sr-diffusion/runs/lrb6nco9
finished step: 8000 micro-steps, 2000 optimizer updates
best one-step checkpoint: step 7500, decoded PSNR 23.2515
```

Sampled mild val100 showed that role-split losses reduced some over-editing at
very low start timesteps, but the diffusion path still damaged the Stage 2
condition output on average:

| Model | Start timestep | Mean SR PSNR | vs condition-only | Wins vs condition |
| --- | ---: | ---: | ---: | ---: |
| Stage 2 XL condition-only | n/a | `25.0449` | n/a | n/a |
| Stage 4 role-split | 25 | `24.5747` | `-0.4702` | `3/100` |
| Stage 4 role-split | 10 | `24.9185` | `-0.1264` | `3/100` |
| Stage 4 role-split | 5 | `24.9935` | `-0.0514` | `6/100` |
| Stage 4 role-split | 1 | `25.0335` | `-0.0114` | `10/100` |

The next probe changed the diffusion prediction parameterization to
`gated_residual_x0`. Instead of predicting the full denoised latent, the U-Net
predicts a bounded residual and gate on top of the condition latent:

```text
x0_hat = condition + residual_scale * tanh(residual_logits) * sigmoid(gate_logits + gate_bias)
residual_scale: 1.25
gate_bias: 0.0
out_channels: 32
```

The probe was initialized from the role-split best checkpoint with partial
initialization and stopped at step 2000 after the sampled validation result
reached condition-only parity:

```text
config: configs/diffusion_photo100k_xl_stage4_condition_v3_gated_residual_mild_b8_probe.yaml
run: diffusion_photo100k_xl_stage4_condition_v3_gated_residual_mild_b8_probe
W&B: https://wandb.ai/jwheo/sr-diffusion/runs/edfko8e8
best one-step checkpoint: step 1000
final checkpoint: step 2000
```

Sampled mild val100:

| Model | Checkpoint | Start timestep | Mean SR PSNR | vs condition-only | Wins vs condition |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 2 XL condition-only | n/a | n/a | `25.0449` | n/a | n/a |
| Stage 4 gated residual | 1000 | 25 | `25.0415` | `-0.0035` | `25/100` |
| Stage 4 gated residual | 1000 | 10 | `25.0415` | `-0.0034` | `25/100` |
| Stage 4 gated residual | 1000 | 5 | `25.0416` | `-0.0034` | `25/100` |
| Stage 4 gated residual | 1000 | 1 | `25.0418` | `-0.0032` | `25/100` |
| Stage 4 gated residual | 2000 | 25 | `25.0445` | `-0.0004` | `34/100` |
| Stage 4 gated residual | 2000 | 10 | `25.0444` | `-0.0006` | `32/100` |
| Stage 4 gated residual | 2000 | 5 | `25.0444` | `-0.0005` | `31/100` |
| Stage 4 gated residual | 2000 | 1 | `25.0443` | `-0.0007` | `32/100` |

This is a partial success. The gated residual parameterization almost eliminated
the condition damage seen in previous Stage 4 probes and increased the number of
samples beating condition-only from `10/100` at role-split t1 to `34/100` at
gated-residual step2000 t25. However, the mean result still does not beat the
Stage 2 condition-only baseline. The current interpretation is that Stage 4 has
learned a safe near-identity residual path, not a reliable missing-detail
generator.

## Systems Notes

Diffusion training now supports PyTorch DDP when launched with `torchrun`.
Without `torchrun`, the same script falls back to the existing single-GPU path.
On the tested 2x A100 SXM environment, a 1 GiB NCCL all-reduce smoke test
reported about `199.5 GB/s`, so multi-GPU communication was not the bottleneck.

The completed Stage 4 XL edge run trained at about `0.78 step/s` in ordinary
train sections. A 5000-step run is roughly 1.8 hours of pure training, plus
eval/checkpoint overhead.

On the tested 2x L40S environment, the gated-residual probe ran with high GPU
utilization, about `0.79` micro-step/s, and roughly `44.7 / 46.1 GiB` allocated
per GPU. No GPU bottleneck or NCCL communication issue was observed.

## Public Artifacts

The latest public artifacts are stored in `jwheo/sr-diffusion` on Hugging Face:

```text
checkpoints/stage4_photo100k_xl_edge_b16_best_eval_condition_decoded.pt
metrics/stage4_photo100k_xl_edge_b16_val100_t50_32step_summary.json
samples/stage4_photo100k_xl_edge_b16_val100_t50_32step_grid_lr_bicubic_sr_gt.png
configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml
```

## Next Work

The highest-signal next direction is not a longer continuation of the same
Stage 4 loss. Gated residual x0 prediction shows that structural constraints can
protect the Stage 2 condition output, but the model still needs a more direct
signal for where residual detail should be added. Candidate next steps are:

- train a deterministic bounded residual refiner first, then distill or warm
  start the diffusion residual path from it;
- add direct residual/gate supervision based on condition-vs-target error;
- add a condition uncertainty or detail-need map so Stage 4 edits only locations
  where the Stage 2 condition encoder is likely missing recoverable detail;
- revisit the Stage 2 degradation curriculum if the condition encoder remains
  the dominant quality ceiling.
