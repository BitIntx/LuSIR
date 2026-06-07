# Vision-Only Latent Diffusion Super-Resolution without T2I Pretraining

Snapshot: detail-preserving Stage 4 curriculum adaptation complete.

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
validation images. LR inputs are generated on the fly from HR crops. Earlier XL
work used `photo_v3_noise_mix`, a strong denoise-focused curriculum with no
clean share and 80% combined v2/v3 cases. The latest work introduces
`photo_detail_mix`: 35% clean, 48% detail-preserving photo degradation, 15%
mild degradation, and 2% strong `photo_v2`. This keeps a small robustness tail
while shifting the primary objective toward user-facing detail restoration.

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

## Stage 2 Residual Diagnostic and Deterministic Refiner

A direct residual/oracle diagnostic was run to separate the Stage 2
condition-only error into lowpass and highpass components on mild val100:

| Metric | Value |
| --- | ---: |
| Bicubic PSNR | `24.4778` |
| Condition decoded PSNR | `25.0543` |
| Oracle full residual PSNR | `41.8207` |
| Oracle full vs condition | `+16.7664` |
| Oracle highpass PSNR | `35.0872` |
| Oracle highpass vs condition | `+10.0329` |
| Oracle lowpass PSNR | `25.0814` |
| Oracle lowpass vs condition | `+0.0270` |
| Residual highpass energy ratio | `0.8988` |
| Residual lowpass energy ratio | `0.0758` |

This indicates that the Stage 2 condition encoder already preserves most
structure and low-frequency color, while the remaining recoverable error is
dominated by high-frequency detail.

The follow-up deterministic bounded residual refiner freezes the Stage 1 VAE
and Stage 2 condition encoder, then predicts only:

```text
condition + residual_scale * tanh(residual_logits) * sigmoid(gate_logits + gate_bias)
```

The sparse-gate probe reached its best validation result at step 500:

| Model | Mean PSNR | vs condition | Wins vs condition | Gate mean |
| --- | ---: | ---: | ---: | ---: |
| Stage 2 condition-only | `25.0449` | n/a | n/a | n/a |
| Sparse-gate residual refiner | `25.1178` | `+0.0729` | `86/100` | `0.2147` |
| Open-gate residual refiner | `25.0972` | `+0.0523` | `73/100` | `0.8680` |

The sparse-gate result is a small but real gain and is qualitatively close to
the condition output, without the destructive edits seen in previous diffusion
Stage 4 probes. The open-gate ablation was worse despite a much larger mean
gate value, so the next step should not simply force larger residual edits.

The sparse-gate refiner was then connected to standalone eval and inference
scripts and evaluated without retraining across the active degradation presets:

| Degradation | Bicubic PSNR | Condition PSNR | Refined PSNR | vs condition | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mild` | `24.4778` | `25.0449` | `25.1178` | `+0.0729` | `86/100` |
| `photo_v2` | `22.4103` | `22.9271` | `22.9767` | `+0.0496` | `77/100` |
| `photo_v3_noise_mix` | `22.3599` | `22.9014` | `22.9600` | `+0.0586` | `86/100` |

Qualitatively, the refined outputs remain very close to the Stage 2 condition
outputs. This is useful because the refiner does not introduce the destructive
edits seen in earlier diffusion probes, but the visible detail gain is still
small. The result is best interpreted as a safe residual teacher or warm start,
not as a final detail generator.

## Teacher-Supervised Stage 4 Residual Probe

The sparse-gate deterministic refiner was used as a frozen teacher for the
gated-residual Stage 4 U-Net. Direct losses supervised the predicted residual,
highpass residual, and gate on `photo_v3_noise_mix`.

```text
config: configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml
train batch size: 8
gradient accumulation: 4
finished step: 8000 micro-steps, 2000 optimizer updates
```

Sampled val100, condition initialization, 32 sampling steps:

| Checkpoint | Start timestep | Mean SR PSNR | vs bicubic | vs condition | Wins vs condition |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher Stage 4 step 2000 | 25 | `22.9640` | `+0.6041` | `+0.0626` | `68/100` |
| Teacher Stage 4 step 2000 | 50 | `22.9639` | `+0.6040` | `+0.0625` | n/a |
| Teacher Stage 4 step 4000 | 25 | `22.9571` | `+0.5972` | `+0.0557` | `65/100` |
| Teacher Stage 4 step 8000 | 25 | `22.9490` | `+0.5891` | `+0.0476` | `59/100` |
| Existing edge Stage 4 step 4250 | 25 | `22.9563` | `+0.5964` | `+0.0549` | `42/100` |
| Existing edge Stage 4 step 4250 | 50 | `23.0799` | `+0.7200` | `+0.1784` | `45/100` |

Teacher supervision therefore produced a small, stable cleanup gain over the
Stage 2 condition output and slightly exceeded the edge model at t25. However,
the user-facing objective was not achieved. Fur, leaves, branches, and building
detail remained strongly smoothed. A simple absolute-Laplacian diagnostic
measured teacher step-2000 output at `21.8%` of GT detail energy, below the
existing edge t25 output at `32.7%`. This is not a complete perceptual metric,
but it agrees with visual inspection.

The best sampled checkpoint was step 2000; continuing to step 8000 reduced both
mean PSNR and condition win count. The active interpretation is that the
current `photo_v3_noise_mix` curriculum overemphasizes severe denoise/cleanup
cases, so direct residual supervision teaches a safer cleanup operator rather
than a missing-detail generator.

## Detail-Preserving Curriculum Adaptation

A fixed-sample degradation audit confirmed the curriculum mismatch. On val100,
`photo_v3_noise_mix` reduced the bicubic baseline to `22.3599` PSNR and produced
mean LR chroma RMS error `0.02040` relative to clean downsampling.
`photo_detail_mix` increased the bicubic baseline to `24.7357` and reduced the
chroma error to `0.00507`.

The existing Stage 2 XL condition encoder was evaluated before retraining. It
already improved `photo_detail_mix` from `24.7357` bicubic PSNR to `25.3103`,
a `+0.5745` gain. Stage 2 was therefore frozen, and the teacher-supervised
gated-residual Stage 4 was adapted from its selected step-2000 checkpoint:

```text
config: configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml
run: diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long
W&B: https://wandb.ai/jwheo/sr-diffusion/runs/so0lbyte
train batch size: 8
gradient accumulation: 4
learning rate: 1e-6
finished step: 12000 micro-steps, 3000 optimizer updates
```

Sampled `photo_detail_mix` val100, condition initialization, start timestep 25,
32 sampling steps:

| Model/checkpoint | Mean SR PSNR | vs bicubic | vs condition | Wins vs condition |
| --- | ---: | ---: | ---: | ---: |
| Stage 2 condition-only | `25.3103` | `+0.5745` | n/a | n/a |
| Teacher Stage 4 initialization | `25.3187` | `+0.5829` | `+0.0084` | `46/100` |
| Photo-detail Stage 4 step 8000 | `25.3406` | `+0.6049` | `+0.0303` | `71/100` |
| Photo-detail Stage 4 step 12000 | `25.3337` | `+0.5980` | `+0.0235` | `67/100` |
| Existing edge Stage 4 step 4250 | `25.1176` | `+0.3818` | `-0.1927` | `13/100` |

Step 8000 is the selected checkpoint. This is the first sampled gated-residual
Stage 4 result that beats the Stage 2 condition-only baseline on both mean PSNR
and a clear majority of samples. Qualitatively, it preserves the condition
structure and sharpness and avoids the broad destructive edits produced by the
edge model on this distribution.

The result remains conservative. Mean absolute-Laplacian energy is `29.7%` of
GT for step 8000, compared with `29.6%` for its teacher initialization and
`41.2%` for the more aggressive edge model. The gain therefore comes primarily
from more accurate bounded corrections, not from strong new texture synthesis.
Rare strong-tail samples can still exhibit bright artifacts.

## Decoded-Detail Residual Refiner v2

The deterministic bounded residual refiner uses 192 hidden channels, 12
residual blocks, and decoded-image/highpass supervision through the frozen VAE
decoder. An initial 12000-step run was continued to 40000 micro-steps with a
lower `2.5e-5` learning rate. Step 39000 achieved the best global decoded PSNR
and was selected.

| Degradation | Condition mean PSNR | Refined mean PSNR | Gain | Wins |
| --- | ---: | ---: | ---: | ---: |
| `photo_detail_mix` | `25.3103` | `25.6410` | `+0.3307` | `94/100` |
| `mild` | `25.0449` | `25.3161` | `+0.2712` | `91/100` |
| `photo_v2` | `22.9271` | `23.0419` | `+0.1148` | `81/100` |
| `photo_v3_noise_mix` | `22.9014` | `23.0787` | `+0.1773` | `81/100` |

On the training curriculum, step 39000 reached global decoded PSNR `24.0305`,
`+0.2927 dB` over condition-only, mean PSNR gain `+0.3307 dB`, SSIM gain
`+0.01076`, and wins on `94/100` images. Step 40000 had a slightly higher SSIM
gain (`+0.01161`) but lower PSNR and fewer wins, so step 39000 is the more
balanced public default.

The continuation improved mean PSNR on every tested degradation. However, the
strong `photo_v2` and `photo_v3_noise_mix` win counts fell versus step 11000,
indicating that the larger correction has a less conservative failure tail.
Future work should prioritize user-facing/detail-focused evaluation and a
strong-input guardrail rather than continuing the same training indefinitely.

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

The decoded-detail residual refiner v2 ran on one L40S at about `0.89`
micro-step/s with `42.0 / 46.1 GiB` allocated and sustained `99-100%` GPU
utilization.

## Public Artifacts

The latest public artifacts are stored in `jwheo/sr-diffusion` on Hugging Face:

```text
checkpoints/stage4_photo100k_xl_edge_b16_best_eval_condition_decoded.pt
checkpoints/residual_refiner_stage2_xl_mild_best_eval_refined.pt
checkpoints/stage4_photo100k_xl_teacher_residual_photo_v3_step_0002000.pt
checkpoints/stage4_photo100k_xl_teacher_residual_photo_detail_best8000.pt
checkpoints/residual_refiner_stage2_xl_photo_detail_v2_best39000.pt
metrics/stage4_photo100k_xl_edge_b16_val100_t50_32step_summary.json
metrics/diagnose_stage2_xl_residuals_mild_val100_summary.json
metrics/residual_refiner_stage2_xl_mild_probe_early_stop_summary.json
metrics/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_summary.json
metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_32step_summary.json
metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t50_32step_summary.json
metrics/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_summary.json
metrics/residual_refiner_stage2_xl_photo_detail_v2_long_summary.json
metrics/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_summary.json
samples/stage4_photo100k_xl_edge_b16_val100_t50_32step_grid_lr_bicubic_sr_gt.png
samples/diagnose_stage2_xl_residuals_mild_val100_grid.png
samples/residual_refiner_stage2_xl_mild_probe_step500_grid.png
samples/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_grid.png
samples/compare_residual_refiner_vs_stage4_edge_0801_photo_v3.png
samples/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_grid.png
samples/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_grid.png
samples/residual_refiner_stage2_xl_photo_detail_v2_best39000_grid.png
samples/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_grid.png
configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml
configs/residual_refiner_stage2_xl_mild_probe.yaml
configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml
configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml
configs/residual_refiner_stage2_xl_photo_detail_v2_continue_40k.yaml
```

## Next Work

The selected residual refiner is now the strongest public deterministic path.
Candidate next steps are:

- evaluate step 39000 on a separate user-facing/detail-focused image set;
- add LPIPS/DISTS-style perceptual metrics alongside the existing SSIM and
  explicit detail metrics;
- add a degradation-aware gate or strong-input guardrail for the lower-win
  `photo_v2` and `photo_v3_noise_mix` tails;
- compare the selected refiner as a deterministic final output and as a Stage 4
  teacher without extending the same continuation further.
