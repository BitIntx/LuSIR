# LuSIR: Latent Upscaling via Self-trained Image Restoration without T2I Pretraining

Snapshot: formal x4 benchmark complete, Stage 2 clean-fidelity learning-rate
probes complete, and signed-wavelet residual diffusion evaluated and rejected
as the current generative-detail objective.

`paper/TECHNICAL_REPORT.md` is the canonical report source.
`paper/sr_diffusion_report.pdf` and `paper/main.tex` are generated from it with
`paper/build_report.sh`, so the Markdown and PDF contain the same report.

## Objective

LuSIR trains a vision-only x4 latent diffusion super-resolution model
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

The numbered stages describe training order, not a mandatory Stage 1 -> 2 -> 3
-> 4 runtime chain. The current runtime choices are:

```text
public Colab default:
  LR -> Stage 2 XL condition encoder -> residual refiner v2 -> Stage 1 decoder

current detail research candidate:
  LR -> dual-context LSDIR Stage 2 -> Stage 1 decoder -> detail branch v1d

generative comparison:
  LR -> Stage 2 condition encoder -> Stage 3 OR Stage 4 diffusion U-Net
     -> Stage 1 decoder
```

Stage 4 is a Stage 3-derived replacement diffusion checkpoint. It does not run
after Stage 3. Planned Stage 5 distillation would similarly replace the slower
diffusion sampler rather than append another serial module.

## Data

The base photo100k split has 103,450 training images and 100 fixed validation
images. The completed dual-context Stage 2 scale-up adds 30,000 unique LSDIR
training images, for 133,450 unique training images and the unchanged 100-image
validation set. LR inputs are generated on the fly from HR crops. Earlier XL
work used `photo_v3_noise_mix`, a strong denoise-focused curriculum with no
clean share and 80% combined v2/v3 cases. Later work introduced
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

## Exploratory Strict-Bicubic Five-Image Comparison

The ordinary LuSIR validation metrics use task-specific degradations, so their
absolute PSNR values should not be compared directly with published bicubic-only
SR benchmarks. A small strict-bicubic diagnostic was added to separate
degradation difficulty from reconstruction capacity. The new
`benchmark_bicubic` preset applies only PIL bicubic factor-4 downsampling and
does not add blur, noise, compression, color changes, or sharpening.

The shared diagnostic protocol is:

```text
source images: DIV2K validation 0801-0805
HR input: deterministic center 512x512 crop
LR input: strict PIL bicubic x4 downsample to 128x128
metric: mean per-image RGB PSNR over the full 512x512 crop
deterministic inference: CPU FP32
diffusion inference: condition initialization, start timestep 50, 32 DDIM steps
```

Loaded parameter counts include the entire 21.10M-parameter Stage 1 VAE because
the current runners load the full module, although inference executes only its
10.52M-parameter decoder. This matches the existing `509.658M` Stage 4 XL
accounting.

| Inference path | Selected checkpoint | Loaded params | Mean RGB PSNR | vs bicubic |
| --- | ---: | ---: | ---: | ---: |
| Bicubic | n/a | n/a | `29.5999` | n/a |
| Stage 2 XL condition-only | step 72000 | `40.040M` | `30.5677` | `+0.9678` |
| Stage 2 XL + residual refiner v2 | step 39000 | `48.106M` | `30.8205` | `+1.2206` |
| Stage 2 multiscale | step 46000 | `76.591M` | `31.6068` | `+2.0069` |
| Stage 2 dual-context LSDIR | step 98000 | `140.334M` | `31.7411` | `+2.1412` |
| Dual-context + detail v1b | step 39500 | `141.675M` | `31.8135` | `+2.2136` |
| Dual-context + detail v1c | step 6000 | `141.689M` | `31.8154` | `+2.2155` |
| Dual-context + detail v1d | step 99500 | `143.354M` | `31.9513` | `+2.3514` |
| Stage 4 XL edge, 32-step sampled | step 4250 | `509.658M` | `29.5487` | `-0.0512` |

The strict-bicubic result confirms that LuSIR already operates in the 30-32 dB
range when the LR input is not additionally corrupted. The much lower
`photo_detail_mix` and strong-preset values primarily reflect degradation
difficulty rather than a direct 6-7 dB gap to bicubic-only SR papers.

The capacity trend is also informative. Stage 2 XL to multiscale gains
`+1.0391 dB`, and multiscale to dual-context gains another `+0.1343 dB`.
The detail branches then provide smaller but consistent gains over dual-context:
v1b `+0.0725 dB`, v1c `+0.0744 dB`, and the completed 3.02M v1d
`+0.2102 dB`. V1d improves v1c by `+0.1358 dB` after three epochs. This
confirms that the long capacity run was useful, but the visual change remains
conservative and is not a perceptual-detail breakthrough.

The 509.7M Stage 4 XL edge checkpoint is the clearest counterexample to treating
parameter count as a quality ranking. On this clean bicubic diagnostic it falls
`-0.0512 dB` below bicubic and `-1.0190 dB` below its own Stage 2 XL
condition-only path. Visual inspection shows that it can add apparent
sharpness, but it also changes clean input details enough to reduce
reconstruction fidelity. This is consistent with its strong-noise
denoise/color-cleanup training role; it should not be routed unconditionally
for clean inputs.

This diagnostic is deliberately small and is not a published-benchmark claim.
It uses five center crops, RGB PSNR, no border shave, and PIL bicubic rather
than the full-image Y-channel/Matlab-bicubic conventions used by many SR
papers. The formal full-image benchmark described next supersedes it for
clean-bicubic fidelity comparison.

The machine-readable snapshot is stored in:

```text
metrics/benchmark_bicubic5_lusir_model_comparison.json
```

## Formal Full-Image x4 Benchmark

A formal evaluator now measures full-image DIV2K validation, Set5, Set14, and
Urban100 using their public x4 bicubic LR pairs. It uses MATLAB-compatible
BT.601 Y conversion, a four-pixel border shave, and MATLAB-style SSIM. All
candidate outputs are required to match the HR dimensions exactly; the
evaluator never hides an error by resizing an output.

| Candidate | DIV2K | Set5 | Set14 | Urban100 |
| --- | ---: | ---: | ---: | ---: |
| Bicubic | 28.1044 | 28.4318 | 26.0928 | 23.1412 |
| RealESRNet x4plus | 28.8250 | 29.3828 | 27.1465 | 24.4613 |
| RealESRGAN x4plus | 26.6125 | 26.6160 | 25.4216 | 22.6709 |
| SwinIR classical x4 | **31.0838** | not run | not run | not run |
| LuSIR refiner v2 | 28.7857 | 28.1896 | 27.3704 | 24.9176 |
| LuSIR dual-context base | 29.9575 | 31.6621 | 28.2441 | 25.4816 |
| **LuSIR detail v1d** | **30.1602** | **31.8892** | **28.4123** | **25.8755** |

The table reports Y PSNR; full Y SSIM and RGB PSNR values are preserved in the
machine-readable results. V1d improves the frozen dual-context base on all four
datasets. Its Y PSNR gains are `+0.2027`, `+0.2271`, `+0.1682`, and
`+0.3939 dB`, respectively. Its Y SSIM is `0.83421`, `0.89440`, `0.77998`,
and `0.77875`. This validates the branch redesign under a standard full-image
protocol, with the strongest gain on texture-heavy Urban100.

The official SwinIR classical x4 checkpoint reaches `31.0838 / 0.85228` on
the same DIV2K evaluator. It is `+0.9235 dB` Y PSNR and `+0.01807` Y SSIM
ahead of detail v1d. The detail branch is doing useful corrective work, but
this remaining gap identifies the Stage 2/base reconstruction path as the
primary clean-fidelity bottleneck.

RealESRNet and RealESRGAN target real-world degradation and perceptual quality,
so their clean-bicubic fidelity scores do not establish a general visual
quality ranking. Likewise, beating these checkpoints under this protocol does
not establish classical-SR SOTA. A classical fidelity baseline, perceptual
metrics, real-degradation evaluation, and blind human review remain necessary.
The reproducible protocol and commands are in `docs/SR_BENCHMARK.md`; full
machine-readable results are in:

```text
metrics/formal_x4_benchmark_lusir_realesr_summary.json
metrics/formal_x4_benchmark_lusir_realesr_metrics.csv
metrics/formal_x4_benchmark_div2k_swinir_summary.json
metrics/formal_x4_benchmark_div2k_swinir_metrics.csv
```

## Stage 2 Clean-Fidelity Continuation and Learning-Rate Probes

The clean-bicubic continuation starts from dual-context Stage 2 best98000 and
uses `benchmark_bicubic` with a PSNR-oriented decoded loss balance. Its
task-specific val100 decoded-PSNR proxy improved gradually:

| Step | Decoded PSNR proxy |
| ---: | ---: |
| 1000 | `25.019` |
| 4000 | `25.031` |
| 9000 | `25.045` |
| 15000 | **`25.057`** |
| 17000 | `25.054` |

The run was stopped manually at step 17825 after no new best beyond step
15000. Its checkpoints remain available for future clean-fidelity work.

These values are not directly comparable with the formal full-image Y-channel
results. The formal DIV2K comparison remains LuSIR detail v1d at `30.1602 dB`
and SwinIR classical x4 at `31.0838 dB`.

Separate learning-rate probes preserved the original step-15000 checkpoint.
A `20x` continuation collapsed to `15.72 dB` at its first evaluation. A `5x`
continuation remained below the original run, while a `5x` from-initialization
run reached `25.033 dB` at step 4000 versus `25.031 dB` for the original
learning rate. The difference is too small to treat as a gain. The original
`5e-6` learning rate is retained, and learning-rate scarcity is rejected as
the main bottleneck.

The result also clarifies the objective boundary. Same-objective Stage 2
continuation can refine deterministic fidelity, but it is unlikely to produce
the visibly new fine texture expected from a generative model.

The first separate generative probe is
`configs/diffusion_photo130k_lsdir_highfreq_residual_v1_b8.yaml`. It freezes the
dual-context Stage 2 base and Stage 1 decoder, then trains a 76.6M
gated/bounded residual U-Net with strong high-frequency supervision and a
lowpass anchor. The output layer is zero-initialized: the smoke-test prediction
and condition images were byte-identical, with zero latent residual and zero
decoded drift before the first optimizer update.

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
W&B: https://wandb.ai/jwheo/LuSIR/runs/lrb6nco9
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
W&B: https://wandb.ai/jwheo/LuSIR/runs/edfko8e8
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
W&B: https://wandb.ai/jwheo/LuSIR/runs/so0lbyte
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
A post-training residual-strength guardrail provides an explicit trade-off:
strength `1.0` gives the best average quality, `0.75` preserves most of the
gain with fewer regressions, and `0.5` raises strong-preset wins from `81/100`
to `86/100` while keeping positive mean gains. These modes are exposed in the
inference CLI and Colab. Future work should prioritize user-facing evaluation
and a learned degradation-aware gate rather than continuing indefinitely.

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

## Visual Review and External Positioning

A fixed-sample visual review now compares LR, bicubic, Stage 2 condition,
residual strengths `0.50`, `0.75`, and `1.00`, and GT side by side. On mild
inputs the selected refiner makes small, generally structure-preserving
improvements. On stronger `photo_v2` and `photo_v3_noise_mix` inputs, the main
failure occurs earlier: the condition encoder removes noise together with real
detail and sometimes leaves small cyan/white grid-like artifacts. The residual
refiner changes are too small to recover the missing fur, leaf veins, text, and
distant structural detail.

Qualitatively, the current model is not competitive with leading generative
restoration systems in perceived sharpness or plausible fine texture. Its
useful distinction is a deterministic, vision-only path trained without a
pretrained text-to-image model, with lower hallucination risk and adjustable
correction strength. This is an architectural and visual assessment, not a
direct SOTA benchmark. A defensible comparison requires same-input blind A/B
testing plus LPIPS/DISTS/MANIQA/MUSIQ and user-preference evaluation.

### Representative Visual Examples

The following two examples use the same lime image so that the role of the
base reconstruction and the later detail branch can be inspected separately.
They are representative diagnostic examples, not cherry-picked evidence of a
formal benchmark win.

![The selected multiscale Stage 2 model restores color and large structure from
the degraded LR input, but remains visibly smoother than ground truth on rind
texture.](../docs/assets/stage2_multiscale_demo.jpg)

*Figure 1. Multiscale Stage 2 restores the degraded input while retaining a
visible fine-texture gap to ground truth.*

![The selected v1d high-frequency detail branch makes a controlled texture correction
over the dual-context base. It remains artifact-light, but the gap to ground
truth fine detail is still clear.](../docs/assets/detail_branch_v1d_lime_demo.jpg)

*Figure 2. Selected detail branch v1d adds a controlled, artifact-light
correction over the base; this illustrates both its improvement and its
remaining conservative behavior.*

## Stage 2 Multiscale Redesign

A decoded pixel/edge/highpass fine-tuning probe confirmed that the flat Stage 2
condition predictor is the current bottleneck. By step 4000, decoded PSNR had
risen from `23.7387` to `24.2792`, but the Laplacian detail ratio fell from
`0.2817` to `0.2773` and fixed samples remained visibly smooth. This is
consistent with the distortion-perception limitation of optimizing
reconstruction losses without sufficient spatial context or high-quality
training exposure.

The replacement keeps all parameters and names of the selected 19M-parameter
step 72000 predictor, then adds a zero-output-initialized 128-to-64-to-32
multiscale context branch. Partial initialization therefore exactly preserves
the previous output before training. The training manifest also changes from
100,000 COCO plus 3,450 DIV2K/Flickr2K rows to a roughly balanced exposure of
100,000 COCO and 103,500 repeated DIV2K/Flickr2K rows. Random crops and
degradations remain stochastic, so repeated high-quality rows generate
different training pairs.

The design is informed by broad-context and multiscale restoration findings in
[SwinIR](https://arxiv.org/abs/2108.10257),
[HAT](https://arxiv.org/abs/2309.05239), and
[NAFNet](https://arxiv.org/abs/2204.04676), together with the data/degradation
emphasis of [Real-ESRGAN](https://arxiv.org/abs/2107.10833). The resulting
55.50M-parameter model passed a full batch-8 forward/backward and val100 smoke
test on one L40S at approximately `34.8/46.1GB` VRAM and `100%` GPU
utilization. The 50,000-micro-step long run is tracked at
<https://wandb.ai/jwheo/LuSIR/runs/6zt2do4v>.

The run completed normally. Step 46000 was selected over the final checkpoint
because it provides the best cross-preset distortion/detail compromise. It
improves the previous Stage 2 by `+1.0348 dB` on `photo_detail_mix` and
`+0.9228 dB` on `mild`, with `99/100` and `97/100` per-image wins and slightly
higher detail ratios. On the stronger `photo_v2` and `photo_v3_noise_mix`
presets it improves PSNR by `+0.9441 dB` and `+0.9650 dB`, but the detail ratio
drops from roughly `0.29-0.30` to `0.22-0.23`. Visual comparison confirms that
the model improves color, large boundaries, and denoising without convincingly
recovering missing fur, text, fabric, or distant texture. The experiment is a
successful base-reconstruction redesign but not a solution to perceptual
fine-detail recovery.

An optional continuation completed from step 46000 using frozen ImageNet
VGG16 features at shallow and intermediate layers. This explicitly introduces
pretrained vision feature supervision, while still avoiding pretrained
text-to-image or generative models. A CUDA smoke test passed at batch 4 with
approximately `20.6/46.1GB` VRAM and `2.62` micro-steps/s. The run completed
12,000 micro-steps. Step 8000 had the best shortlist score, `26.0092`, and
improved decoded PSNR over initialization by `+0.0101` to `+0.0256 dB` across
the four tested presets. Step 11000 was strongest on cleaner presets but
regressed `photo_v3_noise_mix` by `-0.0063 dB`. Fixed contact sheets showed
almost no visible difference from initialization, and missing fine texture
remained smoothed. The run is preserved as a partial metric/latent success but
is not promoted into the public path. The completed run is tracked at
<https://wandb.ai/jwheo/LuSIR/runs/nrqhw05u>.

## Dual-Context Unique-Data Stage 2 Scale-up

The VGG continuation's `+0.0101` to `+0.0256 dB` cross-preset improvements were
not visually distinguishable and did not recover missing texture. Capacity and
data uniqueness are therefore changed together instead of extending the same
objective again. The previous HQ-balanced manifest had 203,600 rows but only
103,550 unique images because DIV2K and Flickr2K were repeatedly exposed.

The new manifest adds 30,000 unique LSDIR images for 133,450 unique training
images plus the unchanged 100-image validation set. The predictor retains the
selected 55.50M-parameter multiscale model and adds a second zero-output-
initialized multiscale context branch. Partial initialization exactly preserves
the selected step 46000 output before training while increasing capacity to
119.24M parameters.

The one-L40S smoke test reproduced the initial `24.48 dB` decoded PSNR and
`0.291` detail ratio. Batch 8 with gradient accumulation 4 used approximately
`37.8/46.1GB` VRAM, reached `99%` GPU utilization, and sustained approximately
`0.75` micro-step/s. The completed long run targeted 100,000 micro-steps, or
25,000 optimizer updates. Evaluation ran every 1,000 micro-steps, while
ordinary milestone checkpoints were limited to every 5,000 micro-steps to stay
within the disk budget. The run is tracked at
<https://wandb.ai/jwheo/LuSIR/runs/4akqckxu>.

The run completed all 100,000 micro-steps. Step 98,000 is the automatic
`eval/decoded_psnr` best checkpoint on `photo_detail_mix`; step 100,000 is
slightly better on stronger degradations. Re-evaluated with the same comparison
tool against selected step 46,000, the gains are `+0.1362 dB` on
`photo_detail_mix`, `+0.1086 dB` on `mild`, `+0.0540 dB` on `photo_v2`, and
`-0.0356 dB` on `photo_v3_noise_mix` for the best checkpoint. The final
checkpoint changes those to `+0.1256`, `+0.1025`, `+0.0668`, and `+0.0132 dB`.
This is a modest reconstruction improvement, not a solved perceptual-detail
problem.

## High-Frequency Detail Branch v1

The latest deterministic detail candidate is an image-space detail branch, not
another Stage 4 continuation. The path is:

```text
LR -> frozen Stage 2 dual-context condition -> frozen Stage 1 decoder -> base SR
base SR + bicubic LR upsample -> gated high-frequency detail branch -> detail SR
```

The branch predicts a bounded RGB residual and a per-pixel gate. The residual is
projected through a local highpass operation before being added to the base SR,
which limits the model's ability to change global color or low-frequency
structure. The output convolution is zero-initialized, so step 0 exactly
reproduces the frozen Stage 2 + Stage 1 base output.

Implemented files:

```text
configs/detail_branch_v1_photo130k_lsdir.yaml
configs/detail_branch_v1b_aug_photo130k_lsdir.yaml
configs/detail_branch_v1c_condition_open_photo130k_lsdir.yaml
configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
configs/hf/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
tools/train/train_detail_branch.py
tools/eval/run_fixed_review_detail_branch.py
tests/test_detail_branch.py
```

A four-micro-step smoke test completed load/eval/backprop/update/checkpoint
successfully. Step 0 matched the base output exactly at `24.6188 dB` val100
PSNR; after one optimizer update, the branch produced a tiny `+0.00005 dB`
aggregate PSNR delta and `69/100` wins versus base. This is only an integration
check, not a quality claim.

The first v1 run was stopped at 7800 micro-steps, or 0.234 epoch over the
133450-image train manifest, because early samples showed that the residual was
safe but visually too small. The v1b run kept the same model and loss, but added
horizontal flips, texture-biased crop retry, and weak HR color jitter while
excluding rotation, vertical flip, affine/perspective transforms, erasing, and
mixup.

The v1b run completed 40000 micro-steps, equal to 10000 optimizer updates with
`grad_accum_steps: 4`. It selected step 39500 by `eval/detail_score`:

```text
base PSNR:        24.6188
detail PSNR:      24.6649
PSNR delta:       +0.0461 dB
base SSIM:        0.80013
detail SSIM:      0.80281
SSIM delta:       +0.00268
mean PSNR delta:  +0.0575
wins vs base:     98/100
detail wins:      100/100
```

Different metrics peak at nearby checkpoints: PSNR delta is highest at step
38500 (`+0.0489 dB`), SSIM delta is highest at step 37000 (`+0.00336`), and the
final step 40000 reaches `+0.0444 dB` PSNR and `+0.00277` SSIM with `98/100`
wins. The selected step 39500 remains preserved as the earlier public detail
artifact because it has the best combined detail score from the completed v1b
run. Qualitatively, it is artifact-light and
slightly sharper on texture-heavy crops, but still conservative and below GT on
fine surface detail. It remains a historical comparison artifact; the later
selected v1d checkpoint is exposed through the Colab WebUI detail runner.

V1c initialized from the selected v1b checkpoint, exposed the frozen Stage 2
condition latent directly to the image-space branch, increased the bounded
residual scale, and opened the gate slightly. The selected step 6000 improves
the fixed `photo_detail_mix` val100 base by `+0.0554 dB` aggregate PSNR and
`+0.00332` SSIM with `99/100` wins. This was better than v1b but plateaued
quickly enough to motivate a controlled capacity test.

V1d keeps the v1c width and objective but increases residual depth from 8 to
18 blocks, raising branch capacity from `1.35M` to `3.02M` parameters. The
original blocks are copied from v1c step 6000 and the ten appended blocks are
identity-initialized, so the step-0 output exactly reproduces v1c. The run
completed 100,086 micro-steps, exactly three passes over the 133,450-image
manifest at batch size 4, and selected step 99,500 by `eval/detail_score`.

On ordinary fixed `photo_detail_mix` val100, selected step 99,500 improves the
frozen base by `+0.1646 dB` aggregate PSNR, `+0.1888 dB` mean PSNR, and
`+0.00647` SSIM with `99/100` PSNR wins and `100/100` detail wins. On the
strict-bicubic five-crop diagnostic it reaches `31.9513 dB`, improving the
frozen base by `+0.2102 dB`, v1c by `+0.1358 dB`, and winning `5/5` images.
This is also `+0.1266 dB` above the early v1d step-9500 snapshot.

Final step 100,086 reaches a nearly identical strict-bicubic `31.9516 dB`, but
selected step 99,500 is stronger on ordinary-val aggregate PSNR, SSIM,
highpass improvement, and the combined detail score. Visual inspection shows
no white-dot, grid, or excessive-sharpening artifacts. The larger branch and
long run therefore produced meaningful stable progress, but still did not
close the visible fine-texture gap to ground truth. Further same-objective
continuation or simple capacity scaling is not planned.

## Public Artifacts

The latest LuSIR public artifacts are stored in `jwheo/LuSIR` on Hugging Face:

```text
checkpoints/stage4_photo100k_xl_edge_b16_best_eval_condition_decoded.pt
checkpoints/residual_refiner_stage2_xl_mild_best_eval_refined.pt
checkpoints/stage4_photo100k_xl_teacher_residual_photo_v3_step_0002000.pt
checkpoints/stage4_photo100k_xl_teacher_residual_photo_detail_best8000.pt
checkpoints/residual_refiner_stage2_xl_photo_detail_v2_best39000.pt
checkpoints/stage2_photo100k_multiscale_hqmix_step_0046000.pt
checkpoints/stage2_photo100k_multiscale_hqmix_perceptual_step_0008000.pt
checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
checkpoints/detail_branch_v1b_aug_photo130k_lsdir_best39500.pt
checkpoints/detail_branch_v1d_deep3m_photo130k_lsdir_best99500.pt
metrics/stage4_photo100k_xl_edge_b16_val100_t50_32step_summary.json
metrics/diagnose_stage2_xl_residuals_mild_val100_summary.json
metrics/residual_refiner_stage2_xl_mild_probe_early_stop_summary.json
metrics/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_summary.json
metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_32step_summary.json
metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t50_32step_summary.json
metrics/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_summary.json
metrics/residual_refiner_stage2_xl_photo_detail_v2_long_summary.json
metrics/residual_refiner_v2_best39000_strength_sweep_summary.json
metrics/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100_summary.json
metrics/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100_summary.json
metrics/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_summary.json
metrics/stage2_multiscale_hqmix_step46000_cross_preset_summary.json
metrics/stage2_multiscale_perceptual_photo_detail_mix_candidates.json
metrics/stage2_multiscale_perceptual_mild_candidates.json
metrics/stage2_multiscale_perceptual_photo_v2_candidates.json
metrics/stage2_multiscale_perceptual_photo_v3_noise_mix_candidates.json
metrics/stage2_photo130k_lsdir_dual_multiscale_final_summary.json
metrics/detail_branch_v1b_aug_photo130k_lsdir_summary.json
metrics/detail_branch_v1d_deep3m_photo130k_lsdir_3ep_summary.json
metrics/benchmark_bicubic5_detail_v1d_best99500_summary.json
samples/stage4_photo100k_xl_edge_b16_val100_t50_32step_grid_lr_bicubic_sr_gt.png
samples/diagnose_stage2_xl_residuals_mild_val100_grid.png
samples/residual_refiner_stage2_xl_mild_probe_step500_grid.png
samples/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_grid.png
samples/compare_residual_refiner_vs_stage4_edge_0801_photo_v3.png
samples/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_grid.png
samples/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_grid.png
samples/residual_refiner_stage2_xl_photo_detail_v2_best39000_grid.png
samples/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100_grid.png
samples/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100_grid.png
samples/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_grid.png
samples/stage2_multiscale_hqmix_checkpoint_comparison.png
samples/stage2_multiscale_perceptual_photo_detail_mix_candidates.png
samples/stage2_multiscale_perceptual_photo_v3_noise_mix_candidates.png
samples/stage2_dual_lsdir_photo_detail_mix_best98k_final100k_contact_sheet.png
samples/stage2_dual_lsdir_mild_best98k_final100k_contact_sheet.png
samples/stage2_dual_lsdir_photo_v2_best98k_final100k_contact_sheet.png
samples/stage2_dual_lsdir_photo_v3_noise_mix_best98k_final100k_contact_sheet.png
samples/detail_branch_v1b_aug_photo130k_lsdir_best39500_grid.png
samples/detail_branch_v1d_deep3m_photo130k_lsdir_best99500_grid.png
samples/benchmark_bicubic5_detail_v1d_best99500_grid.png
configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml
configs/residual_refiner_stage2_xl_mild_probe.yaml
configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml
configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml
configs/residual_refiner_stage2_xl_photo_detail_v2_continue_40k.yaml
configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml
configs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue.yaml
configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml
configs/detail_branch_v1_photo130k_lsdir.yaml
configs/detail_branch_v1b_aug_photo130k_lsdir.yaml
configs/hf/detail_branch_v1b_aug_photo130k_lsdir.yaml
configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
configs/hf/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
```

## Next Work

The selected residual refiner remains the public Colab default, while detail
branch v1d is available as a selectable single-image/tiled Colab research
option. The completed perceptual Stage 2 continuation is not promoted.
Candidate next steps are:

- preserve the deterministic base as the fidelity path and train a separate
  learned detail-need mask before applying stochastic texture synthesis;
- select the generative path with LPIPS, DISTS, fixed visual review,
  high-frequency metrics, lowpass drift, and seed diversity instead of PSNR
  alone;
- continue Stage 2/base architecture research as a separate clean-fidelity
  track for the measured `0.9235 dB` DIV2K gap, rather than mixing that goal
  into the generative detail objective;
- because v1d remains visually conservative and noise-MSE residual diffusion
  collapses toward zero, prioritize patch-level perceptual or adversarial
  supervision instead of increasing branch capacity again;
- keep a degradation-aware gate or strong-input guardrail as the primary
  response to the remaining strong-preset failure tail.

The first latent residual probe increased high-frequency energy but worsened
GT-aligned detail and was stopped. The replacement diffuses only signed Haar
high bands of the image-space residual over the frozen v1d base. Its clipped
oracle improves the val8 base from `28.7012` to `31.4039` dB, confirming that
the representation can express useful corrections. The step-20,000
condition-start continuation removed the early stochastic grain, but the
residual magnitude and seed diversity also collapsed toward zero. On val100,
start timesteps 15, 25, and 50 trail the v1d base by `0.0880`, `0.1392`, and
`0.3152` dB respectively, and all settings worsen GT-aligned Laplacian and
highpass error. The result is not promoted, and the same noise-MSE objective
will not be continued. The next generative detail experiment should use an
explicit learned detail-need mask plus patch-level perceptual or adversarial
supervision rather than relying on conditional-mean residual prediction.
Implementation and evaluation details are in
`docs/HIGH_FREQUENCY_RESIDUAL_DIFFUSION_KO.md`.
