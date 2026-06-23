# LuSIR: Latent Upscaling via Self-trained Image Restoration without T2I Pretraining

Snapshot: masked detail branch v2 and its formal x4 benchmark are complete,
Stage 2 clean-fidelity learning-rate probes are complete, the Stage 2
detail-perceptual continuation and shifted-window attention probes have been
formally reviewed but not promoted, and several deterministic/generative
visible-detail probes were evaluated without producing a clear texture
breakthrough. A Stage 1 decoder capacity audit now points the active
visible-detail bottleneck back to Stage 2 conditional-latent smoothing. The
no-GAN, noise-gated v6 detail branch, Stage 2 latent residual adapter, masked
latent residual-shift, and teacher-filtered v7 detail probes are complete; none
is promoted. V7 used RealESRGAN only on local highpass patches that were
near/better than the base against GT, but step500 still regressed highpass and
laplacian detail ratios. A 256-image teacher patch diagnostic then showed that
RealESRGAN did not beat the base on PSNR or highpass-L1 for any cached sample,
so the v8 probe removed teacher supervision and used GT detail-need masks only
during training. V8 stayed stable and selected step 500, but visible detail
remained small and step2000 began to lose laplacian ratio, so it is not promoted.
The follow-up Stage 2 GT-masked detail probe moved the GT-missing-detail signal
into Stage 2 itself via a train-only masked decoded/highpass loss initialized
from guarded-detail Stage 2 v2 step 10000, but it was stopped at step1000 after
highpass ratio and missing-detail metrics worsened despite a small mean-PSNR
increase. A clean-bicubic overfit64 diagnostic is now separated from deployable
training: it fixes the first 64 train samples under deterministic bicubic
degradation and evaluates on the same set to test whether the current Stage 2
structure/loss can memorize high-frequency detail once generalization is
removed.

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
  LR -> guarded-detail Stage 2 v2 step 10000 -> Stage 1 decoder

Stage 2 GT-masked detail negative result:
  train: current decoded prediction/GT -> GT missing-detail top20 training mask
  eval:  LR -> GT-masked-detail Stage 2 v3 -> Stage 1 decoder

conservative deterministic option:
  LR -> Stage 2 XL condition encoder -> residual refiner v2 -> Stage 1 decoder

current detail research candidate:
  LR -> dual-context LSDIR Stage 2 -> Stage 1 decoder
     -> learned detail mask -> masked detail branch v2

teacher-filtered negative result:
  LR -> dual-context LSDIR Stage 2 -> Stage 1 decoder
     -> relaxed learned detail mask -> v7 teacher-filtered detail branch

GT-mask training diagnostic:
  train: base/GT -> GT detail-need top20 training mask
  eval:  LR -> dual-context LSDIR Stage 2 -> Stage 1 decoder
            -> learned detail mask -> v8 detail branch

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
| **LuSIR masked detail v2** | **30.1636** | **31.9495** | **28.4257** | **25.8922** |

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

The guarded-detail Stage 2 v2 step-10000 Colab default was also evaluated with
test-time augmentation on the same 219-image benchmark using a fast PSNR-only
sweep. Off, horizontal flip x2, and full x8 self-ensemble reach `27.8539`,
`27.9067`, and `27.9496 dB` mean Y PSNR. Full x8 is therefore a real but small
`+0.0957 dB` correction over off. The visual contact sheet shows only subtle
differences and the x8 path costs roughly eight times more inference, so TTA
is retained as an optional review setting rather than a new default. Masked
detail v2 remains higher at `28.1429 dB` in the same PSNR-only comparison.

The Stage 2 latent residual adapter v1 completed the same benchmark after
selecting step 11000 on the val100 composite metric. It is not promoted. Against
the frozen dual-context Stage 2 base, it trades `-0.0138 dB` mean Y PSNR for
`+0.00094` mean Y SSIM. Against the guarded-detail Stage 2 v2 default, it is
lower on both mean Y PSNR (`-0.0246 dB`) and mean Y SSIM (`-0.00109`). Against
masked detail v2, it trails by `0.3135 dB` Y PSNR and `0.00960` Y SSIM. This is
useful negative evidence: a small bounded latent residual can remain stable,
but it does not recover visible detail better than the existing masked/detail
branch path.

The reproducible protocol and commands are in `docs/SR_BENCHMARK.md`; full
machine-readable results are in:

```text
metrics/formal_x4_benchmark_lusir_realesr_summary.json
metrics/formal_x4_benchmark_lusir_realesr_metrics.csv
metrics/formal_x4_benchmark_div2k_swinir_summary.json
metrics/formal_x4_benchmark_div2k_swinir_metrics.csv
metrics/formal_x4_benchmark_detail_v2_masked_summary.json
metrics/formal_x4_benchmark_detail_v2_masked_metrics.csv
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_summary.json
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_metrics.csv
metrics/formal_x4_benchmark_stage2_guarded_tta_compare_summary.json
metrics/formal_x4_benchmark_stage2_guarded_tta_compare_metrics.csv
metrics/formal_x4_benchmark_stage2_latent_adapter_v1_value_compare_summary.json
metrics/formal_x4_benchmark_stage2_latent_adapter_v1_value_compare_metrics.csv
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
results. The formal DIV2K comparison is LuSIR masked detail v2 at `30.1636 dB`
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

## Stage 2 Detail-Perceptual Continuation Review

A separate Stage 2 continuation started from dual-context best98000 and trained
on `photo_detail_mix` with lower latent/decoded weight, stronger highpass
magnitude, VGG feature loss, and a detail-score selection metric:

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_detail_perceptual_v1.yaml
run:    /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_detail_perceptual_v1
W&B:    https://wandb.ai/jwheo/LuSIR/runs/hybqq4rj
```

The val100 proxy confirmed that the loss can raise high-frequency energy:
step6000 reached decoded PSNR `24.5921` with Laplacian energy ratio `0.3824`,
and latest step12000 reached decoded PSNR `24.6144` with ratio `0.3533`.
The original dual-context step98000 baseline on the same proxy was
`24.6197` and `0.3123`.

The decisive review used the same 219-image formal x4 benchmark as above, but
with the Stage 2 base path only:

| Candidate | Mean Y PSNR | Mean Y SSIM | Mean RGB PSNR | Mean RGB SSIM |
| --- | ---: | ---: | ---: | ---: |
| Bicubic | `25.7170` | `0.71773` | `24.2697` | `0.69205` |
| Dual-context step98000 | **`27.8431`** | `0.79742` | **`26.3131`** | `0.77340` |
| Detail-perceptual step6000 | `27.7737` | **`0.79914`** | `26.2482` | **`0.77506`** |
| Detail-perceptual latest12000 | `27.8356` | `0.79827` | `26.3107` | `0.77417` |

Relative to dual-context step98000, step6000 loses `0.0694 dB` Y PSNR while
gaining `0.00172` Y SSIM. Latest step12000 is almost tied: `-0.0076 dB`
Y PSNR, `+0.00085` Y SSIM, and `156/219` Y-SSIM wins. Dataset-level
latest12000 deltas are `-0.0193 dB` on DIV2K, `+0.0272 dB` on Set5,
`-0.0555 dB` on Set14, and `+0.0092 dB` on Urban100.

Visual crop review shows small improvements on some building/window grid
regions, but not a clear user-visible texture breakthrough. The conclusion is
therefore conservative: keep dual-context step98000 as the default Stage 2
checkpoint, preserve latest12000 as an optional SSIM/detail-biased research
candidate, and avoid spending the next run on a longer continuation of the same
objective.

The new machine-readable results and visual crop sheet are:

```text
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_summary.json
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_metrics.csv
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_delta_crop_selection.csv
samples/stage2_detail_perceptual_v1_benchmark_delta_crop_sheet.jpg
```

## Stage 2 Shifted-Window Attention Probe Review

The next Stage 2/base architecture probe added shifted-window self-attention
and a gated depthwise feed-forward branch to the dual-context predictor. The
goal was to test whether broader feature mixing could reduce conditional-mean
smoothing in repetitive structures such as windows, grids, leaves, and small
texture.

Three variants were evaluated on the same val100 proxy:

| Candidate | Stop step | Decoded PSNR range | Detail-ratio range | Best score |
| --- | ---: | ---: | ---: | ---: |
| v2, implementation bug | 8000 | `24.60-24.63` | `0.315-0.343` | `24.950` |
| v3 true-dual, window 8 | 6000 | `24.60-24.63` | `0.315-0.343` | `24.953` |
| v3 true-dual, window 12 | 4000 | `24.60-24.63` | `0.315-0.341` | `24.954` |

The v2 probe is retained only as a cautionary result: the
`dual_multiscale_attention` implementation initially did not enable the second
`extra_context` branch, so it compared single-context-plus-attention against
the dual-context baseline. V3 corrected this by preserving the full
dual-context path and applying attention after `context + extra_context`.
Parameter groups then trained the new attention branch at `2e-5` while keeping
the inherited trunk/context/output at `5e-6`, with a warmup-cosine scheduler.

The corrected `8x8` and `12x12` probes both stayed near the dual-context
baseline and did not produce a visible texture breakthrough. Increasing window
size raised memory and compute cost (`12x12` used about `31.5 / 46.1 GB` on a
single L40S and ran at about `1.83` micro-steps/s) without changing the
trajectory. The interpretation is that attention window size is not the
primary bottleneck. The model still learns a conservative deterministic latent
estimate under the current supervision. Future Stage 2 work should target a
residual/detail correction path on top of a preserved fidelity base, or
decoder-side detail capacity, rather than continuing window-size scaling.

## Stage 1 Decoder Capacity Audit

After the masked v5 PatchGAN detail-branch probe collapsed fidelity inside the
selected detail mask, the next diagnostic separated Stage 1 decoder capacity
from Stage 2 conditional-latent smoothing. A new tool,
`tools/analysis/audit_stage1_decoder_detail_capacity.py`, evaluates HR
autoencoding through Stage 1 and compares it with the Stage 2 decoded base on
the same `photo_detail_mix` val100 set.

| Path | Mean PSNR | SSIM | Highpass ratio | Laplacian ratio | Missing energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 VAE recon | `41.8121` | `0.99187` | `0.9965` | `0.9553` | `0.00174` |
| Stage 2 dual-context base | `26.4889` | `0.80013` | `0.7886` | `0.3191` | `0.01968` |

The result weakens the hypothesis that the current Stage 1 decoder is the main
visible-detail bottleneck. When given the HR latent, it preserves nearly all
highpass energy and most Laplacian energy. The same decoder fed by Stage 2
dual-context best98000 loses far more high-frequency structure. The active
bottleneck is therefore Stage 2 conditional-mean smoothing, not decoder
capacity.

This audit also explains a recurring metric mismatch. The legacy Stage 2
training eval reports `eval/decoded_psnr` from global MSE, whereas the audit
reports mean per-image PSNR. For the same dual-context checkpoint on the same
val100 set these are `24.6197` and `26.4889`, respectively. The Stage 2 trainer
now logs both views, plus `eval/decoded_ssim`, `eval/highpass_energy_ratio`,
`eval/missing_energy`, `eval/excess_energy`, and
`eval/mean_psnr_detail_score`.

The guarded follow-up configuration is
`configs/latent_pretrain_photo130k_lsdir_dual_detail_guarded_v2.yaml`. It starts
from dual-context best98000, keeps the architecture unchanged, avoids VGG/GAN
pressure, and slightly strengthens decoded highpass supervision. Its selection
metric, `decoded_mean_psnr + 2 * highpass_energy_ratio`, is only a shortlist
metric; visual grids, formal benchmark scores, and artifact review remain the
promotion criteria.

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

The guarded-detail Stage 2 v2 continuation from dual-context LSDIR step 98000
was stopped after 20000 micro-steps because the run had plateaued. Its selected
checkpoint is the step-10000 `best_eval_mean_psnr_detail.pt`, not the final
step. This candidate is much lighter at inference time because it only runs
the Stage 2 encoder and Stage 1 decoder: the Stage 2 model has `119.24M`
parameters, the training checkpoint is about `1.4GB`, and the model weights
inside it are about `0.44GB`. On one L40S with bf16 and 128x128 LR tiles,
measured CUDA reserved memory was `1.03GB` at tile batch 1, `3.28GB` at tile
batch 4, and `6.25GB` at tile batch 8. Practical inference can therefore run
on an 8GB GPU with tile batch 1, while 12-16GB GPUs are comfortable for local
review. Long Stage 2 training still targets a 48GB GPU class machine.

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

The current README demo alternates `Set5/img_003` between noisy LR input and
LuSIR output. A fixed Gaussian degradation (`sigma=4/255`, seed `20260622`) is
used; the noisy bicubic baseline is omitted from the animation for clarity but
is retained for metric calculation. Masked detail v2 improves over that noisy
bicubic baseline by `+5.17 dB` Y PSNR and `+0.163` Y SSIM. Because the PDF cannot
animate that comparison, the static report figure below uses
`Urban100/img_043`, the strongest measured clean-bicubic full-image example from
the fixed x4 benchmark among the reviewed candidates. It is intentionally used
as a high-impact visual explanation rather than as a claim of average SOTA
quality: masked detail v2 improves this sample over bicubic by `+10.66 dB` Y
PSNR and `+0.152` Y SSIM, mainly by restoring repeated architectural structure.

![Urban100 img_043 shows the clearest current deterministic LuSIR improvement:
LR and bicubic blur the repeating facade, while masked detail v2 recovers much
of the geometric structure without a generative sampler.](../docs/assets/lusir_current_demo_urban100_img043.jpg)

*Figure 1. High-impact full-image benchmark example. The method still remains a
conservative deterministic restoration path; this image is chosen because the
improvement is easy to see.*

The following two lime examples use the same validation crop so that the role
of the base reconstruction and the later detail branch can be inspected
separately. They are diagnostic examples, not claims of a formal benchmark win.

![The selected multiscale Stage 2 model restores color and large structure from
the degraded LR input, but remains visibly smoother than ground truth on rind
texture.](../docs/assets/stage2_multiscale_demo.jpg)

*Figure 2. Multiscale Stage 2 restores the degraded input while retaining a
visible fine-texture gap to ground truth.*

![The selected v1d high-frequency detail branch makes a controlled texture correction
over the dual-context base. It remains artifact-light, but the gap to ground
truth fine detail is still clear.](../docs/assets/detail_branch_v1d_lime_demo.jpg)

*Figure 3. Selected detail branch v1d adds a controlled, artifact-light
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

## Learned Detail-Need Mask and Masked Detail Branch v2

The next experiment separated `where detail is missing` from `what detail
should be generated`. A compact 460,545-parameter predictor observes the
frozen base reconstruction, bicubic input, Stage-2 condition latent, and four
inference-time detail proxies. On fixed `photo_detail_mix` val100, predictor
step 3250 raises pixel correlation with the GT-supervised detail-need target
from `0.5403` to `0.7456`, raises top-20% missing-detail capture from `0.3252`
to `0.3861`, and lowers excess-detail capture from `0.4838` to `0.4304`.

Masked detail branch v2 initializes from v1d step 99500 and applies the frozen
predictor as a soft spatial gate with a `0.05` floor. Two independent
continuations converged to nearly identical candidates:

| Checkpoint | Detail score | PSNR delta | Mean PSNR delta | SSIM delta |
| --- | ---: | ---: | ---: | ---: |
| step 36000 | 26.69528 | +0.18744 dB | +0.20521 dB | +0.00721 |
| selected step 38000 | **26.69601** | +0.18177 dB | +0.20432 dB | **+0.00755** |

Step 38000 is selected by the configured `eval/detail_score`. No new best
appeared through step 50000, and fixed grids at steps 34000, 38000, and 48000
were nearly indistinguishable, so the run stopped before its 20-epoch upper
bound.

The predictor establishes that missing-detail localization is learnable from
inference-time observations. However, gating the same deterministic
L1/highpass-oriented branch does not visibly synthesize the missing fine
texture. Location selection and texture generation remain separate problems.
The first top-fraction texture-gate review also showed why the mask objective
needs explicit negative examples. With a top-10% binary gate, v1 selected useful
texture-rich regions on clean inputs, but when synthetic Gaussian noise was
injected into a low-texture patch, `45.31%` of that patch fell inside the
top-10% gate and the predictor mean inside the noisy patch rose to `0.6430`.
The GT-supervised missing-detail target stayed low there, so this was not
desired texture discovery; it was the predictor opening on artifact-like
high-frequency content.

A noise-negative v2 mask probe therefore initializes from v1 step 3250 and
adds low-target patch noise augmentation plus explicit in-patch prediction and
excess penalties. Its selected step 1500 preserves clean top-10 selection
quality (`0.7219` score versus `0.7173` for v1, with excess capture improving
from `0.2496` to `0.2375`) while reducing the same injected-noise top-10
coverage from `0.4531` to `0.0000` and noisy-patch predictor mean from
`0.6430` to `0.0018`. This makes v2 the preferred gate for any subsequent
texture-generation branch.

The follow-up v5 texture probe used this v2 mask as a hard top-10% binary gate
with zero floor and retried masked VGG plus PatchGAN pressure from the selected
v2 generator. The gate itself behaved as intended: `detail_mask_mean` stayed
near `0.10` and `outside_mask_residual_l1` remained `0`. However, the
adversarial phase still collapsed fidelity inside the mask. The run moved from
`+0.0537 dB` PSNR delta and `99/100` wins at step 500 to `-0.0953 dB` and
`11/100` wins at step 3500. Visual review showed high-frequency artifact
growth rather than GT-aligned fine texture. This rejects the current PatchGAN
route; stronger adversarial pressure or longer continuation is not planned.

The implemented v6 follow-up retains the frozen fidelity base and v2
noise-negative top-10 mask, but removes the PatchGAN objective. It starts from
selected masked detail v2 step 38000 and combines masked VGG supervision,
GT-filtered RealESRGAN highpass/residual teacher signal, and a new
artifact-negative residual loss. The negative loss directly suppresses
residual highpass energy in flat target regions or regions where the base is
already sharper than the target. Its config is
`configs/detail_branch_v6_noise_gate_teacher_perceptual_no_gan_probe.yaml`,
with design and stop criteria in `docs/DETAIL_BRANCH_V6_NO_GAN_KO.md`.

A four-step smoke run confirmed that the v6 wiring preserves the intended
guardrails before long training: `eval/detail_mask_mean` is `0.1000`,
`eval/outside_mask_residual_l1` is `0.000000`, `eval/lowpass_drift_l1` is
`0.000130`, and the initialized path remains positive versus the frozen base
with `+0.0696 dB` mean PSNR and `+0.00147` SSIM. These values are not a
promotion result; the actual decision depends on 500-6000 step W&B sample
grids and whether the new supervision creates real texture without v5-style
scratch/noise artifacts.

The completed 6000-step v6 run did not collapse, but it also did not improve
over its initialization. The selected checkpoint is step 0. Final step 6000
still keeps `eval/outside_mask_residual_l1` at `0.000000` and remains positive
versus the frozen base (`+0.0534 dB` mean PSNR and `+0.00103` SSIM), but it is
below the initialized score and the fixed grids are visually almost unchanged.
This is useful negative evidence: no-GAN teacher/negative losses can keep the
branch safe, but this image-space branch still does not create the missing
texture.

The follow-up Stage 2 latent residual adapter v1 moved the residual correction
into latent space.
`configs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1.yaml`
freezes the dual-context Stage 2 best98000 predictor, feeds its latent output
and LR input into a zero-initialized 3.75M-parameter adapter, and trains only a
bounded latent residual before decoding through the same Stage 1 decoder. Its
starting output is exactly the frozen Stage 2 base, so any improvement or
regression can be attributed to the adapter rather than a full Stage 2
retraining drift.

The run completed 12000 micro-steps and selected step 11000 on the val100
composite metric. It was stable, but not useful enough to promote. On the
formal 219-image benchmark, it is slightly worse than the frozen Stage 2 base
in mean Y PSNR (`27.8294` versus `27.8431`) and lower than the guarded Stage 2
v2 default in both Y PSNR and Y SSIM. This rules out plain latent residual
adapter continuation as the next priority.

Before v5, the v3/v3b/v4 series tested the same general idea with the original
v1 mask. V3 added conservative masked VGG and PatchGAN losses and selected
step 1000, but remained visually too close to the deterministic v2 branch.
V3b increased perceptual and adversarial pressure; its final step 8000 fell to
`-0.1824 dB` PSNR delta and `11/100` wins. V4 added filtered Real-ESRGAN
teacher highpass supervision, but its best checkpoint was step 0, meaning the
teacher signal did not improve over the v3 initialization. V5 shows that
switching to the cleaner v2 noise-negative gate fixes mask leakage but does
not fix the adversarial texture objective itself.

The evaluator reports lowpass drift and residual energy inside and outside the
learned mask. These values, PSNR/SSIM, wins, and fixed visual grids remain the
guardrails for any future detail-generation probe.

During reproducibility review, the original single-image and formal-benchmark
inference paths were found to omit the learned predictor even though training
and validation applied it. The inference and benchmark runners now load the
predictor checkpoint and floor from the HF config. V1d configs without a mask
retain their previous behavior.

The masked v2 checkpoint also completed the formal 219-image clean-bicubic
benchmark. Relative to v1d, Y PSNR improves by `+0.0034`, `+0.0602`,
`+0.0135`, and `+0.0167 dB` on DIV2K, Set5, Set14, and Urban100. The overall
gain is `+0.0114 dB` Y PSNR and `+0.00118` Y SSIM. This consistent but small
gain supports the same conclusion as the fixed grids: learned gating improves
correction placement, but does not solve visible texture synthesis.

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
checkpoints/detail_mask_predictor_v1_best3250.pt
checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt
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
metrics/detail_branch_v2_masked_photo130k_lsdir_summary.json
metrics/formal_x4_benchmark_detail_v2_masked_summary.json
metrics/formal_x4_benchmark_detail_v2_masked_metrics.csv
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_summary.json
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_metrics.csv
metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_delta_crop_selection.csv
metrics/formal_x4_benchmark_stage2_latent_adapter_v1_value_compare_summary.json
metrics/formal_x4_benchmark_stage2_latent_adapter_v1_value_compare_metrics.csv
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
samples/detail_branch_v2_masked_photo130k_lsdir_best38000_grid.png
samples/stage2_detail_perceptual_v1_benchmark_delta_crop_sheet.jpg
samples/stage2_latent_adapter_v1_value_compare_selected.jpg
samples/stage2_latent_adapter_v1_value_compare_contact_sheet.jpg
docs/assets/lusir_current_demo_urban100_img043.jpg
docs/assets/lusir_current_demo_set5_butterfly.gif
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
configs/hf/detail_mask_predictor_v1.yaml
configs/hf/detail_branch_v2_masked_photo130k_lsdir.yaml
configs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1.yaml
```

## Next Work

The guarded-detail Stage 2 v2 step-10000 checkpoint is now the public Colab
default because it is the best current T4-friendly deterministic path. Residual
refiner v2 remains the conservative deterministic option, while detail branch
v1d and masked detail branch v2 are selectable single-image/tiled Colab research
options. The completed perceptual Stage 2 continuation is not promoted.
Candidate next steps are:

- do not continue masked latent residual-shift in its current form. The v1
  full-trajectory best at step 3500 increased detail ratio from 0.7965 to
  0.8272 but lost 0.7557 dB PSNR and worsened GT-aligned high-frequency error.
  Correction-strength and start-timestep sweeps showed that settings preserving
  fidelity also removed nearly all visible effect. The v2 fidelity continuation
  reduced the loss to 0.0651 dB PSNR and 0.00066 SSIM below base, but detail
  ratio increased only to 0.8003 and high-frequency error remained slightly
  worse. The branch is therefore not promoted;

- do not continue the v6/v7/v8 detail-branch objectives without a new signal.
  V6 was safe but non-improving, v7 showed that RealESRGAN teacher patches were
  not reliable under the GT highpass filter, and v8 showed that training-time
  GT detail masks can preserve a small stable correction but do not create a
  visible texture breakthrough. V8 selected step 500; step2000 was stopped after
  laplacian ratio turned slightly negative;
- do not continue the Stage 2 latent residual adapter v1 objective without a
  new supervision signal; it completed stably but trailed guarded Stage 2 v2
  on the formal benchmark;
- preserve the deterministic base and move from full latent re-prediction to a
  residual/detail correction path that predicts `target_latent - baseline_latent`
  or decoded highpass/detail residual over the existing Stage 2 output;
- use the completed Stage 1 audit result as a guardrail: the decoder preserves
  high-frequency detail when fed HR latents, so the next bottleneck to attack is
  Stage 2 conditional-mean smoothing rather than decoder capacity;
- do not continue
  `configs/latent_pretrain_photo130k_lsdir_dual_detail_gtmasked_v3_probe.yaml`
  in its current form. The train-only `prediction_missing` top20 masked
  decoded/highpass objective increased mean PSNR slightly by step1000, but
  highpass ratio fell from `0.8084` to `0.794` and missing-detail energy rose
  from `0.01897` to `0.01939`. The next Stage 2 attempt needs a changed target
  parameterization or architecture, not another mask-weighted loss continuation;
- use
  `configs/latent_pretrain_photo130k_lsdir_dual_bicubic_overfit64_probe.yaml`
  as a non-deployable upper-bound check. Its smoke baseline on train64 is
  `26.42` mean PSNR, `0.791` highpass ratio, and `0.01971` missing energy. If
  this fixed-set run cannot improve highpass ratio and missing energy, dataset
  scale or longer continuation is unlikely to solve the active smoothing
  bottleneck by itself;
- select the generative path with LPIPS, DISTS, fixed visual review,
  high-frequency metrics, lowpass drift, and seed diversity instead of PSNR
  alone;
- keep Stage 2/base architecture research separate from the generative detail
  objective, but do not continue the shifted-window attention/window-scaling
  branch without a new supervision signal;
- do not spend the next iteration on larger TTA: full x8 improves guarded
  Stage 2 by only `+0.0957 dB` mean Y PSNR while preserving the same visibly
  conservative texture character;
- because v1d remains visually conservative, noise-MSE residual diffusion
  collapses toward zero, and v5 PatchGAN collapsed inside the mask, prioritize
  patch-level perceptual and artifact-negative supervision before trying
  adversarial pressure again;
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

The learned-mask prerequisite was subsequently trained as a compact
460,545-parameter predictor using the frozen fidelity base, bicubic image,
Stage-2 condition latent, and four observable detail proxies. On the fixed
photo-detail val100 set, it improved pixel correlation with the supervised
detail-need target from `0.5403` to `0.7456`, improved top-20% missing-detail
capture from `0.3252` to `0.3861`, and reduced excess-detail capture from
`0.4838` to `0.4304`. This clears the predefined localization gate, but does
not by itself establish improved SR output. The subsequent masked v2 branch
modestly improved ordinary val100 metrics but remained visually close to v1d
and plateaued by step 38000. Therefore the same deterministic objective will
not be continued; the next experiment must change texture-generation
supervision while keeping the validated spatial gate and fidelity guardrails.
