# sr-diffusion

[![Technical Report](https://img.shields.io/badge/technical_report-PDF-blue)](paper/sr_diffusion_report.pdf)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BitIntx/sr-diffusion/blob/main/notebooks/sr_diffusion_colab_demo.ipynb)
[![Code License](https://img.shields.io/badge/code_license-PolyForm_Noncommercial_1.0.0-orange)](LICENSE)
[![Checkpoint License](https://img.shields.io/badge/checkpoints-CC_BY--NC_4.0-orange)](CHECKPOINT_LICENSE.md)

Vision-only x4 latent diffusion super-resolution experiments.
Report source: [paper/main.tex](paper/main.tex), with a plain Markdown snapshot
in [paper/TECHNICAL_REPORT.md](paper/TECHNICAL_REPORT.md).

This is a public source-available, non-commercial research project. The goal is
to train an SR model directly, without using a pretrained text-to-image
diffusion model. The intended final model handles photo and anime/illustration
domains in one codebase with domain conditioning.

This repository is not OSI-approved open source because commercial use is not
permitted.

## Goal

Target task:

```text
LR 128x128 -> HR 512x512
LR 192x192 -> HR 768x768 later
```

Planned model:

```text
HR image
  -> factor-4 VAE / autoencoder
  -> HR latent

LR image
  -> condition encoder
  -> multi-scale LR features

noisy HR latent + LR features + timestep + domain embedding
  -> conditional diffusion U-Net
  -> denoised HR latent
  -> VAE decoder
  -> x4 SR output
```

Constraints:

- PyTorch first.
- ROCm/GPU primary.
- No custom CUDA/ROCm ops.
- TPU/XLA compatibility is a later consideration, so code should stay close to
  standard PyTorch where practical.
- No pretrained T2I model dependency.

## Current Status

We finished the first **Stage 1: VAE / Autoencoder**, **Stage 2:
deterministic LR -> HR latent pretraining**, and **Stage 3: conditional latent
diffusion** passes. The photo100k scale-up has completed through Stage 4
condition-start, and the current best sampled photo100k checkpoint is the
Stage 4 condition-start checkpoint initialized from Stage 3:

```text
Stage 2 photo100k: latent_pretrain_photo100k_b64, finished step 30000
Stage 3 photo100k: diffusion_photo100k_b32, finished step 60000
Stage 4 photo100k: diffusion_photo100k_b32_stage4_condition, finished step 5000
sampled val100: Stage3 25.3745 PSNR, Stage4 25.4072 PSNR
```

For denoise/sharpening work, `photo_v2` degradation is implemented and the
Stage 2/3 photo100k v2 fine-tunes have completed:

```text
Stage 2 photo100k v2: latent_pretrain_photo100k_v2_b64, finished step 20000
Stage 3 photo100k v2: diffusion_photo100k_b32_v2, finished step 20000
Stage 3 v2 sampled val100: SR 22.6699 PSNR, bicubic 22.4103 PSNR, delta +0.2595
Stage 4 photo100k v2: diffusion_photo100k_b32_stage4_condition_v2, finished step 5000
Stage 4 v2 sampled val100: SR 22.8426 PSNR, bicubic 22.4103 PSNR, delta +0.4323
Stage 4 v2 vs Stage 3 v2: +0.1727 PSNR, wins 81 / losses 19
```

Stage 4 v2 improves the Stage 3 v2 sampled result and usually stabilizes the
denoise/sharpening output, but some color/contrast overshoot and small
cyan/green sampling artifacts remain. A stronger `photo_v3_noise_mix` Stage 2
small run was stopped at step 12700 after eval plateaued around
`eval/latent_loss` 0.282. The 500M-class Stage 2 XL condition encoder then
completed 80000 steps and beat the small v3 condition encoder:

```text
XL Stage 2 condition encoder: configs/latent_pretrain_photo100k_v3_noise_xl.yaml
XL Stage 2 best latent:       step 66000, eval/latent_loss 0.27230
XL Stage 2 best PSNR proxy:   step 72000, decoded_psnr 21.52
XL Stage 4 condition-start:   configs/diffusion_photo100k_xl_stage4_condition_v3.yaml
XL full inference params:     509.658M
```

The Stage 2 XL candidates were compared on the same 100 validation images using
condition-only decoded outputs. The step 72000 checkpoint had the best decoded
PSNR proxy and was selected for Stage 4 XL.

```text
XL Stage 2 candidate comparison:
  best_eval_latent step 66000: decoded_psnr 21.3828
  step_0072000 step 72000:     decoded_psnr 21.5241
  latest step 80000:           decoded_psnr 21.5062
```

The first Stage 4 XL condition-start run is complete. It uses the 469.6M
U-Net path, the Stage 2 XL step 72000 condition encoder, partial initialization
from Stage 4 v2, and edge/highpass decoded losses for stronger restoration.

```text
XL Stage 4 edge config: configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml
XL Stage 4 edge run:    diffusion_photo100k_xl_stage4_condition_v3_edge_b16
finished step:          5000
selected checkpoint:    step 4250, best eval/decoded_mse
training eval proxy:    decoded_psnr 21.9872
sampled val100:         SR 23.0793 PSNR, bicubic 22.3599 PSNR, delta +0.7195
W&B:                    https://wandb.ai/jwheo/sr-diffusion/runs/nog04fwr
```

This is better than the previous v3-noise XL condition-only path and beats the
bicubic baseline on the current v3 validation setup, but it is still an active
research checkpoint: noise/color cleanup improved, while fine texture recovery
remains softer than the ground truth.

Follow-up ablations showed that loss-only Stage 4 changes were not enough to
turn the diffusion U-Net into a reliable detail refiner. A role-split
lowpass-anchor probe preserved structure better but still damaged the Stage 2
condition output at larger start timesteps. A newer `gated_residual_x0`
parameterization constrains the U-Net to predict a bounded residual on top of
the Stage 2 condition latent instead of freely predicting full x0. This greatly
reduced condition damage but still did not beat the Stage 2 condition-only
baseline on average:

```text
Stage 2 XL condition-only mild val100: 25.0449 PSNR
Stage 4 role-split mild t25:          24.5747 PSNR, condition wins 3/100
Stage 4 gated residual step2000 t25:  25.0445 PSNR, condition wins 34/100
W&B gated residual probe:             https://wandb.ai/jwheo/sr-diffusion/runs/edfko8e8
```

The current research conclusion is that Stage 4 must receive a more direct
signal for where residual detail is needed. Simply continuing the same
diffusion loss is less promising than supervising the residual/gate path or
adding a condition uncertainty/detail-need signal.

A direct Stage 2 residual diagnostic confirmed that the missing signal is
mostly high-frequency detail rather than lowpass structure. On mild val100,
injecting only the GT highpass residual into the Stage 2 condition latent gives
a large oracle gain, while lowpass residuals barely move PSNR:

```text
Stage2 residual diagnostic mild val100:
  condition-only decoded PSNR: 25.0543
  bicubic PSNR:                24.4778
  oracle full residual PSNR:   41.8207, +16.7664 vs condition
  oracle highpass PSNR:        35.0872, +10.0329 vs condition
  oracle lowpass PSNR:         25.0814, +0.0270 vs condition
  residual highpass energy:    0.8988
  residual lowpass energy:     0.0758
```

The first deterministic bounded residual refiner probe produced a small but
real gain over Stage 2 condition-only without the destructive edits seen in
diffusion Stage 4 probes. A sparse gate was better than simply opening the gate:

```text
Sparse-gate residual refiner step 500:
  condition mean PSNR: 25.0449
  refined mean PSNR:   25.1178
  delta:               +0.0729
  wins vs condition:   86/100
  gate mean:           0.2147

Open-gate residual refiner step 500:
  refined mean PSNR:   25.0972
  delta:               +0.0523
  wins vs condition:   73/100
  gate mean:           0.8680
```

This does not solve final SR quality yet, but it changes the next step: use the
deterministic residual path as a supervised detail teacher or warm start before
asking the diffusion U-Net to hallucinate residual detail.

The residual refiner is now wired into standalone eval/inference tooling and
was checked across the three active degradation presets using the same frozen
step-500 sparse-gate checkpoint:

```text
Residual refiner val100 cross-degradation eval:
  mild:
    bicubic 24.4778, condition 25.0449, refined 25.1178
    refined vs condition +0.0729, wins 86/100
  photo_v2:
    bicubic 22.4103, condition 22.9271, refined 22.9767
    refined vs condition +0.0496, wins 77/100
  photo_v3_noise_mix:
    bicubic 22.3599, condition 22.9014, refined 22.9600
    refined vs condition +0.0586, wins 86/100
```

Qualitatively, the refiner stays very close to the Stage 2 condition output. It
does not introduce the destructive edits seen in earlier diffusion probes, but
the visible change is small. The best next use is therefore as a safe residual
teacher/warm start, not as the final detail generator by itself.

Residual refiner v2 increased model capacity and added decoded-image and
decoded-highpass supervision while training on `photo_detail_mix`. The
12000-step run selected step 11000:

```text
Residual refiner v2, selected step 11000:
  photo_detail_mix: condition 25.3103, refined 25.4420, +0.1318, wins 90/100
  mild:             condition 25.0449, refined 25.1627, +0.1178, wins 87/100
  photo_v2:         condition 22.9271, refined 23.0257, +0.0986, wins 84/100
  photo_v3_noise_mix: condition 22.9014, refined 23.0174, +0.1160, wins 88/100
```

Compared with the original sparse-gate refiner, the cross-preset PSNR gain is
roughly doubled without introducing visible white artifacts, fake texture, or
destructive edits. The result is still conservative: it improves local
contrast and boundaries but does not reconstruct the full GT fine texture.
Download the selected artifacts with:

```bash
python scripts/download_hf_checkpoints.py --preset residual_refiner_v2
```

That teacher-supervision path was then tested in the gated-residual Stage 4
U-Net on `photo_v3_noise_mix`. It produced a small, stable PSNR cleanup gain,
but did not solve visible detail recovery:

```text
Teacher-supervised Stage4, sampled photo_v3_noise_mix val100:
  step 2000, t25: 22.9640 PSNR, +0.0626 vs condition, wins 68/100
  step 2000, t50: 22.9639 PSNR, +0.0625 vs condition
  step 4000, t25: 22.9571 PSNR
  step 8000, t25: 22.9490 PSNR
```

The best sampled checkpoint was step 2000; continuing to step 8000 reduced the
result. Visual inspection still shows strong smoothing of fur, leaves,
branches, and buildings. A simple absolute-Laplacian diagnostic measured only
`21.8%` of GT detail energy for the teacher output, versus `32.7%` for the
existing edge-loss Stage 4 t25 output. The next priority is therefore the
degradation curriculum: `photo_v3_noise_mix` currently contains enough severe
noise cases to bias the model toward denoise/cleanup instead of user-facing
detail restoration.

That curriculum was redesigned as `photo_detail_mix`, with `35%` clean,
`48%` detail-preserving photo degradation, `15%` mild degradation, and only
`2%` strong `photo_v2` cases. On the fixed val100 set, its bicubic baseline is
`24.7357` PSNR versus `22.3599` for `photo_v3_noise_mix`, and chroma corruption
is about 75% lower. The existing Stage 2 XL condition encoder already handles
this distribution well, so Stage 2 was kept frozen and the teacher-supervised
gated-residual Stage 4 was adapted for 12000 micro-steps:

```text
Photo-detail Stage4 long run:
  config: configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml
  W&B: https://wandb.ai/jwheo/sr-diffusion/runs/so0lbyte
  condition-only:       25.3103 PSNR
  teacher init:         25.3187 PSNR, +0.0084 vs condition, wins 46/100
  selected step 8000:   25.3406 PSNR, +0.0303 vs condition, wins 71/100
  latest step 12000:    25.3337 PSNR, +0.0235 vs condition, wins 67/100
  previous edge Stage4: 25.1176 PSNR, -0.1927 vs condition, wins 13/100
```

This is the first sampled Stage 4 gated-residual result that beats the Stage 2
condition-only baseline on both mean PSNR and a clear majority of samples.
Visually it preserves the condition output and avoids the broad destructive
edits of the edge model. It is still a conservative restoration model rather
than a strong missing-detail generator: mean absolute-Laplacian energy is about
`29.7%` of GT, and rare strong-degradation cases can still produce bright
artifacts.

For VM migration and continuation context, read:

- [docs/HANDOFF_KO.md](docs/HANDOFF_KO.md)
- [docs/VM_RECOVERY_KO.md](docs/VM_RECOVERY_KO.md)

Implemented:

- Project scaffold and config loading.
- Manifest-based dataset loader with `photo` / `anime` domain IDs.
- On-the-fly x4 degradation pipeline.
- Factor-4 `AutoencoderKL`.
- Autoencoder training loop with bf16 autocast.
- W&B online/offline logging.
- Fixed validation sample logging for Stage 1:
  - `samples/LR`
  - `samples/GT`
  - `samples/HR`
- Validation eval during training:
  - `eval/loss`
  - `eval/recon`
  - `eval/kl`
  - `eval/mse`
  - `eval/psnr`
  - `eval/num_images`
- Standalone checkpoint eval script.
- Scratch recovery scripts for ephemeral VM storage.
- Stage 2 LR-to-latent predictor and training loop.
- Stage 3 conditional U-Net, noise scheduler, and diffusion training loop.
- Stage 3 DDIM/img2img inference and sampled validation eval.
- Stage 4-lite low-timestep and condition-start fine-tuning.
- `photo_v2` degradation for stronger blur/noise/compression/ringing/color
  shift/banding experiments.
- `photo_v3_noise_mix` degradation and XL configs for 500M-class
  denoise/sharpening experiments.
- condition-only validation tooling for isolating Stage 2 from diffusion.
- gated residual x0 diffusion parameterization for bounded Stage 4 refinement.
- Stage 2 residual oracle diagnostics for lowpass/highpass error isolation.
- Deterministic bounded residual refiner probe with sparse/open gate ablations.
- Standalone residual refiner eval/inference tooling, including tiled LR
  inference for arbitrary-size inputs.
- Partial checkpoint initialization for widened/deepened Stage 2 and diffusion
  models via `--partial-init`.
- Multi-GPU diffusion training through PyTorch DDP, with single-GPU fallback
  when not launched through `torchrun`.

Stage 1 training config:

```text
configs/autoencoder_photo10k.yaml
```

Stage 1 run name:

```text
autoencoder_photo10k_b16_eval_online
```

Selected Stage 1 VAE checkpoint:

```text
/home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt
```

Stage 1 VAE shape:

```text
HR 512x512 -> latent 128x128
latent channels: 16
batch size: 16
max steps: 100000
train set: 10000 photo images
val set: 100 photo images
eval: every 1000 steps
fixed sample logging: every 500 steps
```

The first Stage 1 pass was stopped at step `50000`. The selected checkpoint is
`best_eval_recon.pt`, which matched the 50k checkpoint in the current run:

```text
eval/recon: 0.01198
eval/kl:    9.38684
eval/psnr:  40.19
```

Current Stage 2 config:

```text
configs/latent_pretrain_photo10k.yaml
```

Stage 2 photo100k scale-up config:

```text
configs/latent_pretrain_photo100k.yaml
```

This run uses batch size `64` on MI300X and `max_steps: 30000`, which is about
18.6 passes over the 103,450-image training split.

Current Stage 2 run name:

```text
latent_pretrain_photo10k_b16
```

Selected Stage 2 checkpoint:

```text
/home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
```

Stage 2 final result:

```text
finished step: 50000
best eval latent loss: step 48000, eval/latent_loss 0.21775
best decoded PSNR proxy: step 47000, eval/decoded_psnr 23.89
```

Stage 2 photo100k scale-up result:

```text
run name: latent_pretrain_photo100k_b64
finished step: 30000
selected checkpoint: /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
best eval latent loss: step 28000, eval/latent_loss 0.21230
best decoded PSNR proxy: step 22000, eval/decoded_psnr 23.93
final eval: step 30000, eval/latent_loss 0.21267, eval/decoded_psnr 23.88
```

Current Stage 3 config:

```text
configs/diffusion_photo10k_b32.yaml
```

Next scale-up Stage 3 config:

```text
configs/diffusion_photo100k_b32.yaml
```

Current Stage 3 model:

```text
conditional U-Net params: 76.6M
frozen Stage 2 condition encoder params: 2.4M
latent shape: 16 x 128 x 128
batch size: 32
max steps: 25000
```

Selected Stage 3 checkpoint:

```text
/home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
```

Stage 3 training result:

```text
finished step: 25000
best eval noise/x0: step 24000, eval/noise_mse 0.00766, eval/x0_mse 0.09063
best decoded PSNR diagnostic: step 25000, eval/decoded_psnr 24.10
```

Sampled Stage 3 eval, using `--init condition`, `--start-timestep 50`,
and 32 DDIM steps on 32 fixed validation images:

```text
mean bicubic PSNR: 24.66
mean SR PSNR:      25.55
mean delta:        +0.89 dB
```

Sampled Stage 3 eval on all 100 validation images:

```text
mean bicubic PSNR: 24.478
mean SR PSNR:      25.222
mean delta:        +0.744 dB
```

Stage 4-lite low-timestep fine-tune result:

```text
config: configs/diffusion_photo10k_b32_stage4_lowt.yaml
initialized from: Stage 3 best checkpoint
train timesteps: 0..100
finished step: 5000
best eval/x0_mse: step 5000, eval/x0_mse 0.01186
best decoded PSNR diagnostic: step 4500, eval/decoded_psnr 32.74
sampled val32 SR PSNR: 25.5493
sampled val32 delta vs Stage 3: -0.0037 dB
decision: do not promote; keep Stage 3 as current best sampled checkpoint
```

Stage 4 condition-start fine-tune result:

```text
config: configs/diffusion_photo10k_b32_stage4_condition.yaml
selected checkpoint: /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt
initialized from: Stage 3 best checkpoint
train timesteps: 25..100
stopped early: step 2500, best checkpoint at step 1000
best one-step condition diagnostic: step 1000, eval/decoded_psnr 23.78
best sampled setting: --init condition --start-timestep 25 --steps 32
sampled val32 SR PSNR: 25.660
sampled val100 SR PSNR: 25.293
sampled val100 delta vs Stage 3: +0.071 dB
decision: promote as current best sampled checkpoint
```

This trains the low-timestep path from the Stage 2 condition latent instead of
from a noised ground-truth latent, matching the current inference initialization
more closely.

At `batch_size=32`, one epoch is:

```text
10000 images / 32 = 312.5 steps
```

So the Stage 3 `25000` step config is about `80` epochs.

## Data

The current photo training manifest is:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_photo10k.csv
```

It contains:

```text
photo/train: 10000
photo/val:   100
```

The 10k photo set is built from:

- DIV2K HR.
- Flickr2K HR.
- A deterministic subset of COCO train2017.

The active scale-up target is:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_photo100k.csv
```

It is built from DF2K plus 100,000 deterministic COCO train2017 images selected
with short side `>=320`, for about 103,550 training images and 100 validation
images. COCO only has 45,897 train2017 images with short side `>=480`, so the
stricter high-resolution-only variant is closer to photo50k.

LR images are not stored. They are generated on the fly from HR crops by the
degradation pipeline. The current `mild` degradation already includes light LR
noise:

- Gaussian noise with probability `0.25`, sigma `[0.0, 4.0]`.
- Poisson noise with probability `0.05`.
- JPEG/WebP compression.
- blur, color jitter, sharpening, and mild banding.

The `photo_v2` degradation preset is available for denoise/sharpening work. It
adds stronger blur, LR blur, signal-dependent sensor noise, heavier
Gaussian/Poisson noise, stronger JPEG/WebP artifacts, edge ringing,
oversharpen halos, color shift, and stronger banding. Because it changes the LR
distribution seen by the condition encoder, the recommended path is to
fine-tune Stage 2 on `photo_v2` before running Stage 3/4 experiments that use
the same preset.

The `photo_v3_noise_mix` preset is a stronger denoise-focused curriculum. It
mixes `photo_v2`, `photo_v3_noise`, and `mild` degradations so the condition
encoder sees heavy Gaussian/sensor/chroma noise without losing cleaner inputs
entirely. `photo_v3_noise` adds explicit chroma/color noise and stronger
compression/noise ranges while keeping oversharpen/ringing probabilities
moderate to avoid reinforcing the cyan/green dot artifacts observed in v2.

For VAE training, LR is only used for visual logging. The VAE loss is:

```text
HR -> encode -> latent -> decode -> reconstructed HR
```

LR degradation becomes a core training signal in Stage 2/3.

See [docs/DATASETS.md](docs/DATASETS.md) for dataset notes and licensing caveats.

## Scratch Disk

This VM exposes an ext4 scratch partition labeled `DOSCRATCH`. Mount it before
large datasets or long training runs:

```bash
bash scripts/mount_doscratch.sh
```

The default mount point is:

```text
/home/jwheojjang/scratch
```

The scratch volume is treated as ephemeral. After a VM restart, recover the
scratch layout and development datasets with:

```bash
bash scripts/recover_scratch.sh
```

That recreates:

- scratch directories
- toy dataset
- DIV2K
- Flickr2K
- COCO train2017 subset
- `manifest_photo10k.csv`

To recover the larger photo100k setup after scratch loss:

```bash
bash scripts/recover_scratch.sh --coco-count 100000
```

To recover only the smaller DIV2K seed dataset:

```bash
bash scripts/recover_scratch.sh --skip-flickr2k --skip-coco
```

## Hugging Face

Hugging Face is used as persistent checkpoint storage because scratch can be
lost after VM restarts. The current target is a public model repository:

```text
jwheo/sr-diffusion
```

Upload only selected checkpoints/configs/metrics, not raw datasets. See
[docs/HUGGINGFACE.md](docs/HUGGINGFACE.md) for the exact upload commands.

## Quick Prototype Inference

The public Hugging Face prototype can be downloaded and run from a fresh clone.
The default inference config still points at the smaller 10k Stage 4
condition-start checkpoint for faster setup. The Colab notebook now also lets
you select the larger photo100k Stage 4 checkpoints, including the public XL
edge-loss Stage 4 checkpoint for denoise/sharpening review. The newer gated
residual diffusion checkpoint is documented as an experiment but not promoted.
The deterministic residual refiner probe is preserved on Hugging Face as a
research artifact because it beats Stage 2 condition-only on mild val100, but it
is not wired into the default public inference path yet.

For a click-to-run demo, open the Colab notebook:

```text
notebooks/sr_diffusion_colab_demo.ipynb
```

Install PyTorch for the target machine first. For this ROCm VM:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install -e .
```

Download the selected public checkpoints from Hugging Face:

```bash
python scripts/download_hf_checkpoints.py
```

Download the larger photo100k/v2 artifact set:

```bash
python scripts/download_hf_checkpoints.py --preset photo100k
```

Download the latest XL Stage 4 edge-loss artifact set:

```bash
python scripts/download_hf_checkpoints.py --preset photo100k_xl_stage4_edge
```

Download the latest residual diagnostic/refiner artifact set:

```bash
python scripts/download_hf_checkpoints.py --preset residual_refiner_stage2_xl_mild
```

Download the selected teacher-supervised Stage 4 probe checkpoint and eval
artifacts:

```bash
python scripts/download_hf_checkpoints.py --preset stage4_teacher_residual_probe
```

Evaluate the residual refiner on a fixed validation preset:

```bash
python eval_residual_refiner.py \
  --degradation-preset photo_v3_noise_mix \
  --output-dir outputs/eval_residual_refiner_photo_v3
```

Run deterministic residual-refiner inference from an LR image:

```bash
python infer_residual_refiner.py \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/residual_refiner_demo
```

Run the same path on a larger LR image with tiled blending:

```bash
python infer_residual_refiner.py \
  --input-lr /path/to/larger_lr.png \
  --output-dir outputs/residual_refiner_tiled \
  --tile \
  --tile-overlap 32
```

Run x4 SR from an LR image. The default HF config expects a 128x128 LR crop and
writes a 512x512 output:

```bash
python infer_diffusion.py \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/demo
```

Run tiled x4 SR from a larger LR image:

```bash
python infer_diffusion.py \
  --input-lr /path/to/larger_lr.png \
  --output-dir outputs/tiled_demo \
  --tile \
  --tile-overlap 32
```

Tiled inference splits the LR image into overlapping 128x128 tiles, samples each
tile, and feather-blends the 512x512 tile outputs back into one x4 image. It is
slower than single-tile inference because diffusion sampling runs per tile.
Start with small LR images, for example 256x256 or 384x384, when using Colab.

For a controlled smoke test from an HR image, let the script center-crop HR and
create the degraded LR input first:

```bash
python infer_diffusion.py \
  --input-hr /path/to/hr_image.png \
  --output-dir outputs/demo_from_hr \
  --seed 123
```

The output is `sr_00.png`. The default config is
`configs/hf/diffusion_stage4_condition.yaml`, which points at:

```text
checkpoints/stage1_autoencoder_best_eval_recon.pt
checkpoints/stage2_latent_pretrain_best_eval_latent.pt
checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt
```

For the current photo100k Stage 4 checkpoint:

```bash
python infer_diffusion.py \
  --config configs/hf/diffusion_photo100k_stage4_condition.yaml \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/photo100k_stage4
```

For the current experimental photo100k `photo_v2` Stage 4 checkpoint:

```bash
python infer_diffusion.py \
  --config configs/hf/diffusion_photo100k_stage4_condition_v2.yaml \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/photo100k_v2_stage4
```

For the latest experimental XL `photo_v3_noise_mix` Stage 4 edge-loss
checkpoint:

```bash
python infer_diffusion.py \
  --config configs/hf/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/photo100k_xl_edge_b16
```

For the earlier photo100k `photo_v2` Stage 3 checkpoint:

```bash
python infer_diffusion.py \
  --config configs/hf/diffusion_photo100k_v2.yaml \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/photo100k_v2_stage3
```

To compare the earlier Stage 3 baseline instead:

```bash
python infer_diffusion.py \
  --config configs/hf/diffusion_stage3_baseline.yaml \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/stage3_demo
```

These are research checkpoints under a non-commercial license. They are useful
for inspecting the prototype behavior, not yet a polished production SR model.

## License

Code is released under the [PolyForm Noncommercial License 1.0.0](LICENSE).
Model checkpoints and generated artifacts are released under
[CC BY-NC 4.0](CHECKPOINT_LICENSE.md).

Commercial use is not permitted without separate written permission. This
includes paid hosted inference, resale, or integration into commercial
products.

## Training

Run the current Stage 1 VAE training config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_autoencoder.py \
  --config configs/autoencoder_photo10k.yaml
```

Recommended long-running launch through tmux:

```bash
tmux new-session -d -s sr_ae10k \
  'cd /home/jwheojjang/sr-diffusion && env PYTHONUNBUFFERED=1 /home/jwheojjang/venvs/rocm/bin/python train_autoencoder.py --config configs/autoencoder_photo10k.yaml > /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/train_tmux.log 2>&1'
```

Watch the training log:

```bash
tail -f /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/train_tmux.log
```

Run the current Stage 2 deterministic latent pretraining config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo10k.yaml
```

Run the photo100k Stage 2 scale-up from the selected 10k checkpoint:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo100k.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
```

Run the XL photo100k Stage 2 condition encoder for the 500M-class path:

```bash
/home/jwheojjang/venvs/cuda/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo100k_v3_noise_xl.yaml
```

This run has completed. Important checkpoints are available on Hugging Face:

```text
checkpoints/stage2_photo100k_v3_noise_xl_b64_best_eval_latent.pt
checkpoints/stage2_photo100k_v3_noise_xl_b64_step_0072000.pt
checkpoints/stage2_photo100k_v3_noise_xl_b64_latest.pt
metrics/stage2_photo100k_v3_noise_xl_b64_summary.json
```

Recommended Stage 2 tmux launch:

```bash
tmux new-session -d -s sr_stage2 \
  'cd /home/jwheojjang/sr-diffusion && env PYTHONUNBUFFERED=1 /home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py --config configs/latent_pretrain_photo10k.yaml > /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/train_tmux.log 2>&1'
```

Watch the Stage 2 log:

```bash
tail -f /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/train_tmux.log
```

Run the current Stage 3 conditional diffusion config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo10k_b32.yaml
```

After Stage 2 photo100k finishes, run the photo100k Stage 3 config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo100k_b32.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
```

The completed 500M-class Stage 4 XL edge-loss run used the command below. It
reuses shape-compatible tensors from the smaller Stage 4 v2 checkpoint through
partial initialization. Launch with `torchrun` for two GPUs, or run the same
script directly for single-GPU fallback:

```bash
torchrun --standalone --nproc_per_node=2 train_diffusion.py \
  --config configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints/best_eval_condition_decoded.pt \
  --partial-init
```

The gated residual x0 probe constrains the diffusion U-Net to predict a bounded
residual and gate on top of the Stage 2 condition latent. It was initialized
from the role-split mild probe, then stopped at step 2000 after sampled
validation reached condition-only parity:

```bash
python train_diffusion.py \
  --config configs/diffusion_photo100k_xl_stage4_condition_v3_gated_residual_mild_b8_probe.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_xl_stage4_condition_v3_rolesplit_mild_b8_probe/checkpoints/best_eval_condition_decoded.pt \
  --partial-init
```

Recommended Stage 3 tmux launch:

```bash
tmux new-session -d -s sr_stage3 \
  'cd /home/jwheojjang/sr-diffusion && env PYTHONUNBUFFERED=1 /home/jwheojjang/venvs/rocm/bin/python train_diffusion.py --config configs/diffusion_photo10k_b32.yaml > /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/train_tmux.log 2>&1'
```

Watch the Stage 3 log:

```bash
tail -f /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/train_tmux.log
```

Watch GPU usage:

```bash
watch -n 1 rocm-smi --showuse --showmemuse --showtemp --showpower
```

On CUDA systems:

```bash
watch -n 1 nvidia-smi
```

Attach to the tmux session:

```bash
tmux attach -t sr_ae10k
```

Detach without stopping training:

```text
Ctrl-b d
```

## Eval

Training eval is enabled in `configs/autoencoder_photo10k.yaml`:

```yaml
eval:
  enabled: true
  split: val
  limit: 100
  batch_size: 16
  every: 1000
  run_at_start: true
```

This means:

- eval at step `1`
- eval at step `1000`
- eval at step `2000`
- and so on

The best checkpoint by `eval/recon` is written to:

```text
checkpoints/best_eval_recon.pt
```

Manual checkpoint eval:

```bash
/home/jwheojjang/venvs/rocm/bin/python eval_autoencoder.py \
  --config configs/autoencoder_photo10k.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/latest.pt \
  --split val \
  --limit 100
```

## W&B

The current config logs to W&B online:

```yaml
logging:
  wandb:
    project: sr-diffusion
    name: autoencoder_photo10k_b16_eval_online
    mode: online
```

Image logging uses fixed validation images so improvements are comparable over
time:

```yaml
logging:
  samples:
    split: val
    count: 4
    indices: [0, 1, 2, 3]
```

Logged image keys:

- `samples/LR`: degraded LR, upsampled for viewing.
- `samples/GT`: original HR target.
- `samples/HR`: VAE reconstruction.

The name `samples/HR` currently means reconstructed HR output. If this becomes
confusing, rename it to `samples/Recon` before the next large run.

## Project Roadmap

Stage 0: scaffold and data pipeline

- Done.
- Repo scaffold, configs, manifests, degradation pipeline, smoke tests.

Stage 1: VAE / Autoencoder

- Done for the first pass.
- Train factor-4 VAE on 512 HR crops.
- Select checkpoint using fixed visual samples plus `eval/recon`, `eval/psnr`,
  and residual qualitative checks.
- Possible improvements before moving on:
  - LPIPS/perceptual eval.
  - perceptual training loss.
  - KL weight sweep.
  - larger or domain-balanced data.
  - rename `samples/HR` to `samples/Recon` for clarity.

Stage 2: deterministic LR -> HR latent pretrain

- Done for the first 10k pass; photo100k scale-up is the next active pass.
- Freeze the selected Stage 1 VAE.
- Train an LR-to-latent predictor that maps degraded LR inputs to HR VAE
  encoder means.
- This is where LR degradation quality starts to matter directly.
- Log fixed validation `samples/LR`, `samples/GT`, and `samples/Pred` to W&B.

Run the current Stage 2 pretraining config:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo10k.yaml
```

Run the photo100k Stage 2 scale-up:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_latent_pretrain.py \
  --config configs/latent_pretrain_photo100k.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
```

Stage 3: conditional latent diffusion

- First pass complete. It is the baseline for the current Stage 4 condition
  checkpoint.
- Train diffusion U-Net over HR latents.
- Conditioning:
  - frozen Stage 2 LR-to-latent condition encoder
  - timestep embedding
  - photo/anime domain embedding
- Initial model size is 76.6M trainable U-Net parameters.
- Target model size is roughly 250M-500M parameters after the pipeline is stable.

Stage 4: perceptual / GAN fine-tune

- Current stage.
- First conservative Stage 4-lite low-timestep fine-tune is complete. It
  improved one-step diagnostics but not the fixed 32-step sampled eval, so it
  is not promoted over Stage 3.
- Condition-start fine-tuning, initialized from the Stage 3 best checkpoint,
  is the current best sampled SR checkpoint. It trains low timesteps `25..100`,
  but starts the training noisy latent from the Stage 2 condition latent so the
  train path better matches `infer_diffusion.py --init condition`.
- It uses a small effective-noise loss plus a stronger x0 latent reconstruction
  loss to preserve fidelity. The best sampled setting so far is
  `--start-timestep 25`.
- Use carefully, because later perceptual/GAN tuning can improve apparent
  sharpness while hurting fidelity.
- The first XL Stage 4 edge-loss run is complete. It improves the current v3
  sampled validation result over bicubic by +0.7195 dB, but outputs are still
  softer than GT on fine textures.
- Later role-split and gated residual probes show that Stage 4 must be
  structurally constrained. Gated residual x0 prediction nearly matches the
  Stage 2 condition-only mild baseline at t25 (`25.0445` vs `25.0449` PSNR)
  and improves condition win count to `34/100`, but it still does not beat the
  deterministic condition path on average.
- The next high-signal direction is not simply longer Stage 4 training. Add a
  more direct residual/gate target or condition uncertainty/detail-need signal.
- The residual diagnostic shows the recoverable gap is mostly highpass. A
  deterministic sparse-gate residual refiner reaches `25.1178` mean PSNR on
  mild val100 versus `25.0449` for condition-only, so the next Stage 4 path
  should use that residual signal as supervision or initialization rather than
  only changing diffusion loss weights.
- Direct teacher supervision was tested through 8000 micro steps. It improved
  `photo_v3_noise_mix` condition-only by `+0.0626 dB` at step 2000, but outputs
  remained strongly smoothed and later checkpoints regressed. The next
  high-signal work is a less aggressive, detail-preserving degradation
  curriculum rather than another long continuation of this Stage 4 objective.

Run the Stage 4-lite low-timestep fine-tune:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo10k_b32_stage4_lowt.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
```

Run the Stage 4 condition-start fine-tune:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_photo10k_b32_stage4_condition.yaml \
  --init-checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
```

Stage 5: few-step distillation

- Distill the diffusion model for faster inference.

Stage 6: preference eval

- Fixed private eval set.
- Generate outputs from multiple checkpoints/settings.
- A/B comparisons.
- Accumulate Elo separately for photo and anime.

## Repo Layout

```text
configs/                  experiment configs
docs/                     dataset and project notes
scripts/                  dataset, scratch, and utility scripts
src/sr_diffusion/         package code
  datasets/               manifest dataset
  degradations/           x4 LR degradation pipeline
  eval/                   eval helpers
  losses/                 reconstruction/KL losses
  models/                 AutoencoderKL, LR predictor, diffusion U-Net
train_autoencoder.py      Stage 1 training entrypoint
train_latent_pretrain.py  Stage 2 deterministic latent pretraining entrypoint
train_diffusion.py        Stage 3 conditional diffusion training entrypoint
infer_diffusion.py        Stage 3 DDIM/img2img SR sampling entrypoint
eval_diffusion_samples.py sampled diffusion validation eval
compare_eval_samples.py   sampled eval comparison contact sheets
eval_autoencoder.py       standalone VAE eval entrypoint
infer_reconstruct.py      reconstruction smoke/inference
tests/                    unit tests
```

## Smoke Tests

Create a toy dataset:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/make_toy_dataset.py \
  --output runs/toy_data \
  --count 16
```

Train a tiny autoencoder for a few steps:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_autoencoder.py \
  --config configs/autoencoder_tiny.yaml \
  --limit-steps 10
```

Run unit tests:

```bash
/home/jwheojjang/venvs/rocm/bin/python -m pytest
```

Run a tiny Stage 3 smoke test:

```bash
/home/jwheojjang/venvs/rocm/bin/python train_diffusion.py \
  --config configs/diffusion_scratch_tiny.yaml \
  --limit-steps 1
```

Run current best Stage 4 condition-start sampling from an HR image by creating
a controlled LR input first:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_diffusion.py \
  --config configs/diffusion_photo10k_b32_stage4_condition.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt \
  --input-hr /path/to/hr_image.png \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/infer_diffusion_stage4_condition \
  --steps 32 \
  --seed 123
```

The Stage 4 condition config sets `sampling.start_timestep: 25`, so the command
above uses the best sampled setting found so far unless `--start-timestep` is
passed explicitly.

Run Stage 3 baseline sampling from an HR image by creating a controlled LR input first:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_diffusion.py \
  --config configs/diffusion_photo10k_b32.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt \
  --input-hr /path/to/hr_image.png \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/infer_diffusion_sample \
  --steps 32 \
  --seed 123
```

Run Stage 3 baseline sampling from an existing LR image:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_diffusion.py \
  --config configs/diffusion_photo10k_b32.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt \
  --input-lr /path/to/lr_128.png \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/infer_diffusion_sample \
  --steps 32 \
  --seed 123
```

The default sampler starts from the Stage 2 condition latent with light noise
added (`--init condition`). If a config has `sampling.start_timestep`, that
value is used when `--start-timestep` is omitted. Otherwise condition sampling
falls back to `50`. Pure noise sampling is available with `--init noise`, but
the current checkpoints are more stable in condition-initialized mode.

Run a small sampled validation sweep and compare against bicubic:

```bash
/home/jwheojjang/venvs/rocm/bin/python eval_diffusion_samples.py \
  --config configs/diffusion_photo10k_b32_stage4_condition.yaml \
  --checkpoint /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage4_condition_val8_32step \
  --split val \
  --limit 8 \
  --steps 32 \
  --seed 1337
```

The sampled eval grid is written as `grid_lr_bicubic_sr_gt.png`, with columns in
this order: LR nearest, bicubic, SR, GT.

Compare two sampled eval directories and create top win/loss contact sheets:

```bash
/home/jwheojjang/venvs/rocm/bin/python compare_eval_samples.py \
  --baseline-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage3_val100_t50_32step \
  --candidate-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage4_condition_val100_t25_32step \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/compare_stage3_vs_stage4_condition_val100 \
  --baseline-label stage3 \
  --candidate-label stage4cond
```

Reconstruct one image:

```bash
/home/jwheojjang/venvs/rocm/bin/python infer_reconstruct.py \
  --config configs/autoencoder_tiny.yaml \
  --checkpoint runs/autoencoder_tiny/checkpoints/latest.pt \
  --input runs/toy_data/images/0000.png \
  --output-dir runs/reconstruct_smoke
```
