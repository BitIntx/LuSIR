---
library_name: pytorch
license: cc-by-nc-4.0
tags:
- super-resolution
- image-restoration
- latent-diffusion
- pytorch
- research
---

# LuSIR

**LuSIR: Latent Upscaling via Self-trained Image Restoration** is a
vision-only x4 super-resolution research project trained without a pretrained
text-to-image diffusion model.

GitHub: <https://github.com/BitIntx/LuSIR>

The repository stores selected research checkpoints, configs, metrics, and
sample grids. It does not redistribute training datasets.

## Current Selected Detail Artifact

The latest public stable detail-branch checkpoint remains:

```text
checkpoints/detail_branch_v1d_deep3m_photo130k_lsdir_best99500.pt
```

It is a deterministic 3.02M-parameter image-space detail branch on top of the
frozen dual-context LSDIR Stage 2 step 98000 condition encoder and frozen Stage
1 decoder. The run completed `100086` micro-steps, exactly three epochs, and
selected step `99500` by `eval/detail_score`.

Selected ordinary `photo_detail_mix` val100 result:

```text
aggregate PSNR delta vs frozen base: +0.1646 dB
mean PSNR delta vs frozen base:      +0.1888 dB
SSIM delta vs frozen base:           +0.00647
PSNR wins:                           99/100
detail wins:                         100/100
```

Exploratory strict-bicubic DIV2K five-center-crop result:

```text
mean RGB PSNR: 31.9513 dB
vs frozen base: +0.2102 dB
vs detail v1c:  +0.1358 dB
wins:           5/5
```

The strict-bicubic result is not a formal SOTA benchmark. It uses five
512x512 center crops, PIL bicubic x4 degradation, full-image RGB PSNR, and no
border shave.

Formal full-image clean-bicubic benchmark, reported as Y PSNR / Y SSIM:

| Dataset | Dual-context base | Detail v1d |
| --- | ---: | ---: |
| DIV2K validation | 29.9575 / 0.82887 | **30.1602 / 0.83421** |
| Set5 | 31.6621 / 0.88952 | **31.8892 / 0.89440** |
| Set14 | 28.2441 / 0.77340 | **28.4123 / 0.77998** |
| Urban100 | 25.4816 / 0.76473 | **25.8755 / 0.77875** |

This uses public x4 LR pairs, MATLAB-compatible BT.601 Y, a four-pixel border
shave, and MATLAB-style SSIM. V1d improves its frozen base on all four
datasets. These clean-bicubic fidelity results are not a claim of classical-SR
SOTA or a substitute for real-degradation and perceptual evaluation.

For scale, the official SwinIR classical x4 checkpoint reaches
`31.0838 / 0.85228` on the same DIV2K evaluator, `+0.9235 dB` Y PSNR ahead of
detail v1d. The next clean-fidelity priority is therefore the Stage 2/base
reconstruction path rather than a larger detail branch.

A clean-bicubic Stage 2 continuation improved its task-specific val100 proxy
only gradually and plateaued around `25.05`. Learning-rate probes did not
change that conclusion: `20x` LR collapsed, while a `5x` from-init run matched
the original LR within evaluation noise. These val100 values are not directly
comparable with the formal full-image Y-channel benchmark above.

The signed-high-frequency residual diffusion path was evaluated and rejected:
longer noise-MSE training collapsed residual magnitude and seed diversity
toward zero. The next separate generative research path keeps the
deterministic base and validated learned mask frozen, then tests a small
bounded mask-weighted patch perceptual/adversarial head with fidelity and
artifact guardrails.

## Latest Masked Detail Research Candidate

The learned-mask-gated v2 candidate is:

```text
checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt
```

It combines the frozen 460K-parameter detail-mask predictor step 3250 with the
3.02M-parameter detail branch and a soft-mask floor of `0.05`. On ordinary
`photo_detail_mix` val100, selected step 38000 improves the frozen base by
`+0.18177 dB` aggregate PSNR, `+0.20432 dB` mean PSNR, and `+0.00755` SSIM,
with `100/100` wins.

The score plateaued after step 38000 and fixed grids were nearly
indistinguishable from nearby checkpoints. It modestly improves metrics over
v1d but does not visibly recover the missing fine texture that motivated the
experiment. It is therefore a reproducible research option, not the public
default.

On the same formal 219-image clean-bicubic benchmark, masked v2 reaches
`30.1636 / 0.83512`, `31.9495 / 0.89534`, `28.4257 / 0.78102`, and
`25.8922 / 0.78022` on DIV2K, Set5, Set14, and Urban100. It improves v1d on
all four datasets, but the overall gain is only `+0.0114 dB` Y PSNR and
`+0.00118` Y SSIM.

## Download

From a LuSIR GitHub clone:

```bash
python scripts/download_hf_checkpoints.py --preset detail_branch_v1d
```

Other useful presets include:

```text
residual_refiner_v2
stage2_photo130k_lsdir_dual
detail_branch_v1b
detail_branch_v2_masked
photo100k_xl_stage4_edge
```

The public Colab default remains the conservative deterministic residual
refiner v2 path. Detail v1d and masked detail v2 are available as research
options in the Colab WebUI with single-image and tiled inference.

## Runtime Paths

```text
public deterministic default:
  LR -> Stage 2 XL -> residual refiner v2 -> Stage 1 decoder -> SR

selected detail research path:
  LR -> dual-context LSDIR Stage 2 -> Stage 1 decoder
     -> learned detail mask -> masked detail branch v2 -> SR

generative comparison:
  LR -> Stage 2 condition encoder -> Stage 3 OR Stage 4 diffusion U-Net
     -> Stage 1 decoder -> SR
```

Stage numbers describe training order. Stage 3 and Stage 4 are alternative
diffusion checkpoints, not modules executed sequentially.

## License

- Checkpoints, generated samples, metrics, and other non-code artifacts:
  CC BY-NC 4.0.
- Source code: PolyForm Noncommercial License 1.0.0.

Commercial use is not permitted without separate written permission.
