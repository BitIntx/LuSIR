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

The latest public detail-branch research checkpoint is:

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
photo100k_xl_stage4_edge
```

The public Colab default remains the conservative deterministic residual
refiner v2 path. Detail v1d is preserved as the latest research detail
candidate and is available in the Colab WebUI with single-image and tiled
inference.

## Runtime Paths

```text
public deterministic default:
  LR -> Stage 2 XL -> residual refiner v2 -> Stage 1 decoder -> SR

selected detail research path:
  LR -> dual-context LSDIR Stage 2 -> Stage 1 decoder
     -> detail branch v1d -> SR

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
