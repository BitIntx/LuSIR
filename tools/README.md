# Command-line tools

- `train/`: model training entry points.
- `eval/`: dataset-level evaluation entry points.
- `infer/`: single-image and tiled inference entry points.
- `analysis/`: experiment comparison, diagnostics, and report generation.

Run tools from the repository root, for example:

```bash
python tools/train/train_latent_pretrain.py --config configs/latent_pretrain_photo100k_multiscale_hqmix_long.yaml
python tools/infer/infer_residual_refiner.py --help
```

Fixed visual review workflow:

```bash
python tools/eval/build_fixed_review_set.py \
  --manifest /home/ubuntu/scratch/sr-diffusion/data/manifest_photo130k_lsdir.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/review_sets/detail_v1 \
  --split val \
  --count 12 \
  --presets photo_detail_mix mild photo_v2 photo_v3_noise_mix

python tools/eval/run_fixed_review_residual_refiner.py \
  --config configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml \
  --review-manifest /home/ubuntu/scratch/sr-diffusion/review_sets/detail_v1/review_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/review_outputs/residual_refiner_v2_detail_v1

python tools/eval/eval_fixed_review_outputs.py \
  --review-manifest /home/ubuntu/scratch/sr-diffusion/review_sets/detail_v1/review_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/review_reports/residual_refiner_v2_detail_v1 \
  --candidate condition=/home/ubuntu/scratch/sr-diffusion/review_outputs/residual_refiner_v2_detail_v1/samples/{id}/condition.png \
  --candidate refined=/home/ubuntu/scratch/sr-diffusion/review_outputs/residual_refiner_v2_detail_v1/samples/{id}/refined.png
```

`eval_fixed_review_outputs.py` always reports PSNR, SSIM, Laplacian/detail
energy ratios, and highpass L1. Optional metrics such as `lpips`, `dists`,
`maniqa`, or `musiq` can be requested with `--optional-metric`; missing optional
packages are reported in `summary.json` instead of failing the run.

High-frequency detail branch v1:

```bash
python tools/train/train_detail_branch.py \
  --config configs/detail_branch_v1_photo130k_lsdir.yaml

python tools/eval/run_fixed_review_detail_branch.py \
  --config configs/detail_branch_v1_photo130k_lsdir.yaml \
  --checkpoint /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1_photo130k_lsdir/checkpoints/best_eval_detail.pt \
  --review-manifest /home/ubuntu/scratch/sr-diffusion/review_sets/detail_v1/review_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/review_outputs/detail_branch_v1_detail_v1
```

The detail branch is image-space and deterministic:
`LR -> Stage 2 dual-context condition -> Stage 1 decoder -> base SR -> detail branch`.
It does not run Stage 3/4 diffusion sampling.

Repository-operation utilities such as dataset downloads, manifest generation,
Hugging Face uploads, and W&B organization remain in `scripts/`.
