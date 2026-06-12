# LuSIR Hugging Face Artifacts

Hugging Face is used as LuSIR checkpoint storage for artifacts that should survive
scratch disk loss. The default target is a public model repository:

```text
jwheo/LuSIR
```

The repository id is `jwheo/LuSIR`. Older `jwheo/sr-diffusion` links may still
resolve through Hugging Face redirects, but new scripts and docs should use the
LuSIR id.

Keep dataset files and validation images out of the Hub repository unless their
licenses are reviewed. Upload configs, metrics, and selected checkpoints only.

The Hub repository uses `license: cc-by-nc-4.0` for checkpoints and artifacts.
The source code is separately licensed under PolyForm Noncommercial 1.0.0.

## Auth

Check the active login:

```bash
/home/jwheojjang/venvs/rocm/bin/python - <<'PY'
from huggingface_hub import whoami
print(whoami()["name"])
PY
```

## Download For Inference

From a fresh GitHub clone, install dependencies and download the selected public
prototype checkpoints:

```bash
python scripts/download_hf_checkpoints.py
```

For the current Colab/default deterministic path, download the residual refiner
v2 artifacts:

```bash
python scripts/download_hf_checkpoints.py --preset residual_refiner_v2
```

Download the latest residual diagnostic/refiner artifact set:

```bash
python scripts/download_hf_checkpoints.py --preset residual_refiner_stage2_xl_mild
```

Download the selected teacher-supervised Stage 4 probe artifact set:

```bash
python scripts/download_hf_checkpoints.py --preset stage4_teacher_residual_probe
```

Download the completed dual-context LSDIR Stage 2 artifact set:

```bash
python scripts/download_hf_checkpoints.py --preset stage2_photo130k_lsdir_dual
```

Download the completed detail branch v1b review artifact set:

```bash
python scripts/download_hf_checkpoints.py --preset detail_branch_v1b
```

V1b is the latest public detail-branch artifact. V1c and the active v1d
capacity experiment remain local research runs until checkpoint selection and
fixed visual review are complete.

Each preset creates the local `checkpoints/`, `configs/`, `metrics/`, and
`samples/` files expected by its matching inference or review config.

Run the residual refiner default path:

```bash
python tools/infer/infer_residual_refiner.py \
  --config configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/demo
```

Run the Stage 4 condition-start prototype for diffusion comparison:

```bash
python tools/infer/infer_diffusion.py \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/demo
```

Run the same checkpoint in tiled mode for larger LR images:

```bash
python tools/infer/infer_diffusion.py \
  --input-lr /path/to/larger_lr.png \
  --output-dir outputs/tiled_demo \
  --tile \
  --tile-overlap 32
```

The default `tools/infer/infer_diffusion.py` config is the HF-friendly Stage 4 config. It
uses relative checkpoint paths, so it works outside the original training VM.

## Upload Selected Artifacts

Upload the selected Stage 1 VAE and the current Stage 2 checkpoint:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --update-card \
  --message "Upload Stage 1 and Stage 2 checkpoints" \
  --artifact LICENSE=LICENSE \
  --artifact CHECKPOINT_LICENSE.md=CHECKPOINT_LICENSE.md \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt=checkpoints/stage1_autoencoder_best_eval_recon.pt \
  --artifact configs/autoencoder_photo10k.yaml=configs/autoencoder_photo10k.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/latest.pt=checkpoints/stage2_latent_pretrain_latest.pt \
  --artifact configs/latent_pretrain_photo10k.yaml=configs/latent_pretrain_photo10k.yaml
```

For a dry run:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --dry-run \
  --artifact configs/latent_pretrain_photo10k.yaml=configs/latent_pretrain_photo10k.yaml
```

Upload the current best sampled Stage 4 condition-start checkpoint and metrics:

```bash
/home/jwheojjang/venvs/rocm/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload Stage 4 condition-start checkpoint" \
  --artifact configs/diffusion_photo10k_b32_stage4_condition.yaml=configs/diffusion_photo10k_b32_stage4_condition.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt=checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_stage4_condition_val100_t25_32step/summary.json=metrics/stage4_condition_val100_t25_32step_summary.json
```

Upload the current photo100k Stage 4 condition-start checkpoint and the v2
degradation config:

```bash
/home/jwheojjang/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload photo100k Stage 4 and v2 configs" \
  --artifact configs/hf/diffusion_photo100k_stage4_condition.yaml=configs/hf/diffusion_photo100k_stage4_condition.yaml \
  --artifact configs/diffusion_photo100k_b32_stage4_condition.yaml=configs/diffusion_photo100k_b32_stage4_condition.yaml \
  --artifact configs/latent_pretrain_photo100k_v2.yaml=configs/latent_pretrain_photo100k_v2.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt=checkpoints/stage4_photo100k_condition_b32_best_eval_condition_decoded.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_photo100k_stage4_condition_val100_t25_32step_final/summary.json=metrics/stage4_photo100k_condition_val100_t25_32step_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/compare_photo100k_stage3_vs_stage4_condition_val100/summary.json=metrics/stage4_photo100k_condition_compare_stage3_summary.json
```

Upload the photo100k Stage 2 `photo_v2` condition encoder and follow-up v2
diffusion configs:

```bash
/home/jwheojjang/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload photo100k Stage 2 v2 condition encoder" \
  --artifact configs/hf/diffusion_photo100k_v2.yaml=configs/hf/diffusion_photo100k_v2.yaml \
  --artifact configs/diffusion_photo100k_b32_v2.yaml=configs/diffusion_photo100k_b32_v2.yaml \
  --artifact configs/diffusion_photo100k_b32_stage4_condition_v2.yaml=configs/diffusion_photo100k_b32_stage4_condition_v2.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt=checkpoints/stage2_photo100k_v2_b64_best_eval_latent.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/summary.json=metrics/stage2_photo100k_v2_b64_summary.json
```

Upload the photo100k Stage 3 `photo_v2` diffusion checkpoint and sampled eval:

```bash
/home/jwheojjang/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload photo100k Stage 3 v2 diffusion checkpoint" \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/checkpoints/best_eval_noise.pt=checkpoints/stage3_photo100k_v2_b32_best_eval_noise.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/summary.json=metrics/stage3_photo100k_v2_b32_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_photo100k_v2_val100_t50_32step_gpu/summary.json=metrics/stage3_photo100k_v2_val100_t50_32step_summary.json
```

Upload the photo100k Stage 4 `photo_v2` condition-start checkpoint and sampled eval:

```bash
/home/jwheojjang/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload photo100k Stage 4 v2 condition checkpoint" \
  --artifact configs/hf/diffusion_photo100k_stage4_condition_v2.yaml=configs/hf/diffusion_photo100k_stage4_condition_v2.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints/best_eval_condition_decoded.pt=checkpoints/stage4_photo100k_condition_v2_b32_best_eval_condition_decoded.pt \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/summary.json=metrics/stage4_photo100k_condition_v2_b32_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_photo100k_stage4_condition_v2_val100_t25_32step/summary.json=metrics/stage4_photo100k_condition_v2_val100_t25_32step_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/compare_photo100k_stage3_v2_vs_stage4_condition_v2_val100/summary.json=metrics/stage4_photo100k_condition_v2_compare_stage3_v2_summary.json
```

Upload the Stage2 residual diagnostic and deterministic residual refiner probe
artifacts:

```bash
/home/ubuntu/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload Stage2 residual refiner probe artifacts" \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_probe/checkpoints/best_eval_refined.pt=checkpoints/residual_refiner_stage2_xl_mild_best_eval_refined.pt \
  --artifact configs/residual_refiner_stage2_xl_mild_probe.yaml=configs/residual_refiner_stage2_xl_mild_probe.yaml \
  --artifact configs/residual_refiner_stage2_xl_mild_open_gate_probe.yaml=configs/residual_refiner_stage2_xl_mild_open_gate_probe.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diagnose_stage2_xl_residuals_mild_val100/summary.json=metrics/diagnose_stage2_xl_residuals_mild_val100_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diagnose_stage2_xl_residuals_mild_val100/metrics.csv=metrics/diagnose_stage2_xl_residuals_mild_val100_metrics.csv \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diagnose_stage2_xl_residuals_mild_val100/residual_diagnostic_grid.png=samples/diagnose_stage2_xl_residuals_mild_val100_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_probe/early_stop_summary.json=metrics/residual_refiner_stage2_xl_mild_probe_early_stop_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_probe/metrics.jsonl=metrics/residual_refiner_stage2_xl_mild_probe_metrics.jsonl \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_probe/eval_step_000500/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/residual_refiner_stage2_xl_mild_probe_step500_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_open_gate_probe/early_stop_summary.json=metrics/residual_refiner_stage2_xl_mild_open_gate_probe_early_stop_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_open_gate_probe/metrics.jsonl=metrics/residual_refiner_stage2_xl_mild_open_gate_probe_metrics.jsonl \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_open_gate_probe/eval_step_000500/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/residual_refiner_stage2_xl_mild_open_gate_probe_step500_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_stage2_xl_mild_val100/summary.json=metrics/eval_residual_refiner_stage2_xl_mild_val100_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_stage2_xl_photo_v2_val100/summary.json=metrics/eval_residual_refiner_stage2_xl_photo_v2_val100_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100/summary.json=metrics/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_stage2_xl_mild_val100/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/eval_residual_refiner_stage2_xl_mild_val100_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_stage2_xl_photo_v2_val100/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/eval_residual_refiner_stage2_xl_photo_v2_val100_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/compare_residual_refiner_vs_stage4_edge_0801_photo_v3.png=samples/compare_residual_refiner_vs_stage4_edge_0801_photo_v3.png
```

Upload the selected teacher-supervised Stage 4 probe checkpoint and sampled
evaluation artifacts:

```bash
/home/ubuntu/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload Stage4 teacher residual probe artifacts" \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe/checkpoints/step_0002000.pt=checkpoints/stage4_photo100k_xl_teacher_residual_photo_v3_step_0002000.pt \
  --artifact configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml=configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_teacher_residual_photo_v3_step2000_val100_t25/summary.json=metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_32step_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_teacher_residual_photo_v3_step2000_val100_t50/summary.json=metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t50_32step_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_teacher_residual_photo_v3_step2000_val100_t25/grid_lr_bicubic_sr_gt.png=samples/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_grid.png
```

Upload the selected detail-preserving Stage 4 checkpoint and sampled evaluation:

```bash
/home/ubuntu/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload detail-preserving Stage4 artifacts" \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long/checkpoints/best_eval_condition_decoded.pt=checkpoints/stage4_photo100k_xl_teacher_residual_photo_detail_best8000.pt \
  --artifact configs/degradation_presets.yaml=configs/degradation_presets.yaml \
  --artifact configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml=configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_stage4_teacher_photo_detail_long_best8000_val100_t25/summary.json=metrics/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_stage4_teacher_photo_detail_long_best8000_val100_t25/grid_lr_bicubic_sr_gt.png=samples/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_grid.png
```

Upload the selected decoded-detail residual refiner v2 step 39000 and cross-preset evals:

```bash
/home/ubuntu/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload residual refiner v2 best39000 artifacts" \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_photo_detail_v2_long/checkpoints/best_eval_refined.pt=checkpoints/residual_refiner_stage2_xl_photo_detail_v2_best39000.pt \
  --artifact configs/residual_refiner_stage2_xl_photo_detail_v2_continue_40k.yaml=configs/residual_refiner_stage2_xl_photo_detail_v2_continue_40k.yaml \
  --artifact configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml=configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_photo_detail_v2_long/summary.json=metrics/residual_refiner_stage2_xl_photo_detail_v2_long_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_v2_best39000_strength_sweep_summary.json=metrics/residual_refiner_v2_best39000_strength_sweep_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100/summary.json=metrics/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100/summary.json=metrics/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100/summary.json=metrics/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_summary.json \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_photo_detail_v2_long/eval_step_039000/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/residual_refiner_stage2_xl_photo_detail_v2_best39000_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100_grid.png \
  --artifact /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100/eval_grid_lr_bicubic_condition_refined_oracle_gt.png=samples/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_grid.png
```

Upload the completed, non-promoted Stage 2 perceptual continuation candidate:

```bash
/home/ubuntu/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Upload Stage2 perceptual continuation comparison" \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue/checkpoints/step_0008000.pt=checkpoints/stage2_photo100k_multiscale_hqmix_perceptual_step_0008000.pt \
  --artifact configs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue.yaml=configs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue.yaml \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_perceptual_candidates/photo_detail_mix/stage2_xl_candidate_metrics.json=metrics/stage2_multiscale_perceptual_photo_detail_mix_candidates.json \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_perceptual_candidates/mild/stage2_xl_candidate_metrics.json=metrics/stage2_multiscale_perceptual_mild_candidates.json \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_perceptual_candidates/photo_v2/stage2_xl_candidate_metrics.json=metrics/stage2_multiscale_perceptual_photo_v2_candidates.json \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_perceptual_candidates/photo_v3_noise_mix/stage2_xl_candidate_metrics.json=metrics/stage2_multiscale_perceptual_photo_v3_noise_mix_candidates.json \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_perceptual_candidates/photo_detail_mix/stage2_xl_candidate_contact_sheet.png=samples/stage2_multiscale_perceptual_photo_detail_mix_candidates.png \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_perceptual_candidates/photo_v3_noise_mix/stage2_xl_candidate_contact_sheet.png=samples/stage2_multiscale_perceptual_photo_v3_noise_mix_candidates.png
```

This checkpoint is preserved for research comparison only. It is not the
public Colab default and did not produce a meaningful visible detail gain.

Upload the completed dual-context LSDIR Stage 2 best checkpoint, final summary,
and cross-preset contact sheets:

```bash
/home/ubuntu/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Add completed dual Stage2 LSDIR results" \
  --artifact metrics/stage2_photo130k_lsdir_dual_multiscale_final_summary.json=metrics/stage2_photo130k_lsdir_dual_multiscale_final_summary.json \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_multiscale_long/checkpoints/best_eval_decoded.pt=checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_dual_lsdir_photo_detail_mix/stage2_xl_candidate_contact_sheet.png=samples/stage2_dual_lsdir_photo_detail_mix_best98k_final100k_contact_sheet.png \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_dual_lsdir_mild/stage2_xl_candidate_contact_sheet.png=samples/stage2_dual_lsdir_mild_best98k_final100k_contact_sheet.png \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_dual_lsdir_photo_v2/stage2_xl_candidate_contact_sheet.png=samples/stage2_dual_lsdir_photo_v2_best98k_final100k_contact_sheet.png \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/compare_stage2_dual_lsdir_photo_v3_noise_mix/stage2_xl_candidate_contact_sheet.png=samples/stage2_dual_lsdir_photo_v3_noise_mix_best98k_final100k_contact_sheet.png
```

This checkpoint is preserved for research comparison and human visual review.
It is not the current public Colab default.

Upload the completed high-frequency detail branch v1b selected checkpoint and
review artifacts:

```bash
/home/ubuntu/venvs/cuda/bin/python scripts/upload_hf_artifact.py \
  --repo-id jwheo/LuSIR \
  --repo-type model \
  --message "Add detail branch v1b best39500 results" \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1b_aug_photo130k_lsdir/checkpoints/best_eval_detail.pt=checkpoints/detail_branch_v1b_aug_photo130k_lsdir_best39500.pt \
  --artifact configs/detail_branch_v1b_aug_photo130k_lsdir.yaml=configs/detail_branch_v1b_aug_photo130k_lsdir.yaml \
  --artifact configs/hf/detail_branch_v1b_aug_photo130k_lsdir.yaml=configs/hf/detail_branch_v1b_aug_photo130k_lsdir.yaml \
  --artifact metrics/detail_branch_v1b_aug_photo130k_lsdir_summary.json=metrics/detail_branch_v1b_aug_photo130k_lsdir_summary.json \
  --artifact /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1b_aug_photo130k_lsdir/eval_step_039500/eval_grid_lr_bicubic_base_detail_residual_gt.png=samples/detail_branch_v1b_aug_photo130k_lsdir_best39500_grid.png
```

This checkpoint is the latest public detail artifact. V1c and the active v1d
capacity run remain local research candidates. It is not yet the public Colab
default because the WebUI does not have a detail branch single-image/tiled
inference runner.

Make the Hub repository public after license files and the model card are in
place:

```bash
/home/jwheojjang/venvs/rocm/bin/python - <<'PY'
from huggingface_hub import HfApi
HfApi().update_repo_settings("jwheo/LuSIR", repo_type="model", private=False)
PY
```

## Policy

- Upload only checkpoints worth preserving.
- Prefer `best_eval_*.pt` over every intermediate step checkpoint.
- Use public visibility only with the non-commercial license files and model
  card in place.
- Do not upload raw training data. Upload generated validation/contact sheets
  only when they are selected review artifacts tied to a preserved checkpoint.
