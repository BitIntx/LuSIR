from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


PROTOTYPE_FILES = [
    "checkpoints/stage1_autoencoder_best_eval_recon.pt",
    "checkpoints/stage2_latent_pretrain_best_eval_latent.pt",
    "checkpoints/stage3_diffusion_b32_best_eval_noise.pt",
    "checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt",
    "CHECKPOINT_LICENSE.md",
    "LICENSE",
]

PHOTO100K_FILES = [
    *PROTOTYPE_FILES,
    "checkpoints/stage2_photo100k_b64_best_eval_latent.pt",
    "checkpoints/stage2_photo100k_v2_b64_best_eval_latent.pt",
    "checkpoints/stage2_photo100k_v3_noise_xl_b64_best_eval_latent.pt",
    "checkpoints/stage3_photo100k_b32_best_eval_noise.pt",
    "checkpoints/stage3_photo100k_v2_b32_best_eval_noise.pt",
    "checkpoints/stage4_photo100k_condition_b32_best_eval_condition_decoded.pt",
    "checkpoints/stage4_photo100k_condition_v2_b32_best_eval_condition_decoded.pt",
    "configs/latent_pretrain_photo100k.yaml",
    "configs/latent_pretrain_photo100k_v2.yaml",
    "configs/latent_pretrain_photo100k_v3_noise_xl.yaml",
    "configs/diffusion_photo100k_b32.yaml",
    "configs/diffusion_photo100k_b32_v2.yaml",
    "configs/diffusion_photo100k_b32_stage4_condition.yaml",
    "configs/diffusion_photo100k_b32_stage4_condition_v2.yaml",
    "configs/diffusion_photo100k_xl_stage4_condition_v3.yaml",
    "configs/hf/diffusion_photo100k_stage4_condition.yaml",
    "configs/hf/diffusion_photo100k_stage4_condition_v2.yaml",
    "configs/hf/diffusion_photo100k_v2.yaml",
    "metrics/stage2_photo100k_b64_summary.json",
    "metrics/stage2_photo100k_v2_b64_summary.json",
    "metrics/stage2_photo100k_v3_noise_xl_b64_summary.json",
    "metrics/stage3_photo100k_b32_summary.json",
    "metrics/stage3_photo100k_v2_b32_summary.json",
    "metrics/stage3_photo100k_v2_val100_t50_32step_summary.json",
    "metrics/stage4_photo100k_condition_val100_t25_32step_summary.json",
    "metrics/stage4_photo100k_condition_compare_stage3_summary.json",
    "metrics/stage4_photo100k_condition_v2_b32_summary.json",
    "metrics/stage4_photo100k_condition_v2_val100_t25_32step_summary.json",
    "metrics/stage4_photo100k_condition_v2_compare_stage3_v2_summary.json",
]

PHOTO100K_XL_CANDIDATE_FILES = [
    *PHOTO100K_FILES,
    "checkpoints/stage2_photo100k_v3_noise_xl_b64_step_0072000.pt",
    "checkpoints/stage2_photo100k_v3_noise_xl_b64_latest.pt",
]

PHOTO100K_XL_STAGE4_EDGE_FILES = [
    *PHOTO100K_XL_CANDIDATE_FILES,
    "checkpoints/stage4_photo100k_xl_edge_b16_best_eval_condition_decoded.pt",
    "configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml",
    "metrics/stage4_photo100k_xl_edge_b16_val100_t50_32step_summary.json",
    "samples/stage4_photo100k_xl_edge_b16_val100_t50_32step_grid_lr_bicubic_sr_gt.png",
]

RESIDUAL_REFINER_STAGE2_XL_MILD_FILES = [
    *PHOTO100K_XL_STAGE4_EDGE_FILES,
    "checkpoints/residual_refiner_stage2_xl_mild_best_eval_refined.pt",
    "configs/residual_refiner_stage2_xl_mild_probe.yaml",
    "configs/residual_refiner_stage2_xl_mild_open_gate_probe.yaml",
    "metrics/diagnose_stage2_xl_residuals_mild_val100_summary.json",
    "metrics/diagnose_stage2_xl_residuals_mild_val100_metrics.csv",
    "samples/diagnose_stage2_xl_residuals_mild_val100_grid.png",
    "metrics/residual_refiner_stage2_xl_mild_probe_early_stop_summary.json",
    "metrics/residual_refiner_stage2_xl_mild_probe_metrics.jsonl",
    "samples/residual_refiner_stage2_xl_mild_probe_step500_grid.png",
    "metrics/residual_refiner_stage2_xl_mild_open_gate_probe_early_stop_summary.json",
    "metrics/residual_refiner_stage2_xl_mild_open_gate_probe_metrics.jsonl",
    "samples/residual_refiner_stage2_xl_mild_open_gate_probe_step500_grid.png",
    "metrics/eval_residual_refiner_stage2_xl_mild_val100_summary.json",
    "metrics/eval_residual_refiner_stage2_xl_photo_v2_val100_summary.json",
    "metrics/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_summary.json",
    "samples/eval_residual_refiner_stage2_xl_mild_val100_grid.png",
    "samples/eval_residual_refiner_stage2_xl_photo_v2_val100_grid.png",
    "samples/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_grid.png",
    "samples/compare_residual_refiner_vs_stage4_edge_0801_photo_v3.png",
]

STAGE4_TEACHER_RESIDUAL_PROBE_FILES = [
    *RESIDUAL_REFINER_STAGE2_XL_MILD_FILES,
    "checkpoints/stage4_photo100k_xl_teacher_residual_photo_v3_step_0002000.pt",
    "configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml",
    "metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_32step_summary.json",
    "metrics/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t50_32step_summary.json",
    "samples/stage4_photo100k_xl_teacher_residual_photo_v3_step2000_val100_t25_grid.png",
]

STAGE4_PHOTO_DETAIL_FILES = [
    *STAGE4_TEACHER_RESIDUAL_PROBE_FILES,
    "checkpoints/stage4_photo100k_xl_teacher_residual_photo_detail_best8000.pt",
    "configs/degradation_presets.yaml",
    "configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml",
    "metrics/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_summary.json",
    "samples/stage4_photo100k_xl_teacher_residual_photo_detail_best8000_val100_t25_grid.png",
]

RESIDUAL_REFINER_V2_FILES = [
    *STAGE4_PHOTO_DETAIL_FILES,
    "checkpoints/residual_refiner_stage2_xl_photo_detail_v2_best39000.pt",
    "configs/residual_refiner_stage2_xl_photo_detail_v2_continue_40k.yaml",
    "configs/hf/residual_refiner_stage2_xl_photo_detail_v2.yaml",
    "metrics/residual_refiner_stage2_xl_photo_detail_v2_long_summary.json",
    "metrics/residual_refiner_v2_best39000_strength_sweep_summary.json",
    "metrics/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100_summary.json",
    "metrics/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100_summary.json",
    "metrics/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_summary.json",
    "samples/residual_refiner_stage2_xl_photo_detail_v2_best39000_grid.png",
    "samples/eval_residual_refiner_v2_best39000_stage2_xl_mild_val100_grid.png",
    "samples/eval_residual_refiner_v2_best39000_stage2_xl_photo_v2_val100_grid.png",
    "samples/eval_residual_refiner_v2_best39000_stage2_xl_photo_v3_noise_mix_val100_grid.png",
]

STAGE2_MULTISCALE_HQMIX_FILES = [
    *PHOTO100K_XL_CANDIDATE_FILES,
    "checkpoints/stage2_photo100k_multiscale_hqmix_step_0046000.pt",
    "configs/latent_pretrain_photo100k_multiscale_hqmix_long.yaml",
    "configs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue.yaml",
    "metrics/stage2_multiscale_hqmix_step46000_cross_preset_summary.json",
    "samples/stage2_multiscale_hqmix_checkpoint_comparison.png",
]

STAGE2_MULTISCALE_PERCEPTUAL_FILES = [
    *STAGE2_MULTISCALE_HQMIX_FILES,
    "checkpoints/stage2_photo100k_multiscale_hqmix_perceptual_step_0008000.pt",
    "metrics/stage2_multiscale_perceptual_photo_detail_mix_candidates.json",
    "metrics/stage2_multiscale_perceptual_mild_candidates.json",
    "metrics/stage2_multiscale_perceptual_photo_v2_candidates.json",
    "metrics/stage2_multiscale_perceptual_photo_v3_noise_mix_candidates.json",
    "samples/stage2_multiscale_perceptual_photo_detail_mix_candidates.png",
    "samples/stage2_multiscale_perceptual_photo_v3_noise_mix_candidates.png",
]

STAGE2_PHOTO130K_LSDIR_DUAL_FILES = [
    *STAGE2_MULTISCALE_PERCEPTUAL_FILES,
    "checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt",
    "configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml",
    "metrics/stage2_photo130k_lsdir_dual_multiscale_final_summary.json",
    "samples/stage2_dual_lsdir_photo_detail_mix_best98k_final100k_contact_sheet.png",
    "samples/stage2_dual_lsdir_mild_best98k_final100k_contact_sheet.png",
    "samples/stage2_dual_lsdir_photo_v2_best98k_final100k_contact_sheet.png",
    "samples/stage2_dual_lsdir_photo_v3_noise_mix_best98k_final100k_contact_sheet.png",
]

DETAIL_BRANCH_V1B_FILES = [
    *STAGE2_PHOTO130K_LSDIR_DUAL_FILES,
    "checkpoints/detail_branch_v1b_aug_photo130k_lsdir_best39500.pt",
    "configs/detail_branch_v1b_aug_photo130k_lsdir.yaml",
    "configs/hf/detail_branch_v1b_aug_photo130k_lsdir.yaml",
    "metrics/detail_branch_v1b_aug_photo130k_lsdir_summary.json",
    "samples/detail_branch_v1b_aug_photo130k_lsdir_best39500_grid.png",
]

DETAIL_BRANCH_V1D_FILES = [
    *DETAIL_BRANCH_V1B_FILES,
    "checkpoints/detail_branch_v1d_deep3m_photo130k_lsdir_best99500.pt",
    "configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml",
    "configs/hf/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml",
    "metrics/detail_branch_v1d_deep3m_photo130k_lsdir_3ep_summary.json",
    "metrics/benchmark_bicubic5_detail_v1d_best99500_summary.json",
    "metrics/formal_x4_benchmark_lusir_realesr_summary.json",
    "metrics/formal_x4_benchmark_lusir_realesr_metrics.csv",
    "metrics/formal_x4_benchmark_div2k_swinir_summary.json",
    "metrics/formal_x4_benchmark_div2k_swinir_metrics.csv",
    "samples/detail_branch_v1d_deep3m_photo130k_lsdir_best99500_grid.png",
    "samples/benchmark_bicubic5_detail_v1d_best99500_grid.png",
]

DETAIL_BRANCH_V2_MASKED_FILES = [
    *DETAIL_BRANCH_V1D_FILES,
    "checkpoints/detail_mask_predictor_v1_best3250.pt",
    "checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt",
    "configs/detail_branch_v2_masked_long_20ep.yaml",
    "configs/hf/detail_mask_predictor_v1.yaml",
    "configs/hf/detail_branch_v2_masked_photo130k_lsdir.yaml",
    "metrics/detail_branch_v2_masked_photo130k_lsdir_summary.json",
    "metrics/formal_x4_benchmark_detail_v2_masked_summary.json",
    "metrics/formal_x4_benchmark_detail_v2_masked_metrics.csv",
    "samples/detail_branch_v2_masked_photo130k_lsdir_best38000_grid.png",
]

DETAIL_MASK_PREDICTOR_V2_NOISE_NEGATIVE_FILES = [
    "checkpoints/stage1_autoencoder_best_eval_recon.pt",
    "checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt",
    "checkpoints/detail_mask_predictor_v1_best3250.pt",
    "checkpoints/detail_mask_predictor_v2_noise_negative_best1500.pt",
    "configs/autoencoder_photo10k.yaml",
    "configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml",
    "configs/detail_mask_predictor_v1_probe.yaml",
    "configs/detail_mask_predictor_v2_noise_negative_probe.yaml",
    "configs/hf/detail_mask_predictor_v1.yaml",
    "configs/hf/detail_mask_predictor_v2_noise_negative.yaml",
    "metrics/detail_mask_predictor_v1_val100_summary.json",
    "metrics/detail_mask_predictor_v2_noise_negative_summary.json",
    "metrics/detail_mask_predictor_v2_noise_response_top10_sigma010_summary.json",
    "samples/detail_mask_predictor_v2_noise_response_top10_sigma010_grid.png",
]

PRESETS = {
    "prototype": PROTOTYPE_FILES,
    "photo100k": PHOTO100K_FILES,
    "photo100k_xl_candidates": PHOTO100K_XL_CANDIDATE_FILES,
    "photo100k_xl_stage4_edge": PHOTO100K_XL_STAGE4_EDGE_FILES,
    "residual_refiner_stage2_xl_mild": RESIDUAL_REFINER_STAGE2_XL_MILD_FILES,
    "stage4_teacher_residual_probe": STAGE4_TEACHER_RESIDUAL_PROBE_FILES,
    "stage4_photo_detail": STAGE4_PHOTO_DETAIL_FILES,
    "residual_refiner_v2": RESIDUAL_REFINER_V2_FILES,
    "stage2_multiscale_hqmix": STAGE2_MULTISCALE_HQMIX_FILES,
    "stage2_multiscale_perceptual": STAGE2_MULTISCALE_PERCEPTUAL_FILES,
    "stage2_photo130k_lsdir_dual": STAGE2_PHOTO130K_LSDIR_DUAL_FILES,
    "detail_branch_v1b": DETAIL_BRANCH_V1B_FILES,
    "detail_branch_v1d": DETAIL_BRANCH_V1D_FILES,
    "detail_branch_v2_masked": DETAIL_BRANCH_V2_MASKED_FILES,
    "detail_mask_predictor_v2_noise_negative": DETAIL_MASK_PREDICTOR_V2_NOISE_NEGATIVE_FILES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public LuSIR Hugging Face inference artifacts.")
    parser.add_argument("--repo-id", default="jwheo/LuSIR")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="prototype",
        help=(
            "Artifact set to download. 'photo100k' includes selected handoff checkpoints; "
            "'photo100k_xl_candidates' also includes Stage 2 XL candidate condition encoders; "
            "'photo100k_xl_stage4_edge' includes the latest XL Stage 4 edge-loss checkpoint and eval artifacts; "
            "'residual_refiner_stage2_xl_mild' also includes residual diagnostic, deterministic refiner, "
            "and cross-degradation eval artifacts; 'stage4_teacher_residual_probe' also includes the selected "
            "teacher-supervised Stage 4 probe checkpoint and sampled eval artifacts; "
            "'stage4_photo_detail' includes the selected detail-preserving Stage 4 checkpoint; "
            "'residual_refiner_v2' includes the selected decoded-detail residual refiner and cross-preset evals; "
            "'stage2_multiscale_hqmix' includes the selected multiscale Stage 2 condition checkpoint; "
            "'stage2_multiscale_perceptual' includes the non-promoted VGG continuation step 8000 and comparisons; "
            "'stage2_photo130k_lsdir_dual' includes the completed dual-context LSDIR Stage 2 best98000 checkpoint; "
            "'detail_branch_v1b' includes the earlier high-frequency detail branch best39500 checkpoint; "
            "'detail_branch_v1d' includes the selected 3.02M detail branch best99500 checkpoint and eval artifacts; "
            "'detail_branch_v2_masked' includes the learned-mask-gated research candidate best38000; "
            "'detail_mask_predictor_v2_noise_negative' includes the noise-negative detail gate best1500 "
            "and its top10 noise-response review."
        ),
    )
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Specific repo file to download. Can be repeated and overrides --preset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.file or PRESETS[args.preset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        destination = args.output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        cached_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            filename=filename,
            local_dir=args.output_dir,
        )
        print(f"{filename} -> {cached_path}")


if __name__ == "__main__":
    main()
