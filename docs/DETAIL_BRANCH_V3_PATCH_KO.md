# Masked Detail Branch v3 Patch Probe

## 목적

masked detail branch v2는 learned mask로 수정 위치를 잘 제한했고 정식 219장
benchmark에서도 v1d를 소폭 개선했다. 하지만 L1/highpass 계열 손실만 계속
최적화해서는 사용자가 지적한 missing fine texture를 만들지 못했다.

v3 probe는 선택된 v2 step 38000을 그대로 초기값으로 사용하면서 다음 두
supervision만 보수적으로 추가한다.

1. learned detail mask로 위치를 가중한 frozen VGG feature loss
2. frozen base SR와 후보 이미지의 high-frequency 성분을 함께 보는 작은
   conditional PatchGAN

생성기는 새 구조로 교체하지 않는다. 기존 branch의 bounded residual,
highpass-only residual, learned soft mask와 `0.05` floor를 유지한다. 따라서
perceptual/adversarial supervision이 과하게 작동해도 저주파 색상과 전체
구조를 바꾸기 어렵게 제한되어 있다.

## 설정

```text
config: configs/detail_branch_v3_masked_patch_gan_probe.yaml
init: checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt
mask: checkpoints/detail_mask_predictor_v1_best3250.pt
maximum: 5000 micro-steps
adversarial warmup: first 500 micro-steps
```

기존 v2 설정은 새 옵션을 정의하지 않으므로 동작이 바뀌지 않는다.

## 관찰 지표

- `eval/sr_vs_base_mean_psnr`, `eval/sr_vs_base_ssim`: fidelity가 무너지지
  않는지 확인한다.
- `eval/sr_vs_base_highpass_l1`, `eval/sr_vs_base_laplacian_l1`: GT에 맞는
  고주파 correction인지 확인한다.
- `eval/lowpass_drift_l1`: branch가 frozen base의 저주파를 얼마나 바꾸는지
  확인한다.
- `eval/outside_mask_residual_l1`: learned mask가 낮은 영역에 residual이
  새는지 확인한다.
- `train/masked_perceptual`, `train/adversarial`, `train/discriminator`:
  새 objective가 실제로 활성화되고 발산하지 않는지 확인한다.
- W&B `samples/eval_grid`: 수치보다 먼저 반복 무늬, 흰 점, ringing,
  가짜 edge가 생기는지 확인한다.

## 중단 기준

5k step은 성능 보장이 아니라 방향성 probe의 상한이다. 다음 중 하나면 조기
중단한다.

- val100 PSNR/SSIM이 v2 시작점보다 지속적으로 하락한다.
- `lowpass_drift_l1` 또는 `outside_mask_residual_l1`가 계속 상승한다.
- grid에서 texture가 아니라 반복 패턴, ringing, 흰 점이 나타난다.
- discriminator loss가 거의 0으로 붕괴하거나 generator adversarial loss가
  계속 급증한다.

probe가 통과해야만 장기 학습과 정식 219장 benchmark를 진행한다.

## 실행

장기 학습을 시작하기 전 smoke:

```bash
source /home/ubuntu/venvs/cuda/bin/activate
python -u tools/train/train_detail_branch.py \
  --config configs/detail_branch_v3_masked_patch_gan_probe.yaml \
  --limit-steps 4 \
  --skip-initial-eval \
  --adversarial-start-step 0 \
  --disable-wandb \
  --output-dir /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v3_masked_patch_gan_smoke
```

probe를 시작할 때:

```bash
source /home/ubuntu/venvs/cuda/bin/activate
python -u tools/train/train_detail_branch.py \
  --config configs/detail_branch_v3_masked_patch_gan_probe.yaml
```
