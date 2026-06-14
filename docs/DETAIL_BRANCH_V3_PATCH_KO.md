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

## v3 설정

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

## v3 결과

v3는 5k step까지 완료했다. best checkpoint는 step 1000이었다.

```text
run: /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v3_masked_patch_gan_probe
wandb: https://wandb.ai/jwheo/LuSIR/runs/jjz0ylip
best checkpoint: checkpoints/best_eval_detail.pt
val100:
  detail_score:          26.699334
  PSNR delta vs base:   +0.18418 dB
  SSIM delta vs base:   +0.00718
  lowpass_drift_l1:      0.000189
  outside_mask_l1:       0.004150
  wins:                  100/100
formal 219 benchmark vs v2:
  Y PSNR:               +0.00667 dB
  Y SSIM:               -0.000234
  RGB PSNR:             +0.00470 dB
```

v3는 안정적이지만 너무 보수적이다. artifact나 lowpass drift는 없었지만,
사용자가 문제로 본 missing fine texture를 눈에 띄게 복구하지 못했다. 따라서
v3 best를 public/default로 올리지 않고 더 강한 objective probe로 넘어간다.

## v3b stronger-detail 설정

```text
config: configs/detail_branch_v3b_stronger_patch_gan_probe.yaml
init: /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v3_masked_patch_gan_probe/checkpoints/best_eval_detail.pt
maximum: 8000 micro-steps
adversarial warmup: first 250 micro-steps
```

v3b는 구조를 키우지 않고 objective balance만 바꾼다.

- `masked_perceptual_weight`: `0.05 -> 0.20`
- VGG layer: `[3, 8, 15] -> [3, 8, 15, 22]`
- `adversarial.generator_weight`: `0.005 -> 0.02`
- discriminator: `base_channels 32 -> 48`
- `image/residual/highpass` anchor는 낮추고, `laplacian` weight는 올린다.
- `lowpass_anchor`, learned mask, bounded highpass residual은 유지한다.

v3b의 성공 기준은 단순 PSNR `+0.00x dB`가 아니다. grid에서 실제 texture
correction이 보여야 하며, 동시에 `lowpass_drift_l1`,
`outside_mask_residual_l1`, PSNR/SSIM guardrail이 크게 악화되지 않아야 한다.

## v3b 결과와 v4 전환

v3b는 8k step까지 완료했지만 장기적으로 실패했다. best는 step 500 근처였고
v3보다 수치상 아주 조금 높았으나, grid에서 visible texture correction은 거의
없었다. 마지막 step 8000은 PSNR/SSIM과 wins가 크게 무너졌다.

```text
best step: 500
best val100:
  PSNR delta vs base:   +0.18647 dB
  SSIM delta vs base:   +0.00770
  lowpass_drift_l1:      0.000212
  outside_mask_l1:       0.004600
  wins:                  100/100
final step: 8000
final val100:
  PSNR delta vs base:   -0.18243 dB
  SSIM delta vs base:   -0.00113
  lowpass_drift_l1:      0.000354
  outside_mask_l1:       0.007939
  wins:                  11/100
```

따라서 다음 실험은 stronger GAN continuation이 아니라 teacher-highpass
distillation probe다. `RealESRGAN_x4plus` output을 전체 정답으로 쓰지 않고,
GT와 비교해 locally no worse인 위치의 고주파 residual만 보조 signal로 사용한다.
config는 `configs/detail_branch_v4_teacher_highpass_realesrgan_probe.yaml`이다.

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

v3b를 시작할 때:

```bash
source /home/ubuntu/venvs/cuda/bin/activate
python -u tools/train/train_detail_branch.py \
  --config configs/detail_branch_v3b_stronger_patch_gan_probe.yaml
```
