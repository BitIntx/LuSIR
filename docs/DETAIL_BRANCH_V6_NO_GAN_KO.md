# Detail Branch v6 No-GAN Teacher/Negative Probe

## 목적

v5는 v2 noise-negative mask의 hard top10 gate 자체는 성공했지만, PatchGAN이
켜진 뒤 mask 내부에서 긁힌 듯한 고주파 artifact를 만들며 붕괴했다. 실패
원인은 gate 밖으로 residual이 새는 것이 아니라, gate 안쪽에서 adversarial
pressure가 GT-aligned correction 대신 artifact를 키운 것이다.

v6는 같은 top10 gate를 유지하되 GAN을 제거한다. 대신 RealESRGAN teacher를
전체 정답으로 모방하지 않고, GT와 비교해 locally no worse인 위치의 고주파
신호만 약하게 사용한다. 동시에 target detail 근거가 약하거나 base가 이미
고주파를 과하게 가진 위치에서는 residual 자체를 누르는 negative residual
loss를 추가한다.

## 설정

```text
config: configs/detail_branch_v6_noise_gate_teacher_perceptual_no_gan_probe.yaml
init: checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt
mask: checkpoints/detail_mask_predictor_v2_noise_negative_best1500.pt
teacher cache: /home/ubuntu/scratch/sr-diffusion/teacher_cache/realesrgan_x4plus_photo_detail_mix_2048
maximum: 6000 micro-steps
GAN: disabled
```

핵심 차이:

- `detail_mask.top_fraction: 0.10`, `top_mode: binary`, `floor: 0.0`
- `masked_perceptual_weight: 0.10`
- `teacher_residual_weight: 0.20`
- `teacher_highpass_weight: 0.08`
- `teacher_gt_filter: true`
- `negative_residual_weight: 0.75`
- `adversarial.enabled: false`

새 loss 항은 `tools/train/train_detail_branch.py`의
`artifact_negative_residual_loss`다. GT target highpass가 약한 평탄 영역과,
base highpass가 GT보다 이미 과한 영역에서 residual highpass 에너지를 직접
줄인다. mask loss처럼 정규화하지 않고 전체 평균으로 계산하므로, unsafe
weight가 낮은 배치에서는 loss 자체도 낮아진다.

## 성공 조건

v6의 성공 기준은 `+0.00x dB`만이 아니다. W&B sample grid에서 artifact 없이
실제 texture correction이 보여야 한다.

- `eval/sr_vs_base_mean_psnr` 양수 유지
- `eval/sr_vs_base_ssim` 양수 유지
- `eval/detail_mask_mean` 약 `0.10`
- `eval/outside_mask_residual_l1`가 `0` 근처
- `eval/lowpass_drift_l1`이 `0.00015` 안팎에서 크게 증가하지 않음
- `train/negative_residual`과 `train/negative_weight`가 non-zero로 로깅됨
- grid에서 노이즈 patch, 평탄 피부/하늘, 압축 artifact에 긁힌 residual이 생기지 않음

중단 기준:

- `eval/sr_vs_base_mean_psnr`가 연속 eval에서 음수로 내려감
- `eval/wins_vs_base`가 빠르게 감소
- `eval/lowpass_drift_l1`이 `0.0002` 이상으로 증가
- W&B sample grid에서 v5와 비슷한 scratch/noise artifact가 보임

## 실행

Smoke:

```bash
source /home/ubuntu/venvs/cuda132/bin/activate
python -u tools/train/train_detail_branch.py \
  --config configs/detail_branch_v6_noise_gate_teacher_perceptual_no_gan_probe.yaml \
  --limit-steps 4 \
  --skip-initial-eval \
  --disable-wandb \
  --output-dir /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v6_noise_gate_teacher_perceptual_no_gan_smoke
```

Probe:

```bash
source /home/ubuntu/venvs/cuda132/bin/activate
python -u tools/train/train_detail_branch.py \
  --config configs/detail_branch_v6_noise_gate_teacher_perceptual_no_gan_probe.yaml
```

## 2026-06-22 smoke 결과

4-step smoke는 정상 동작했다.

```text
eval/sr_vs_base_psnr:      +0.0515 dB
eval/sr_vs_base_mean_psnr: +0.0696 dB
eval/sr_vs_base_ssim:      +0.00147
eval/detail_mask_mean:      0.1000
eval/outside_mask_residual_l1: 0.000000
eval/lowpass_drift_l1:      0.000130
eval/wins_vs_base:          93/100
```

이 수치는 v2/top10 초기 상태가 깨지지 않았다는 smoke 확인일 뿐이다. 실제
판정은 500-6000 step 사이의 W&B sample grid와 val100 추세로 한다.
