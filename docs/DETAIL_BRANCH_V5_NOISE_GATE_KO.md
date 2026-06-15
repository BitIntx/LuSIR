# Detail Branch v5 Noise-Gated Top10 Probe

## 목적

v3/v3b/v4는 learned mask와 perceptual/adversarial/teacher signal을 넣었지만,
눈에 띄는 texture 생성으로 이어지지 않았다. 특히 v1 mask는 synthetic noise
patch에도 gate를 많이 열어 texture generator용 위치 제한으로는 위험했다.

v5는 generator 구조를 키우지 않고 gate 정의를 먼저 바꾼다.

- mask predictor: v2 noise-negative best1500
- mask policy: `top_fraction 0.10`, `top_mode binary`, `floor 0.0`
- init generator: selected deterministic masked v2 step38000
- supervision: masked VGG + small PatchGAN, v3와 v3b 사이의 중간 강도

즉, generator가 residual을 낼 수 있는 위치를 v2가 확신하는 top10 texture
후보로 제한하고, synthetic noise/excess 영역은 gate가 닫히도록 한다.

## 설정

```text
config: configs/detail_branch_v5_noise_gate_top10_patch_gan_probe.yaml
init: checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt
mask: checkpoints/detail_mask_predictor_v2_noise_negative_best1500.pt
maximum: 5000 micro-steps
adversarial warmup: first 500 micro-steps
```

v5는 v3b처럼 강한 GAN continuation을 오래 밀지 않는다. 성공 조건은
`+0.00x dB`가 아니라, W&B sample grid에서 실제 texture correction이 보이면서
다음 guardrail이 유지되는 것이다.

- `eval/sr_vs_base_mean_psnr`와 `eval/sr_vs_base_ssim`이 양수 유지
- `eval/lowpass_drift_l1`이 v3/v2 수준에서 크게 벗어나지 않음
- `eval/outside_mask_residual_l1`이 낮게 유지
- `eval/detail_mask_mean`이 약 `0.10` 근처
- grid에서 노이즈 patch, 평탄 피부/하늘, 압축 artifact에 residual이 생기지 않음

## 실행

Smoke:

```bash
source /home/ubuntu/venvs/cuda/bin/activate
python -u tools/train/train_detail_branch.py \
  --config configs/detail_branch_v5_noise_gate_top10_patch_gan_probe.yaml \
  --limit-steps 4 \
  --skip-initial-eval \
  --adversarial-start-step 0 \
  --disable-wandb \
  --output-dir /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v5_noise_gate_top10_patch_gan_smoke
```

Probe:

```bash
source /home/ubuntu/venvs/cuda/bin/activate
python -u tools/train/train_detail_branch.py \
  --config configs/detail_branch_v5_noise_gate_top10_patch_gan_probe.yaml
```

## 결과

v5는 사용자가 W&B sample grid에서 붕괴를 확인했고, step 3500 eval 직후
중단했다.

```text
run: /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v5_noise_gate_top10_patch_gan_probe
wandb: https://wandb.ai/jwheo/LuSIR/runs/u9sbs752
stopped: step 3500
```

주요 val100 추이:

| step | PSNR delta | mean PSNR delta | SSIM delta | wins | detail wins | lowpass drift | outside mask |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `+0.0502` | `+0.0668` | `+0.00147` | `93/100` | `80/100` | `0.000133` | `0.000000` |
| 500 | `+0.0537` | `+0.0761` | `+0.00137` | `99/100` | `95/100` | `0.000098` | `0.000000` |
| 1750 | `+0.0373` | `+0.0501` | `+0.00168` | `71/100` | `59/100` | `0.000116` | `0.000000` |
| 2250 | `+0.0218` | `+0.0215` | `+0.00164` | `51/100` | `36/100` | `0.000131` | `0.000000` |
| 3250 | `-0.0165` | `-0.0312` | `+0.00079` | `29/100` | `18/100` | `0.000133` | `0.000000` |
| 3500 | `-0.0953` | `-0.1299` | `-0.00030` | `11/100` | `5/100` | `0.000169` | `0.000000` |

판단:

- v2 noise-negative top10 gate 자체는 동작했다. `detail_mask_mean`은 약
  `0.10`이고 `outside_mask_residual_l1`은 끝까지 `0`이었다.
- 그러나 PatchGAN이 켜진 뒤 masked residual/gate가 점점 커졌고, 실제 GT-aligned
  detail correction보다 긁힌 듯한 고주파 artifact를 mask 안쪽에 넣었다.
- step 500의 best는 adversarial이 실질적으로 들어가기 전의 작은 이득이다.
  이후 장기 학습은 high-frequency ratio만 올리면서 PSNR/wins를 빠르게 잃었다.
- 따라서 v5 PatchGAN 방향은 실패로 기록한다. 같은 config continuation이나
  더 강한 adversarial weight는 하지 않는다.

다음 방향:

- texture 생성 실험에서 PatchGAN을 우선 제외한다.
- v2 gate는 유지하되, mask 안쪽 후보 patch를 고정 review set으로 더 작게 보고
  artifact/noise negative loss를 generator에도 직접 넣는 쪽을 검토한다.
- 또는 generator가 아니라 Stage 1 decoder-side/detail-capacity 병목을 먼저
  점검한다.
