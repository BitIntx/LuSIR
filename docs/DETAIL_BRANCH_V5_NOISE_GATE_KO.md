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
