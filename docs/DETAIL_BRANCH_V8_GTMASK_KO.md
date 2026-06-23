# Detail Branch v8 GT-Mask Training Probe

## 목적

RealESRGAN teacher patch-quality 진단에서 teacher가 base보다 GT-aligned detail을
안정적으로 제공하지 못한다는 것이 확인됐다. v8은 teacher를 제거하고, 학습시에만
GT detail-need mask를 사용해 branch가 정확한 위치에서 residual/highpass target을
보도록 한다.

중요한 제약:

- GT mask는 training loss와 model forward gate에만 사용한다.
- eval과 inference는 기존 learned noise-negative detail mask를 사용한다.
- 따라서 런타임에는 GT 정보가 들어가지 않는다.

## 설정

```text
config: configs/detail_branch_v8_gtmask_training_probe.yaml
init:   checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt
wandb:  https://wandb.ai/jwheo/LuSIR/runs/099kwayk
log:    /home/ubuntu/scratch/sr-diffusion/detail_branch_v8_gtmask_training_probe.log
```

Training mask:

```yaml
training_detail_mask:
  source: gt_detail_need
  top_fraction: 0.20
  top_mode: binary
  highpass_kernel: 15
  patch_kernel: 9
  score_quantile: 0.95
```

Eval/inference mask:

```yaml
detail_mask:
  checkpoint: checkpoints/detail_mask_predictor_v2_noise_negative_best1500.pt
  floor: 0.05
  top_fraction: 0.20
  top_mode: binary
```

## Smoke

4-step smoke는 통과했다.

```text
step1 train:
  mask=0.2000
  teacher_res=0
  teacher_hp=0

step4 eval:
  sr_psnr delta      +0.1010 dB
  mean_psnr delta    +0.1199 dB
  SSIM delta         +0.00373
  highpass ratio     +0.0126
  laplacian ratio    +0.0024
  lowpass drift      0.000200
  outside mask L1    0.000480
```

## 판정 기준

step250/500에서 먼저 본다:

- `eval/sr_vs_base_psnr`와 `eval/sr_vs_base_ssim`이 양수여야 한다.
- v7과 달리 `eval/sr_vs_base_highpass_ratio`와
  `eval/sr_vs_base_laplacian_ratio`도 양수를 유지해야 한다.
- grid에서 visible detail이 거의 없거나 가짜 texture/노이즈가 늘면 중단한다.
- step500까지 highpass/laplacian이 유지되면 3000 step까지 본다.

## 중간 결과

step250과 step500은 통과했다.

```text
step250:
  sr_psnr delta      +0.1000 dB
  mean_psnr delta    +0.1146 dB
  SSIM delta         +0.00439
  highpass ratio     +0.0177
  laplacian ratio    +0.0040
  lowpass drift      0.000203
  outside mask L1    0.000657

step500:
  sr_psnr delta      +0.1000 dB
  mean_psnr delta    +0.1194 dB
  SSIM delta         +0.00431
  highpass ratio     +0.0158
  laplacian ratio    +0.0024
  lowpass drift      0.000192
  outside mask L1    0.000658
```

해석:

- v7은 step500에서 highpass ratio `-0.0021`, laplacian ratio `-0.0167`로 꺾였지만,
  v8은 둘 다 양수를 유지했다.
- step500 grid에서 artifact 붕괴는 보이지 않는다.
- 다만 visible detail 변화는 아직 작다. 현 시점 판정은 "계속 볼 가치 있음"이며,
  promotion 후보는 아니다.

현재 run은 3000 step까지 계속 진행 중이다.
