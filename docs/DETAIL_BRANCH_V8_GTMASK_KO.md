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
- step500까지 highpass/laplacian이 유지되면 계속 보되, 후반 detail score가
  정체되거나 laplacian이 음수로 꺾이면 중단한다.

## 결과

run은 step2000 eval 이후 중단했다. step2000에서 PSNR은 step500보다 아주
조금 높았지만, laplacian ratio delta가 음수로 꺾였고 best detail score는
계속 step500에 머물렀다. 더 오래 돌리는 것은 같은 작은 correction을 흔드는
쪽으로 보였다.

```text
step0:
  sr_psnr delta      +0.0967 dB
  mean_psnr delta    +0.1133 dB
  SSIM delta         +0.00364
  highpass ratio     +0.0129
  laplacian ratio    +0.0049
  lowpass drift      0.000211
  outside mask L1    0.000497
  wins               95/100

step250:
  sr_psnr delta      +0.1000 dB
  mean_psnr delta    +0.1146 dB
  SSIM delta         +0.00439
  highpass ratio     +0.0177
  laplacian ratio    +0.0040
  lowpass drift      0.000203
  outside mask L1    0.000657
  wins               91/100

step500:
  sr_psnr delta      +0.1000 dB
  mean_psnr delta    +0.1194 dB
  SSIM delta         +0.00431
  highpass ratio     +0.0158
  laplacian ratio    +0.0024
  lowpass drift      0.000192
  outside mask L1    0.000658
  wins               92/100

step1500:
  sr_psnr delta      +0.0981 dB
  mean_psnr delta    +0.1129 dB
  SSIM delta         +0.00431
  highpass ratio     +0.0167
  laplacian ratio    +0.0048
  lowpass drift      0.000191
  outside mask L1    0.000711
  wins               90/100

step1750:
  sr_psnr delta      +0.1000 dB
  mean_psnr delta    +0.1181 dB
  SSIM delta         +0.00429
  highpass ratio     +0.0151
  laplacian ratio    +0.0019
  lowpass drift      0.000185
  outside mask L1    0.000706
  wins               92/100

step2000:
  sr_psnr delta      +0.1008 dB
  mean_psnr delta    +0.1191 dB
  SSIM delta         +0.00413
  highpass ratio     +0.0138
  laplacian ratio    -0.0006
  lowpass drift      0.000180
  outside mask L1    0.000702
  wins               95/100
```

Best checkpoint:

```text
checkpoint: /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v8_gtmask_training_probe/checkpoints/best_eval_detail.pt
best step:  500
latest:     step 2000
```

해석:

- v7은 step500에서 highpass ratio `-0.0021`, laplacian ratio `-0.0167`로 꺾였지만,
  v8은 적어도 step1750까지 둘 다 대체로 양수를 유지했다.
- step500/1000 grid에서 artifact 붕괴는 보이지 않는다.
- 다만 visible detail 변화는 여전히 작고, step2000에서는 laplacian ratio가
  음수로 꺾였다.
- GT-mask training은 RealESRGAN teacher를 제거하는 판단을 지지하지만,
  masked detail branch 구조 자체의 texture 생성 한계를 풀지는 못했다.

판정:

- v8은 보존할 diagnostic result지만 public/default artifact로 승격하지 않는다.
- 같은 config를 3000+ step으로 길게 이어 돌리지 않는다.
- 다음 실험은 teacher/detail branch continuation이 아니라 Stage2/base
  conditional-latent smoothing을 줄이는 방향으로 잡는다.
