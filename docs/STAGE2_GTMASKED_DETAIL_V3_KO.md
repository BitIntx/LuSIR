# Stage2 GT-Masked Detail v3 Probe

## 목적

detail branch v6/v7/v8은 frozen Stage2 base 위에서 안전한 작은 correction은
만들었지만, visible texture breakthrough는 만들지 못했다. Stage1 audit에서는
decoder가 HR latent를 받으면 detail을 보존할 수 있음이 확인됐으므로, 다음 병목은
Stage2 LR-to-latent predictor의 conditional-mean smoothing으로 본다.

이 probe는 Stage2 자체가 GT 대비 missing-detail 위치를 더 강하게 보도록
train-only detail mask loss를 추가한다. inference에는 GT mask가 들어가지 않는다.

## 설정

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_detail_gtmasked_v3_probe.yaml
init:   checkpoints/stage2_photo130k_lsdir_dual_detail_guarded_v2_best10000.pt
run:    /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_detail_gtmasked_v3_probe
log:    /home/ubuntu/scratch/sr-diffusion/latent_pretrain_photo130k_lsdir_dual_detail_gtmasked_v3_probe.log
wandb:  https://wandb.ai/jwheo/LuSIR/runs/ch8ma1sk
```

새 loss:

```yaml
loss:
  detail_weighted:
    source: prediction_missing
    decoded_weight: 0.35
    highpass_weight: 1.25
    top_fraction: 0.20
    top_mode: binary
    mask_floor: 0.05
    highpass_kernel: 15
    patch_kernel: 9
    score_quantile: 0.95
    laplacian_kernel: 3
```

`source: prediction_missing`은 현재 decoded prediction과 GT를 비교해 missing-detail
score를 만들고, 상위 20% 영역에 추가 decoded/highpass loss를 건다. mask 계산은
`decoded.detach()`로 수행해 mask 선택 자체로 gradient를 흘리지 않는다.

## Smoke

4-step smoke는 통과했다.

```text
step1:
  loss              0.50075
  detail_decoded    0.15427
  detail_highpass   0.11050
  detail_mask       0.2400

eval step1:
  decoded_psnr      24.63
  mean_psnr         26.50
  detail_ratio      0.328
  highpass_ratio    0.808
  missing           0.01897
  psnr_detail_score 25.286
```

## 결과

step1000 eval 이후 중단했다. train-only detail loss는 정상적으로 non-zero였고
mask도 기대값을 유지했지만, eval 방향은 목표와 반대였다.

```text
baseline guarded v2 best10000:
  mean_psnr         26.5050
  highpass_ratio    0.8084
  missing           0.01897

step500:
  decoded_psnr      24.64
  mean_psnr         26.52
  detail_ratio      0.301
  highpass_ratio    0.795
  missing           0.01933
  psnr_detail_score 25.238

step1000:
  decoded_psnr      24.64
  mean_psnr         26.53
  detail_ratio      0.302
  highpass_ratio    0.794
  missing           0.01939
  psnr_detail_score 25.243
```

해석:

- mean PSNR은 소폭 올랐지만, highpass ratio와 missing-detail metric은 악화했다.
- 즉 이 loss는 Stage2 smoothing을 줄이지 못했고, 오히려 더 보수적인 평균 복원
  방향으로 움직였다.
- `prediction_missing` top20 mask 경로는 구현/로깅 검증에는 유용하지만, 이
  weight/config를 더 오래 continuation하지 않는다.
- 다음 Stage2 시도는 mask-weighted loss만 더하는 방식이 아니라 target 자체나
  architecture/output parameterization을 바꿔야 한다.

## 중단 기준

- `eval/decoded_mean_psnr`가 guarded v2 best10000 기준 `26.5050` 근처를 유지해야 한다.
- `eval/highpass_energy_ratio`가 `0.8084`보다 올라가도, `eval/missing_energy`와
  sample grid가 나빠지면 artifact성 detail로 본다.
- `train/detail_mask_mean`은 floor 포함 약 `0.24` 근처가 정상이다.
- `train/detail_decoded`, `train/detail_highpass`가 계속 0이면 mask/loss 경로가
  막힌 것이다.
- step500/1000에서 `mean_psnr_detail_score`가 개선되지 않거나 grid가 더 거칠어지면
  조기 중단한다.
