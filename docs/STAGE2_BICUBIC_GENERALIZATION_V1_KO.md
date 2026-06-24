# Stage2 clean-bicubic generalization v1

deterministic512 probe는 같은 train512에서 detail metric을 올릴 수 있었지만,
held-out val100에서는 PSNR/SSIM이 하락하고 excess/highpass error가 증가했다.
V1의 목적은 fixed-subset memorization이 아니라 detail signal의 전이 여부를
직접 평가하는 것이다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_bicubic_generalization_v1.yaml
init:   checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
data:   benchmark_bicubic, full photo130k+LSDIR train split
wandb:  https://wandb.ai/jwheo/LuSIR/runs/yr815agn
```

설계:

- train은 stochastic crop, horizontal flip, texture-aware crop retry를 복원한다.
- 회전/기하 변형은 사용하지 않는다.
- 약한 HR color jitter(`[0.98, 1.02]`, probability `0.15`)만 추가한다.
- det512의 강한 highpass balance를 완화하고 원래 LR `5e-6`을 쓴다.
- GT보다 강한 local highpass energy만 soft hinge로 벌주는
  `artifact_excess_weight: 2.0`을 추가한다.
- primary eval은 held-out deterministic clean-bicubic val100이며 checkpoint도
  `eval/decoded_mean_psnr`로 선택한다.
- 동일 시점에 deterministic train512를 `eval_train512/*`로 함께 기록한다.
  train만 오르고 val이 떨어지는 순간을 W&B에서 바로 확인하기 위함이다.
- batch 8, gradient accumulation 1, 12000 micro-step/update로 실행한다.
- 중복 저장을 피하기 위해 best와 latest만 남기고 numbered step checkpoint는
  만들지 않는다.

중단/승격 기준:

- val mean PSNR과 SSIM이 best98000보다 개선되거나 최소한 유지돼야 한다.
- highpass ratio 상승만으로 성공 판정하지 않는다.
- val highpass L1 또는 excess energy가 초기값보다 지속 증가하면 실패다.
- train512와 val100 간 PSNR gap이 벌어지고 fixed grid에 ripple/grid texture가
  나타나면 조기 중단한다.
- 이 run은 Stage2 연구 probe이며 결과 확인 전 public/HF/Colab에 승격하지 않는다.

## 실행 상태

2026-06-24에 장기 run을 시작했다. step1 dual eval은 기존 best98000 기준을
재현했다.

```text
held-out val100: mean PSNR 26.92, SSIM 0.82143, highpass ratio 0.824,
                    highpass L1 0.03113, missing 0.01799, excess 0.00678
fixed train512:  mean PSNR 26.95, SSIM 0.80987, highpass ratio 0.791,
                    highpass L1 0.02935, missing 0.01743, excess 0.00539
```

L40S에서 batch 8은 약 `37.8 / 46.1 GiB`를 사용하고, eval/checkpoint 구간을
제외한 학습 속도는 약 `1.13 step/s`다.

## V1 조기 중단

V1은 첫 trained eval인 step500에서 중단했다.

| step | val mean PSNR | val SSIM | highpass ratio | highpass L1 | missing | excess |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.91998 | 0.82143 | 0.82445 | 0.03113 | 0.01799 | 0.00678 |
| 500 | 26.98610 | 0.81879 | 0.77892 | 0.03100 | 0.01935 | 0.00555 |

PSNR은 `+0.0661 dB`, highpass L1과 excess는 소폭 개선됐지만 SSIM은
`-0.00265`, highpass ratio는 `-0.04554`, missing은 `+0.00137` 악화했다.
고정 grid 차이는 작았으나 metric은 artifact 억제와 함께 실제 detail까지
줄어든 smooth bias를 명확히 보였다. 따라서 12000 step까지 계속하지 않았다.

## V2

후속 config는
`configs/latent_pretrain_photo130k_lsdir_dual_bicubic_generalization_v2.yaml`
이다. W&B는 <https://wandb.ai/jwheo/LuSIR/runs/9b0lgtbf>다. 데이터, 모델,
LR, dual eval은 V1과 같고 손실만 최소 변경한다.

- excess hinge weight: `2.0 -> 1.0`
- target-aligned missing hinge weight: `0.0 -> 2.0`
- missing hinge는 GT highpass 부호로 prediction을 projection하므로 prediction
  highpass가 0이어도 올바른 방향의 gradient가 생긴다.
- excess는 absolute local energy로 계속 측정해 GT 근거보다 강한 texture를 막는다.

2-step CUDA smoke에서 missing loss는 `0.02434`, active fraction은 `0.8841`로
실제 활성화됐고 전체 손실 기여도는 약 15%였다. V2도 step500에서 val PSNR,
SSIM, highpass ratio, missing, excess가 함께 움직이지 않으면 즉시 조정한다.

## V2 결과와 V3

V2는 step1000 이후 중단했다.

| step | val mean PSNR | val SSIM | highpass ratio | highpass L1 | missing | excess |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.91977 | 0.82145 | 0.82468 | 0.03113 | 0.01798 | 0.00679 |
| 500 | 26.91545 | 0.82836 | 0.88370 | 0.03128 | 0.01585 | 0.00860 |
| 1000 | 26.86881 | 0.82921 | 0.90093 | 0.03141 | 0.01533 | 0.00915 |

target-aligned missing hinge는 SSIM과 missing energy를 개선했지만 weight `2.0`은
과했다. highpass ratio와 excess가 계속 증가하고 PSNR도 step1000에서
`-0.0510 dB` 내려갔다.

V3 config는
`configs/latent_pretrain_photo130k_lsdir_dual_bicubic_generalization_v3.yaml`
이다. V1/V2의 step500 highpass ratio를 경계로 선형 보간해 excess weight
`1.5`, missing weight `0.8`을 사용한다. 데이터, 모델, LR, scheduler,
dual eval은 모두 동일하다. 목표는 highpass ratio를 초기 `0.825` 근처에
유지하면서 PSNR/SSIM 또는 highpass L1을 개선하는 것이다.
