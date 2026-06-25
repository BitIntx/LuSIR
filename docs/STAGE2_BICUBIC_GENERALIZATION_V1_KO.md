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
이다. W&B는 <https://wandb.ai/jwheo/LuSIR/runs/2cospx1j>다. V1/V2의
step500 highpass ratio를 경계로 선형 보간해 excess weight
`1.5`, missing weight `0.8`을 사용한다. 데이터, 모델, LR, scheduler,
dual eval은 모두 동일하다. 목표는 highpass ratio를 초기 `0.825` 근처에
유지하면서 PSNR/SSIM 또는 highpass L1을 개선하는 것이다.

### V3 step500 중간 결과

V3는 첫 trained eval에서 모든 주요 held-out guardrail을 동시에 통과했다.

| step | val mean PSNR | val SSIM | highpass ratio | highpass L1 | missing | excess |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.91987 | 0.82145 | 0.82463 | 0.03113 | 0.01798 | 0.00679 |
| 500 | 27.00503 | 0.82422 | 0.82375 | 0.03090 | 0.01774 | 0.00674 |

delta:

- mean PSNR `+0.08517 dB`
- SSIM `+0.00277`
- highpass ratio `-0.00088`로 사실상 유지
- highpass L1 `-0.000227`
- missing `-0.000239`
- excess `-0.000041`

fixed train512도 mean PSNR `26.9494 -> 26.9864`, SSIM
`0.80989 -> 0.81336`으로 개선됐다. fixed grid에서 뚜렷한 smoothing,
ripple/grid artifact, 과한 sharpening은 보이지 않았다. V3는 계속 학습하되
이 결과는 중간값이며 아직 public/HF/Colab에 승격하지 않는다.

## V3 최종 결과

V3는 12000 step을 정상 완료했다. held-out clean-bicubic val100 mean PSNR
최고점인 step11500을 선택했다.

| candidate | mean PSNR | SSIM | highpass ratio | highpass L1 | missing | excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| init best98000 | 26.91987 | 0.82145 | 0.82463 | 0.031130 | 0.017983 | 0.006785 |
| V3 best11500 | **27.05483** | **0.82665** | 0.83316 | **0.030733** | **0.017283** | 0.006992 |
| V3 final12000 | 27.05443 | **0.82689** | 0.83706 | 0.030738 | **0.017181** | 0.007084 |

best11500은 init 대비 mean PSNR `+0.13496 dB`, SSIM `+0.00520`,
highpass L1 `-0.000397`, missing `-0.000700`이다. final12000은 detail
energy가 조금 더 높지만 excess도 더 커서 best11500을 유지한다.

같은 219-image formal clean-bicubic protocol에서도 개선은 재현됐다.

| candidate | mean Y PSNR | mean Y SSIM | mean RGB PSNR | mean RGB SSIM |
| --- | ---: | ---: | ---: | ---: |
| dual best98000 | 27.84314 | 0.79742 | 26.31306 | 0.77340 |
| V3 best11500 | **27.99167** | **0.80295** | **26.47050** | **0.77969** |

그러나 real-degradation val100에서는 robustness가 명확히 퇴행했다.

| preset | dual best98000 | V3 best11500 | delta |
| --- | ---: | ---: | ---: |
| mild | 24.3583 | 24.1458 | -0.2125 |
| photo_detail_mix | 24.6197 | 24.3449 | -0.2749 |
| photo_v2 | 22.7726 | 21.8082 | -0.9644 |
| photo_v3_noise_mix | 22.4044 | 21.6237 | -0.7808 |

판정:

- V3는 clean-bicubic Stage2/base 일반화에는 성공했다.
- clean fidelity와 실제 열화 robustness는 현재 objective에서 상충한다.
- V3는 clean-bicubic 연구 checkpoint로 보존한다.
- public HF/Colab 기본 checkpoint는 교체하지 않는다.
- 다음 실험은 V3에서 시작해 clean sample을 유지하면서 mild/real degradation을
  점진적으로 섞는 짧은 robustness curriculum으로 제한한다.

산출물:

```text
metrics/formal_x4_benchmark_stage2_bicubic_generalization_v3_summary.json
metrics/formal_x4_benchmark_stage2_bicubic_generalization_v3_metrics.csv
metrics/stage2_bicubic_generalization_v3_cross_preset_summary.json
samples/stage2_bicubic_generalization_v3_contact_sheet.jpg
```

## 148M high-resolution trunk probe

V3에서 확인된 clean fidelity 개선이 모델 용량에 막힌 것인지 분리하기 위해
full-resolution residual trunk만 제한적으로 확장한다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_bicubic_trunk148m_probe.yaml
init:   V3 best11500
blocks: 16 -> 40
params: 119.238M -> 147.587M (1.238x)
steps:  4000
wandb:  https://wandb.ai/jwheo/LuSIR/runs/4y21n40o
```

추가 blocks 16-39는 두 번째 convolution을 zero-init한다. 기존 16개 block과
context branch를 partial load하면 확장 모델의 첫 출력은 V3 best11500과
bit-exact하게 같다. 따라서 초기 metric 변화 없이 추가 용량 자체의 학습 효과를
볼 수 있다.

L40S smoke 결과:

- batch 8: decoder backward OOM
- batch 7: 첫 update는 통과하지만 5개 eval 이후 다음 backward OOM
- batch 6: eval 전후 3 update 통과
- batch 6 VRAM: 약 `39.2 / 46.1GB`
- 학습 구간 GPU utilization: `98~100%`
- 전체 test suite: `94 passed`

장기 run은 2026-06-24 시작했다. 초기 5개 val100 eval 이후 학습 안정 구간은
약 `1.11 step/s`, VRAM `40.4/46.1GB`, GPU utilization `100%`, 약
`318W`다.

판정 기준:

- clean-bicubic mean PSNR이 2000-4000 step 내 최소 `+0.05 dB` 개선돼야 한다.
- SSIM/highpass L1과 fixed grid가 함께 유지돼야 한다.
- 네 real-degradation preset 중 하나라도 지속적으로 `-0.10 dB` 이상
  후퇴하면 public 후보로 보지 않는다.
- clean 개선이 `+0.02 dB` 미만에서 정체하면 parameter 부족이 주 병목이
  아니라고 판정하고 조기 중단한다.

### 148M probe 최종 결과

4000 step을 정상 완료했고 clean val100 mean PSNR 최고점인 step3500을
선택했다.

| candidate | clean PSNR | clean SSIM | mild delta | detail-mix delta | photo_v2 delta | v3-noise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| init V3 best11500 | 27.05473 | 0.82669 | 0 | 0 | 0 | 0 |
| 148M best3500 | 27.05837 | 0.82719 | -0.02011 | -0.01005 | -0.05466 | -0.06268 |

clean val100 이득은 `+0.00364 dB`로 목표 `+0.05 dB`의 약 7%에 불과하다.
highpass ratio는 `0.83366 -> 0.83905`, missing은
`0.017266 -> 0.017102`로 움직였지만 excess도
`0.007007 -> 0.007144`로 증가했다.

정식 219-image 결과:

| candidate | mean Y PSNR | mean Y SSIM | mean RGB PSNR | mean RGB SSIM |
| --- | ---: | ---: | ---: | ---: |
| V3 best11500 | 27.99167 | 0.802953 | 26.47050 | 0.779686 |
| 148M best3500 | **27.99716** | **0.803327** | **26.47654** | **0.780054** |

평균 delta는 Y PSNR `+0.00549 dB`, Y SSIM `+0.000374`이며 Y-PSNR
승률은 `114/219`다. DIV2K는 `+0.00189 dB`, wins `44/100`으로 사실상
동률이다. contact sheet와 fixed grid에서도 차이를 구분하기 어렵다.

판정:

- `119.24M` Stage2의 주 병목은 parameter 수가 아니다.
- 28.35M parameter와 더 높은 VRAM/추론 비용을 정당화할 이득이 없다.
- 148M checkpoint를 HF/Colab/public 기본값으로 승격하지 않는다.
- 다음 실험은 V3의 clean fidelity를 보존하면서 mild/strong 열화를 섞는
  robustness curriculum이어야 한다.

```text
metrics/stage2_bicubic_trunk148m_probe_summary.json
metrics/formal_x4_benchmark_stage2_trunk148m_summary.json
metrics/formal_x4_benchmark_stage2_trunk148m_metrics.csv
samples/stage2_trunk148m_contact_sheet.jpg
```

## Mixed-degradation robustness bridge v1

148M 확장에서 실질 이득이 없었으므로 V3 best11500의 `119.238M` 구조로
돌아간다. 목표는 clean-bicubic fidelity를 유지한 채 V3에서 잃은 실제 열화
robustness를 일부 회복하는 것이다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_robustness_bridge_v1.yaml
init:   V3 best11500
steps:  6000
batch:  8, grad accumulation 1
LR:     2e-6 warmup cosine
wandb:  https://wandb.ai/jwheo/LuSIR/runs/7fidh724
```

train degradation mix:

| preset | weight |
| --- | ---: |
| benchmark_bicubic | 55% |
| photo_detail | 20% |
| mild | 15% |
| photo_v2 | 8% |
| photo_v3_noise | 2% |

checkpoint selection은 clean 단일 PSNR이 아니라 clean `45%`, mild와
photo-detail-mix 각 `15%`, photo-v2와 photo-v3-noise-mix 각 `12.5%`의
weighted mean PSNR을 사용한다. 단 다음 guardrail을 모두 통과해야 한다.

- clean mean PSNR `>= 27.02`
- clean SSIM `>= 0.8260`
- clean excess energy `<= 0.0075`

trainer는 이제 primary eval에도 `eval.data` override를 지원한다. 따라서 학습
preset이 mixed여도 primary val100은 항상 `benchmark_bicubic`이고, 네 실제
열화 평가는 별도 namespace에 기록된다. CUDA 3-step smoke에서 초기 clean
mean PSNR `27.0550`, composite score `25.7430`, guardrail valid를 확인했고
전체 test suite는 `97 passed`다.

장기 run은 2026-06-24 시작했다. 초기 composite score는 `25.74295`,
selection valid는 `1.0`이며 안정 학습 구간은 약 `1.13 step/s`, VRAM
`37.3/46.1GB`, GPU utilization `99~100%`, 약 `318W`다.

### Bridge v1 결과와 v2

v1은 6000 step을 완료했다. clean guardrail을 통과한 trained checkpoint는
없었지만 실제 열화 학습 방향 자체는 유효했다.

| step | clean delta | mild delta | detail-mix delta | photo_v2 delta | v3-noise delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | -0.03873 | +0.27055 | +0.26293 | +0.77224 | +0.72275 |
| 4500 | -0.06485 | +0.29902 | +0.30680 | +0.81660 | +0.77431 |

step500은 clean threshold `27.02`보다 `0.00372 dB` 낮았다. 이후 strong
성능은 비슷한 수준에서 정체되고 clean만 더 내려갔다. 자동 best step1은 V3
초기값 복제본이고 latest6000은 더 나쁜 균형이므로 두 checkpoint를 삭제했다.

v2는 다음처럼 보수적으로 조정한다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_robustness_bridge_v2.yaml
clean/degraded: 70% / 30%
mix: benchmark 70, photo-detail 15, mild 10, photo-v2 4, photo-v3 1
LR: 1e-6
steps: 1500
eval: every 250 steps
```

selection score와 guardrail은 v1과 같아 직접 비교할 수 있다. smoke 초기값은
composite `25.74284`, clean `27.05493`, SSIM `0.82665`, excess
`0.006994`, selection valid `1.0`이다. 전체 test suite는 `98 passed`다.
