# 시행착오 리포트

이 문서는 직접 학습 x4 latent diffusion SR 실험에서 실패/부분 성공/다음 가설을
계속 누적하기 위한 기록이다. 최종 성능표가 아니라, 왜 다음 실험을 그렇게 잡았는지
추적하는 용도다.

## 2026-06-04 VM 복구 후 상태

- GitHub HEAD: `900d1cd Fix report table layout`
- GPU: 1x NVIDIA L40S 46GB
- PyTorch: `2.12.0+cu130`
- 데이터: photo100k train `103450`, val `100`
- Stage2 XL condition encoder:
  `/home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/step_0072000.pt`
- Stage4 XL edge-loss checkpoint:
  `checkpoints/stage4_photo100k_xl_edge_b16_best_eval_condition_decoded.pt`
- W&B API/HF/GitHub auth 확인됨.

## 관찰 1: Stage4 XL edge-loss는 주로 cleanup 역할

기존 최신 Stage4 XL edge-loss run:

- config: `configs/diffusion_photo100k_xl_stage4_condition_v3_edge_b16.yaml`
- checkpoint step: `4250`
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/nog04fwr>

`photo_v3_noise_mix` val100 sampled eval:

| 모델 | start timestep | SR PSNR | bicubic PSNR | bicubic 대비 | condition 대비 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage2 condition-only | n/a | 22.9014 | 22.3599 | +0.5415 | n/a |
| Stage4 edge | 25 | 22.9563 | 22.3599 | +0.5964 | +0.0549 |
| Stage4 edge | 50 | 23.0799 | 22.3599 | +0.7200 | +0.1784 |

샘플별 관찰:

- t50은 condition이 크게 깨진 샘플에서 artifact/noise suppression으로 이득이 큼.
- condition이 이미 좋은 fine texture/skin/building/snow 샘플에서는 diffusion이 자주 손해를 냄.
- t50은 condition보다 좋은 샘플이 `45/100`, t25는 `42/100`.
- W&B `samples/PredX0`는 full sampled output이 아니라 one-step x0 proxy라 실제 SR 판단에는 부족함.

결론:

- Stage4 edge-loss는 평균 PSNR을 올리지만, 역할이 "missing HR detail generation"보다는
  "condition output cleanup/restoration"에 가까움.
- t50은 v3 noise 계열 cleanup에는 도움이 되지만 over-editing 위험이 큼.

## 관찰 2: Stage2 condition-only는 degradation별로 이미 강한 base

Stage2 condition encoder를 직접 decode해서 같은 val100/seed에서 평가했다.
평가 스크립트:

```bash
python eval_condition_samples.py \
  --config configs/diffusion_photo100k_xl_stage4_condition_v3_resdetail_photo_v2_b8.yaml \
  --output-dir /home/ubuntu/scratch/sr-diffusion/runs/eval_stage2_xl_condition_only_${preset}_val100 \
  --degradation-preset ${preset} \
  --split val \
  --limit 100 \
  --batch-size 8 \
  --seed 1337 \
  --grid-count 8
```

결과:

| degradation | bicubic PSNR | condition PSNR | delta |
| --- | ---: | ---: | ---: |
| `mild` | 24.4778 | 25.0449 | +0.5672 |
| `photo_v2` | 22.4103 | 22.9271 | +0.5167 |
| `photo_v3_noise_mix` | 22.3599 | 22.9014 | +0.5415 |

결론:

- Stage2는 단순히 약한 baseline이 아니라 이미 꽤 강한 base reconstruction이다.
- 따라서 Stage4가 전체 x0/image를 다시 맞추는 방식이면 condition을 쉽게 망칠 수 있다.
- Stage1 VAE는 현재 주범으로 보이지 않아 건드리지 않는다.
- Stage3 noise-start 방향으로 돌아가는 것도 우선순위가 낮다.

## 실험 1: residual-detail photo_v2 Stage4 probe

목표:

- Stage2를 고정하고 Stage4 objective만 바꿔서 "condition 대비 residual detail"을 학습하는지 확인.
- 기존 edge-loss continuation이 아니라 새 Stage4 U-Net을 처음부터 학습.

추가된 config:

- `configs/diffusion_photo100k_xl_stage4_condition_v3_resdetail_photo_v2_b8.yaml`

핵심 설정:

- degradation: `photo_v2`
- batch: `8`, grad accumulation: `4`
- micro steps: `20000`
- 실제 optimizer update: `5000`
- start/sample timestep: `50`, 추가 sampled eval은 `25`도 확인
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/xyvqg0n6>

추가한 loss:

- `sobel_residual_magnitude_loss`
- `laplacian_residual_magnitude_loss`

학습 상태:

- 정상 종료: `finished step=20000`
- best one-step decoded eval: step `19000`
- one-step decoded PSNR: `21.06 -> 21.81`
- GPU 병목 없음: L40S에서 약 `0.87 micro-step/s`, VRAM 약 `45.2/46.1GB`

sampled val100 결과, 같은 `photo_v2` 기준:

| 모델 | start timestep | SR/condition PSNR | bicubic PSNR | bicubic 대비 | condition 대비 | condition 이긴 샘플 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2 condition-only | n/a | 22.9271 | 22.4103 | +0.5167 | n/a | n/a |
| Stage4 residual-detail best | 25 | 22.8492 | 22.4103 | +0.4389 | -0.0779 | 28/100 |
| Stage4 residual-detail latest | 25 | 22.8388 | 22.4103 | +0.4285 | -0.0882 | 27/100 |
| Stage4 residual-detail best | 50 | 22.6339 | 22.4103 | +0.2236 | -0.2932 | 24/100 |

시각 관찰:

- t25는 condition보다 약간 더 선명하거나 거칠게 보이는 샘플이 있음.
- 그러나 GT detail 복원이라기보다 grain/contrast를 얹는 경우가 많음.
- t50은 과하게 건드려 노이즈성 texture, 색 얼룩, 거친 표면이 늘어남.
- fine texture/skin/snow/building류에서는 condition을 손상하는 경우가 많음.

결론:

- 이 run은 최종 성능 관점에서는 실패/부분 실패.
- 하지만 "highpass/residual magnitude만 추가하면 Stage4가 detail refiner가 된다"는 가설을 반박했다.
- Stage4에는 condition의 구조/저주파를 보존하는 제약이 필요하다.

## 실험 2: role-split lowpass-anchor mild probe

목표:

- Stage2를 base reconstruction으로 고정.
- Stage4가 condition을 덮어쓰지 못하게 저주파를 condition에 anchor.
- GT 대비 필요한 detail이 적은 위치에서는 fake highpass를 추가하지 못하게 gate.
- t50 over-editing을 피하고 t25 중심의 작은 refiner로 제한.

추가된 config:

- `configs/diffusion_photo100k_xl_stage4_condition_v3_rolesplit_mild_b8_probe.yaml`

추가한 loss:

- `lowpass_anchor_loss`
- `laplacian_detail_gate_anchor_loss`

핵심 설정:

- degradation: `mild`
- train timestep range: `5..75`
- sample/eval timestep: `25`
- batch: `8`, grad accumulation: `4`
- micro steps: `8000`
- 실제 optimizer update: `2000`
- save every: `2000` micro steps

학습 결과:

- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/lrb6nco9>
- 정상 종료: `finished step=8000`
- micro steps: `8000`
- 실제 optimizer update: `2000`
- best one-step decoded eval: step `7500`
- one-step decoded PSNR: `23.1841 -> 23.2515`
- GPU 병목 없음: L40S에서 약 `0.86-0.87 micro-step/s`, VRAM 약 `45.2/46.1GB`

초기 로그 예:

```text
step=1 loss=0.19193 noise_mse=35.00126 x0_mse=0.37928 decoded=0.08468
edge=0.04472 highpass=0.05081 res_edge_mag=0.04556 res_high_mag=0.03586
low_anchor=0.00533 detail_gate=0.02187 steps_per_sec=0.39
eval step=1 noise_mse=42.67671 decoded_psnr=23.18
```

sampled val100 결과, 같은 `mild` 기준:

| 모델 | start timestep | SR/condition PSNR | bicubic PSNR | bicubic 대비 | condition 대비 | condition 이긴 샘플 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2 condition-only | n/a | 25.0449 | 24.4778 | +0.5672 | n/a | n/a |
| Stage4 role-split best | 25 | 24.5747 | 24.4778 | +0.0969 | -0.4702 | 3/100 |
| Stage4 role-split best | 10 | 24.9185 | 24.4778 | +0.4408 | -0.1264 | 3/100 |
| Stage4 role-split best | 5 | 24.9935 | 24.4778 | +0.5158 | -0.0514 | 6/100 |
| Stage4 role-split best | 1 | 25.0335 | 24.4778 | +0.5557 | -0.0114 | 10/100 |

시각 관찰:

- t25는 아직 condition을 꽤 손상한다. 특히 fine texture/edge가 좋은 샘플에서 blur나
  fake texture가 섞인다.
- t10/t5로 낮추면 손상이 줄어들지만, 새 GT detail을 안정적으로 추가하는 느낌은 약하다.
- t1은 눈으로도 거의 Stage2 condition-only와 같다. 평균 PSNR도 condition-only에
  거의 붙지만, condition을 이긴 샘플은 `10/100`뿐이다.

결론:

- role-split loss는 "덜 망가뜨리는 방향"으로는 효과가 있다.
- 그러나 full x0/image를 예측하는 Stage4 구조에서는 diffusion을 충분히 태울수록
  condition을 덮어쓰는 문제가 남는다.
- t1에서만 condition과 비슷하다는 것은 diffusion이 유용한 SR detail을 추가했다기보다
  거의 condition을 통과시킨다는 뜻에 가깝다.
- 따라서 추가 loss weight 튜닝만으로 해결될 가능성은 낮아졌다.

## 현재 판단

- Stage1 VAE는 건드리지 않는다.
- Stage3로 되돌아가지 않는다.
- Stage2는 강한 base 역할을 이미 하고 있으므로 즉시 재학습하지 않는다.
- Stage4 loss만 바꾸는 실험은 두 번 모두 condition-only를 넘지 못했다.
  - residual-detail photo_v2: highpass/detail을 넣으면 거칠어지고 condition을 손상.
  - role-split mild: 보존은 좋아졌지만 SR detail 추가는 거의 없음.
- 다음 우선순위는 Stage4 architecture/parameterization 변경이다.
  예: U-Net이 full x0를 직접 예측하지 않고, condition 위의 bounded residual 또는
  gated residual만 예측하게 만들기.

## 실험 3: gated residual x0 parameterization

목표:

- Stage4 U-Net이 full x0/noise-to-x0를 마음대로 예측하지 못하게 제한.
- U-Net output을 noise가 아니라 `condition + bounded residual * learned gate`로 해석.
- DDIM sampler에는 이 x0에서 역산한 noise를 사용해 학습/평가/샘플링 의미를 일치.
- mild val100에서 Stage2 condition-only `25.0449 dB`를 넘는지 확인.

추가된 config:

- `configs/diffusion_photo100k_xl_stage4_condition_v3_gated_residual_mild_b8_probe.yaml`

핵심 설정:

- degradation: `mild`
- prediction type: `gated_residual_x0`
- model output channels: `32`
  - first 16 channels: residual logits
  - next 16 channels: gate logits
- latent residual bound: `1.25`
  - val100 mild 기준 `abs(target_latent - condition_latent)` 분포:
    - p95 `0.695`
    - p99 `1.25`
    - p99.5 `1.617`
- batch: `8`, grad accumulation: `4`
- micro steps: `8000`
- 실제 optimizer update: `2000`
- train timestep range: `1..75`
- sample/eval timestep: `25`

초기화:

- role-split mild best checkpoint에서 partial init.
- output head shape만 달라서 2개 tensor는 새로 초기화.
- CUDA smoke 결과:
  - matched params: `469599616/469636512`
  - batch 8 forward/backward 성공
  - max allocated: 약 `39.9GB`

학습 결과:

- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/edfko8e8>
- step `2000`에서 중단.
  - 원래 config는 `8000` micro steps였지만 one-step decoded proxy가 step `500` 이후
    보합이라 step `2000` checkpoint를 확보한 뒤 sampled eval을 먼저 보기로 했다.
- best one-step decoded eval: step `1000`
- one-step decoded PSNR:
  - step 1: `22.86`
  - step 500: `23.47`
  - step 1000: `23.48`
  - step 1500: `23.47`
  - step 2000: `23.47`
- GPU 병목 없음:
  - L40S VRAM 약 `44.7/46.1GB`
  - train util `96-100%`
  - steady speed 약 `0.79 micro-step/s`
  - thermal slowdown 없음, SW power cap만 active

sampled val100 결과, 같은 `mild` 기준:

| 모델 | checkpoint | start timestep | SR/condition PSNR | bicubic PSNR | bicubic 대비 | condition 대비 | condition 이긴 샘플 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage2 condition-only | n/a | n/a | 25.0449 | 24.4778 | +0.5672 | n/a | n/a |
| Stage4 role-split best | 25 | 25 | 24.5747 | 24.4778 | +0.0969 | -0.4702 | 3/100 |
| Stage4 role-split best | 5 | 5 | 24.9935 | 24.4778 | +0.5158 | -0.0514 | 6/100 |
| Stage4 role-split best | 1 | 1 | 25.0335 | 24.4778 | +0.5557 | -0.0114 | 10/100 |
| Stage4 gated residual | 1000 | 25 | 25.0415 | 24.4778 | +0.5637 | -0.0035 | 25/100 |
| Stage4 gated residual | 1000 | 10 | 25.0415 | 24.4778 | +0.5637 | -0.0034 | 25/100 |
| Stage4 gated residual | 1000 | 5 | 25.0416 | 24.4778 | +0.5638 | -0.0034 | 25/100 |
| Stage4 gated residual | 1000 | 1 | 25.0418 | 24.4778 | +0.5640 | -0.0032 | 25/100 |
| Stage4 gated residual | 2000 | 25 | 25.0445 | 24.4778 | +0.5667 | -0.0004 | 34/100 |
| Stage4 gated residual | 2000 | 10 | 25.0444 | 24.4778 | +0.5666 | -0.0006 | 32/100 |
| Stage4 gated residual | 2000 | 5 | 25.0444 | 24.4778 | +0.5667 | -0.0005 | 31/100 |
| Stage4 gated residual | 2000 | 1 | 25.0443 | 24.4778 | +0.5665 | -0.0007 | 32/100 |

시각 관찰:

- role-split t25에서 보였던 over-editing/condition 손상은 크게 줄었다.
- t25/t10/t5/t1 결과가 거의 같아서 sampler가 강하게 새 detail을 만들기보다
  condition 주변의 작은 residual만 적용하는 상태로 보인다.
- grid는 Stage2 condition-only와 매우 비슷하다.

결론:

- gated residual parameterization은 성공한 부분이 있다.
  - full x0 덮어쓰기 문제를 크게 줄였다.
  - t25에서도 condition-only와 거의 동률까지 보존한다.
  - condition을 이긴 샘플 수가 role-split t1 `10/100`에서 gated step2000 t25 `34/100`으로 늘었다.
- 하지만 목표 기준에서는 아직 실패/부분 성공이다.
  - 평균 PSNR은 condition-only를 넘지 못했다.
  - 새 GT detail을 안정적으로 추가했다기보다 condition output을 거의 보존하는 쪽이다.
- 다음 방향은 단순히 더 오래 학습하는 것이 아니라, residual이 어디서/얼마나 필요한지
  더 직접적으로 알려주는 신호가 필요하다.
  예:
  - residual/gate supervised loss를 latent 또는 decoded domain에 명시적으로 추가.
  - Stage2가 uncertainty/detail-need map을 같이 예측하게 해서 Stage4 gate 조건으로 사용.
  - residual branch를 diffusion 전체가 아니라 deterministic residual refiner로 먼저 검증.

## 실험 4: Stage2 residual/oracle diagnostic

목표:

- Stage2 condition-only가 실제로 무엇을 놓치는지 분해한다.
- 저주파/구조가 문제인지, 고주파/detail residual이 문제인지 확인한다.
- Stage4 diffusion을 더 돌리기 전에 residual refiner가 풀어야 할 target을 분명히 한다.

추가된 스크립트:

- `diagnose_stage2_residuals.py`

실행:

```bash
python diagnose_stage2_residuals.py \
  --config configs/diffusion_photo100k_xl_stage4_condition_v3_gated_residual_mild_b8_probe.yaml \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/diagnose_stage2_xl_residuals_mild_val100 \
  --split val \
  --limit 100 \
  --batch-size 8 \
  --num-workers 4 \
  --sample-count 8
```

주요 결과, `mild` val100:

| 항목 | 값 |
| --- | ---: |
| bicubic PSNR | 24.4778 |
| condition decoded PSNR | 25.0543 |
| oracle full residual decoded PSNR | 41.8207 |
| oracle full vs condition | +16.7664 |
| oracle highpass decoded PSNR | 35.0872 |
| oracle highpass vs condition | +10.0329 |
| oracle lowpass decoded PSNR | 25.0814 |
| oracle lowpass vs condition | +0.0270 |
| residual highpass energy ratio | 0.8988 |
| residual lowpass energy ratio | 0.0758 |
| `abs(residual_gt) > 1.25` fraction | 0.0098 |

시각 관찰:

- Stage2 condition은 구조/색/저주파는 이미 꽤 잘 맞춘다.
- GT와의 차이는 대부분 branch, fur, water, building edge 같은 고주파 detail이다.
- highpass oracle은 texture/detail을 크게 회복하지만, lowpass oracle은 거의 차이가 없다.

결론:

- Stage1/VAE나 Stage2 전체를 처음부터 의심할 상황은 아니다.
- Stage4가 full x0를 다시 그리는 방식은 target과 맞지 않는다.
- 다음 실험은 "condition 위에 필요한 고주파 residual을 제한적으로 더하는가"만 먼저
  deterministic하게 검증하는 것이 맞다.

## 실험 5: deterministic bounded residual refiner probe

목표:

- diffusion sampler를 빼고, frozen Stage1 VAE + frozen Stage2 condition encoder 위에서
  작은 residual refiner가 condition-only를 넘을 수 있는지 확인한다.
- gated residual Stage4에서 보였던 near-identity 문제를 direct residual/gate supervision으로
  풀 수 있는지 본다.
- 성공하면 이후 Stage4 diffusion residual path의 teacher/warm-start 후보로 쓴다.

추가된 스크립트/config:

- `train_residual_refiner.py`
- `configs/residual_refiner_stage2_xl_mild_probe.yaml`
- `configs/residual_refiner_stage2_xl_mild_open_gate_probe.yaml`

구조:

```text
input: condition latent + normalized LR
output: condition + residual_scale * tanh(residual_logits) * sigmoid(gate_logits + gate_bias)
loss: latent L1 + residual L1 + highpass L1 + gate L1
```

Sparse-gate probe:

- run dir:
  `/home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_probe`
- best checkpoint:
  `checkpoints/best_eval_refined.pt`
- best step: `500`
- step `1000`까지 확인 후 step `500`이 가장 좋아서 중단.

| step | global PSNR delta | mean PSNR delta | wins vs condition | gate mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.0000 | +0.0000 | 0/100 | 0.5000 |
| 250 | +0.0333 | n/a | 77/100 | 0.3612 |
| 500 | +0.0455 | +0.0729 | 86/100 | 0.2147 |
| 750 | +0.0312 | n/a | 76/100 | 0.1685 |
| 1000 | +0.0364 | n/a | 76/100 | 0.1488 |

Best sparse-gate eval:

```text
condition_mean_psnr:              25.0449
refined_mean_psnr:                25.1178
refined_vs_condition_mean_psnr:   +0.0729
wins_vs_condition:                86/100
global_condition_decoded_psnr:    23.4794
global_refined_decoded_psnr:      23.5249
global_delta:                     +0.0455
gate_mean:                        0.2147
```

Open-gate ablation:

- run dir:
  `/home/jwheojjang/scratch/sr-diffusion/runs/residual_refiner_stage2_xl_mild_open_gate_probe`
- `gate_bias: 2.0`, `gate_l1_weight: 0`, `highpass_weight: 2`
- step `500`에서 sparse-gate보다 나빠서 중단.

```text
condition_mean_psnr:              25.0449
refined_mean_psnr:                25.0972
refined_vs_condition_mean_psnr:   +0.0523
wins_vs_condition:                73/100
global_delta:                     +0.0337
gate_mean:                        0.8680
```

시각 관찰:

- sparse-gate refined output은 condition과 매우 가깝고, 작은 detail/edge 쪽만 보정한다.
- 큰 artifact나 과한 fake texture는 보이지 않는다.
- open-gate는 gate가 크게 열리지만 평균/승률 모두 sparse-gate보다 낮다.

결론:

- residual detail은 학습 가능하다. `+0.0729 dB`, `86/100` wins는 작은 probe치고
  의미 있는 진전이다.
- 그러나 "gate를 더 열고 residual을 더 많이 더하면 된다"는 가설은 약해졌다.
- 다음 Stage4는 decoded weight를 무작정 키우는 continuation보다, deterministic residual
  refiner를 teacher/warm-start로 쓰거나 Stage4 U-Net에 residual/gate target을 직접 주는
  방향이 더 타당하다.

HF 보존:

```text
checkpoints/residual_refiner_stage2_xl_mild_best_eval_refined.pt
metrics/diagnose_stage2_xl_residuals_mild_val100_summary.json
metrics/residual_refiner_stage2_xl_mild_probe_early_stop_summary.json
metrics/residual_refiner_stage2_xl_mild_open_gate_probe_early_stop_summary.json
samples/diagnose_stage2_xl_residuals_mild_val100_grid.png
samples/residual_refiner_stage2_xl_mild_probe_step500_grid.png
samples/residual_refiner_stage2_xl_mild_open_gate_probe_step500_grid.png
```

## 실험 6: residual refiner inference/eval 연결 및 cross-degradation 확인

목표:

- residual refiner가 학습 스크립트 안에서만 쓰이는 상태를 벗어나 실제 inference/eval
  도구로 연결한다.
- `mild`에서 얻은 작은 이득이 `photo_v2`, `photo_v3_noise_mix`에서도 유지되는지 확인한다.
- Stage4 XL edge와 단일 샘플에서 체감 차이를 비교한다.

추가된 스크립트:

- `eval_residual_refiner.py`
- `infer_residual_refiner.py`

실행:

```bash
python eval_residual_refiner.py \
  --degradation-preset photo_v3_noise_mix \
  --output-dir /home/jwheojjang/scratch/sr-diffusion/runs/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100 \
  --limit 100 \
  --batch-size 8 \
  --num-workers 4 \
  --sample-count 8
```

같은 frozen sparse-gate checkpoint step `500`으로 val100 평가:

| degradation | bicubic PSNR | condition PSNR | refined PSNR | refined-condition | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mild` | 24.4778 | 25.0449 | 25.1178 | +0.0729 | 86/100 |
| `photo_v2` | 22.4103 | 22.9271 | 22.9767 | +0.0496 | 77/100 |
| `photo_v3_noise_mix` | 22.3599 | 22.9014 | 22.9600 | +0.0586 | 86/100 |

시각 관찰:

- 세 preset 모두에서 refined는 condition과 매우 가깝다.
- 큰 색 변형, fake texture, over-editing은 보이지 않는다.
- 다만 눈으로 보이는 detail 회복도 작다. PSNR/win-count로는 안정적 이득이 있지만,
  사용자가 기대하는 "업스케일 detail 생성"이라고 보기에는 아직 약하다.
- 같은 DIV2K val 샘플에서 Stage4 XL edge와 비교하면 Stage4 edge가 더 많이 건드려
  cleanup 효과는 강하지만, 둘 다 GT의 fine texture를 복원하지는 못한다.

결론:

- residual refiner는 `mild`에 과적합된 실패가 아니다. v2/v3에서도 condition-only를
  안정적으로 이긴다.
- 현재 best refiner는 안전한 미세 보정기다. final SR 모델이라기보다 Stage4 residual
  teacher/warm-start로 쓰기 적합하다.
- 다음 우선순위는 다음 중 하나다.
  - refiner capacity/loss를 조금 키워 눈에 보이는 detail gain이 커지는지 확인.
  - Stage4 diffusion U-Net에 refiner residual/gate target을 직접 supervision으로 넣기.
  - Stage2가 detail-need/uncertainty map을 내도록 해서 refiner/Stage4 gate 조건으로 쓰기.

HF 추가 보존:

```text
metrics/eval_residual_refiner_stage2_xl_mild_val100_summary.json
metrics/eval_residual_refiner_stage2_xl_photo_v2_val100_summary.json
metrics/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_summary.json
samples/eval_residual_refiner_stage2_xl_mild_val100_grid.png
samples/eval_residual_refiner_stage2_xl_photo_v2_val100_grid.png
samples/eval_residual_refiner_stage2_xl_photo_v3_noise_mix_val100_grid.png
samples/compare_residual_refiner_vs_stage4_edge_0801_photo_v3.png
```

## 실험 7: deterministic refiner teacher supervision Stage4 probe

목표:

- sparse-gate residual refiner의 residual/highpass/gate를 frozen teacher target으로 사용한다.
- gated-residual Stage4가 near-identity에 머무르지 않고 필요한 detail 위치와 크기를
  직접 학습할 수 있는지 확인한다.
- `photo_v3_noise_mix`에서 cleanup 이득과 사용자 체감 detail 복원을 같이 확인한다.

설정:

- config:
  `configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml`
- init: gated-residual mild step `2000`
- teacher: sparse-gate residual refiner best step `500`
- batch `8`, grad accumulation `4`
- 완료: `8000` micro steps = `2000` optimizer updates
- W&B:
  - step 0-2000: <https://wandb.ai/jwheo/sr-diffusion/runs/6h0124us>
  - step 2000-8000: <https://wandb.ai/jwheo/sr-diffusion/runs/0p3lfqt7>
- GPU 병목 없음: L40S util `97-100%`, steady speed 약 `0.85 micro-step/s`

one-step decoded proxy는 step `2000` 이후 개선되지 않았다:

| checkpoint step | decoded PSNR |
| ---: | ---: |
| 2000 | 21.5888 |
| 4000 | 21.5886 |
| 8000 | 21.5669 |

`photo_v3_noise_mix` sampled val100, condition init, 32 steps:

| checkpoint | start timestep | SR PSNR | bicubic PSNR | bicubic 대비 | condition 대비 | condition wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher Stage4 step 2000 | 25 | 22.9640 | 22.3599 | +0.6041 | +0.0626 | 68/100 |
| teacher Stage4 step 2000 | 50 | 22.9639 | 22.3599 | +0.6040 | +0.0625 | n/a |
| teacher Stage4 step 4000 | 25 | 22.9571 | 22.3599 | +0.5972 | +0.0557 | 65/100 |
| teacher Stage4 step 8000 | 25 | 22.9490 | 22.3599 | +0.5891 | +0.0476 | 59/100 |
| 기존 Stage4 edge step 4250 | 25 | 22.9563 | 22.3599 | +0.5964 | +0.0549 | 42/100 |
| 기존 Stage4 edge step 4250 | 50 | 23.0799 | 22.3599 | +0.7200 | +0.1784 | 45/100 |

시각/주파수 관찰:

- teacher step 2000은 t25에서 condition과 edge t25를 PSNR 기준으로 소폭 이긴다.
- 하지만 털, 잎, 나뭇가지, 건물 같은 고주파 구조를 복원하지 못하고 매끈한 덩어리로
  바꾸는 경향이 강하다.
- 평균 absolute-Laplacian energy는 teacher step 2000 SR이 GT의 `21.8%`이고,
  기존 edge t25는 GT의 `32.7%`다. 이 값은 정식 perceptual metric은 아니지만
  teacher 출력이 더 부드럽다는 시각 관찰과 일치한다.
- t25와 t50 결과가 거의 같아, teacher-supervised gated residual sampler가
  start timestep 변화에도 유용한 새 detail을 만들지 못한다.
- `photo_v3_noise_mix` 입력 중 일부는 색/센서 노이즈가 과도하게 강하다. 현재 curriculum은
  사용자 체감 SR보다 denoise/cleanup 학습을 과하게 유도할 가능성이 높다.

결론:

- teacher supervision은 수치상 안정적인 cleanup residual을 전달하는 데는 성공했다.
- 그러나 사용자가 기대하는 업스케일 detail 생성 목표에는 실패했다.
- step `2000` 이후 긴 continuation은 오히려 sampled PSNR과 condition win count가 감소했다.
- 다음 실험은 같은 Stage4를 더 오래 돌리거나 teacher weight만 조정하지 않는다.
- 우선순위는 degradation curriculum을 현실적인 강도로 재설계하고, clean/mild 비중을
  높인 고주파 복원 평가를 별도로 두는 것이다.

## 실험 8: detail-preserving curriculum Stage4 long adaptation

문제:

- `photo_v3_noise_mix`는 clean 샘플이 없고 `photo_v2`/`photo_v3_noise`가 합계 `80%`다.
- sample logging에서 일부 LR은 색/센서 노이즈가 과도해, 업스케일보다 denoise/cleanup
  학습을 강하게 유도했다.
- Stage2 condition은 mild/detail 입력에서 이미 구조와 질감을 잘 보존하므로 Stage2를
  즉시 재학습하는 것보다 Stage4 학습 분포를 먼저 바로잡는 편이 타당했다.

추가:

- `configs/degradation_presets.yaml`
  - `photo_detail`
  - `photo_detail_mix`: clean `35%`, photo_detail `48%`, mild `15%`, photo_v2 `2%`
- `analyze_degradation_presets.py`
- `configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml`

val100 degradation audit:

| preset | bicubic PSNR | LR chroma RMS vs clean | LR TV ratio vs clean |
| --- | ---: | ---: | ---: |
| `clean` | 25.0575 | 0.00000 | 1.0000 |
| `photo_detail` | 24.6502 | 0.00513 | 1.0041 |
| `photo_detail_mix` | 24.7357 | 0.00507 | 1.0174 |
| `mild` | 24.4778 | 0.00776 | 1.0205 |
| `photo_v2` | 22.4103 | 0.02003 | 1.1658 |
| `photo_v3_noise_mix` | 22.3599 | 0.02040 | 1.1879 |

기존 Stage2 XL baseline:

| preset | bicubic PSNR | condition PSNR | condition-bicubic |
| --- | ---: | ---: | ---: |
| `photo_detail` | 24.6502 | 25.2067 | +0.5565 |
| `photo_detail_mix` | 24.7357 | 25.3103 | +0.5745 |

결론:

- Stage2 구조가 처음부터 잘못된 것은 아니다.
- 기존 Stage2는 detail-preserving 입력에서 구조/질감을 실제로 복원하므로 동결 유지했다.
- 이전 smoothing의 주요 원인은 Stage4 objective/teacher 한계와 과도한 degradation
  curriculum의 결합이었다.

Stage4 장기 적응:

- init: teacher-supervised Stage4 step `2000`
- degradation: `photo_detail_mix`
- batch `8`, grad accumulation `4`
- lr `1e-6`
- 완료 `12000` micro steps = `3000` optimizer updates
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/so0lbyte>
- L40S util `99-100%`, VRAM 약 `45.0/46.1GB`, steady `0.856 micro-step/s`

sampled `photo_detail_mix` val100, condition init, t25, 32 steps:

| 모델/checkpoint | SR PSNR | bicubic 대비 | condition 대비 | condition wins |
| --- | ---: | ---: | ---: | ---: |
| Stage2 condition-only | 25.3103 | +0.5745 | n/a | n/a |
| teacher Stage4 init | 25.3187 | +0.5829 | +0.0084 | 46/100 |
| photo-detail Stage4 best step 8000 | 25.3406 | +0.6049 | +0.0303 | 71/100 |
| photo-detail Stage4 latest step 12000 | 25.3337 | +0.5980 | +0.0235 | 67/100 |
| 기존 edge Stage4 step 4250 | 25.1176 | +0.3818 | -0.1927 | 13/100 |

시각/주파수 관찰:

- step 8000은 condition의 구조와 선명도를 유지하면서 작은 residual correction을 더한다.
- 기존 edge Stage4의 넓은 over-editing과 smoothing은 크게 줄었다.
- step 12000은 step 8000과 시각적으로 유사하지만 sampled PSNR/승률은 소폭 후퇴했다.
- 평균 absolute-Laplacian energy ratio:
  - teacher init: GT의 `29.6%`
  - best step 8000: GT의 `29.7%`
  - latest step 12000: GT의 `29.9%`
  - edge Stage4: GT의 `41.2%`
- 즉 이번 성공은 fake texture를 크게 늘린 결과가 아니라, condition을 보존하면서
  correction 정확도를 높인 결과다.
- 2% `photo_v2` strong tail에서는 동상 표면의 밝은 점 같은 artifact가 여전히 보인다.

결론:

- curriculum 변경은 성공했다. gated-residual Stage4가 처음으로 condition-only를
  평균 PSNR과 condition win count 모두에서 명확히 이겼다.
- 공식 선택은 step `8000`; 동일 설정의 더 긴 continuation은 우선순위가 아니다.
- 아직 강한 missing-detail generator는 아니다. 다음은 perceptual/detail 평가를
  강화하고 strong tail을 별도 robustness 경로로 분리하는 방향이 타당하다.

## 실험 9: residual refiner v2 decoded-detail 장기 학습 및 40k continuation

목표:

- 기존 sparse-gate refiner의 안전성은 유지하면서 보정 폭과 decoded detail을 늘린다.
- VAE decoder를 통과한 image/highpass supervision을 사용한다.
- 초기 12k 결과가 계속 상승할 여지가 있는지 lower-LR continuation으로 검증한다.

설정:

- 초기 config: `configs/residual_refiner_stage2_xl_photo_detail_v2_long.yaml`
- continuation config: `configs/residual_refiner_stage2_xl_photo_detail_v2_continue_40k.yaml`
- Stage1/Stage2 frozen, hidden channels `192`, residual blocks `12`
- batch `12`, grad accumulation `2`, effective batch `24`
- continuation LR `2.5e-5`, 완료 `40000` micro steps
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/3v6wmf5o>
- L40S util `99-100%`, VRAM 약 `41.8/46.1GB`, steady `0.87~0.91 step/s`

`photo_detail_mix` val100 주요 checkpoint:

| checkpoint | refined global PSNR | global delta | mean delta | SSIM delta | wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11000 | 23.8356 | +0.0979 | +0.1318 | n/a | 90/100 |
| 20000 | 23.9300 | +0.1922 | +0.2290 | +0.00878 | 92/100 |
| 30000 | 23.9802 | +0.2425 | +0.2991 | +0.00930 | 95/100 |
| 39000 best | 24.0305 | +0.2927 | +0.3307 | +0.01076 | 94/100 |
| 40000 latest | 24.0281 | +0.2904 | +0.3262 | +0.01161 | 91/100 |

선택 step `39000` cross-preset val100:

| degradation | condition mean PSNR | refined mean PSNR | mean delta | wins | detail wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `photo_detail_mix` | 25.3103 | 25.6410 | +0.3307 | 94/100 | 72/100 |
| `mild` | 25.0449 | 25.3161 | +0.2712 | 91/100 | 76/100 |
| `photo_v2` | 22.9271 | 23.0419 | +0.1148 | 81/100 | 54/100 |
| `photo_v3_noise_mix` | 22.9014 | 23.0787 | +0.1773 | 81/100 | 59/100 |

관찰과 결론:

- 초기 판단과 달리 step 11000 이후에도 lower-LR continuation은 유의미하게 상승했다.
- step 39000은 global decoded PSNR 최고이며 새 공개 기본 checkpoint로 선택한다.
- step 40000은 SSIM delta가 더 높지만 PSNR과 승률이 소폭 낮고 detail energy가 더 커서
  기본값으로는 step 39000이 더 균형적이다.
- 모든 preset에서 평균 PSNR 이득은 증가했다. 다만 strong preset의 승률은 step 11000보다
  낮아져 더 큰 correction이 일부 샘플을 악화시키는 tail risk가 확인됐다.
- residual strength sweep으로 재학습 없는 guardrail을 검증했다.

| strength | photo-detail mean/wins | mild mean/wins | photo_v2 mean/wins | photo_v3 mean/wins |
| ---: | ---: | ---: | ---: | ---: |
| `1.00` | +0.3307 / 94 | +0.2712 / 91 | +0.1148 / 81 | +0.1773 / 81 |
| `0.90` | +0.3227 / 95 | +0.2648 / 93 | +0.1133 / 83 | +0.1755 / 81 |
| `0.75` | +0.2997 / 95 | +0.2460 / 94 | +0.1077 / 83 | +0.1661 / 83 |
| `0.50` | +0.2290 / 97 | +0.1882 / 95 | +0.0840 / 86 | +0.1269 / 86 |

- `1.0`은 평균 품질 최고, `0.75`는 balanced, `0.5`는 strong-tail 승률 우선 모드로
  추론 CLI와 Colab에 노출한다. 자동 degradation 판별기는 아직 신뢰 근거가 없어 넣지 않는다.
- 동일 샘플 시각 비교 리포트에서 clean/mild 입력은 Refiner가 구조를 보존하며
  소폭 개선했지만, strong 입력은 Condition 단계에서 세부가 이미 크게 사라지고
  청록/흰 격자형 점도 남았다. Refiner 강도 변경만으로는 이 병목을 해결하지 못했다.
- 공개 생성형 SOTA보다 미세 질감과 선명도는 아직 크게 부족하며, 현재 강점은
  낮은 환각 위험과 deterministic한 구조 보존이다.
- 다음 작업은 실사용/detail-focused blind A/B, perceptual metric 추가,
  degradation-aware gate, Condition 표현 개선이다.

## 2026-06-07 Stage 2 decoded-detail loss probe

Residual Refiner 결과를 `LR / bicubic / condition / refined / VAE oracle /
GT`로 다시 비교했다. VAE oracle은 GT와 거의 동일하게 선명했지만 Condition
출력부터 털, 잎맥, 글자, 먼 구조가 사라졌다.

```text
photo_detail_mix val100:
  VAE oracle mean PSNR:          41.8124
  Stage 2 condition mean PSNR:   25.3103
  Condition Laplacian ratio:      0.2891
  Refined Laplacian ratio:        0.3237
```

따라서 현재 주 병목은 Stage 1 VAE가 아니라 latent Charbonnier만으로 학습한
Stage 2 Condition encoder라고 판단했다. 기존 Stage 2 XL step 72000에서
초기화하고 다음 손실을 직접 적용하는 5000-step probe를 시작했다.

```text
config: configs/latent_pretrain_photo100k_xl_stage2_detail_loss_probe.yaml
loss: latent 0.25 + decoded 1.0 + edge 1.0 + highpass 2.0
      + highpass residual magnitude 1.0
data: photo_detail_mix
effective batch: 8 x grad_accum 4 = 32
W&B: https://wandb.ai/jwheo/sr-diffusion/runs/hgr8ilhk
```

초기 실행은 L40S 한 장에서 VRAM 약 `32.9/46.1GB`, GPU util `100%`,
steady 약 `1.31 micro-step/s`였다. 판정 기준은 decoded PSNR만이 아니라
Laplacian detail ratio와 고정 샘플의 실제 질감 복원이다. 이 probe에서
Condition 선명도가 움직이지 않으면 단일 해상도 residual CNN을 멀티스케일
Stage 2 구조로 교체한다.
