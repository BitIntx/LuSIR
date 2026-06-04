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

현재 상태:

- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/lrb6nco9>
- PID on current VM: `76933`
- 시작 직후 정상:
  - GPU util: `99-100%`
  - VRAM: 약 `45.2/46.1GB`
  - speed: 약 `0.86-0.87 micro-step/s`
  - step 1 decoded proxy PSNR: `23.18`

초기 로그 예:

```text
step=1 loss=0.19193 noise_mse=35.00126 x0_mse=0.37928 decoded=0.08468
edge=0.04472 highpass=0.05081 res_edge_mag=0.04556 res_high_mag=0.03586
low_anchor=0.00533 detail_gate=0.02187 steps_per_sec=0.39
eval step=1 noise_mse=42.67671 decoded_psnr=23.18
```

평가 계획:

1. 학습 완료 후 best/latest checkpoint를 `mild` val100에서 sampled eval.
2. 우선 `start_timestep=25`, `init=condition`, `steps=32`.
3. 반드시 Stage2 condition-only `mild` 결과와 비교:
   - condition baseline: `25.0449 dB`
   - bicubic baseline: `24.4778 dB`
4. condition보다 평균 PSNR이 낮거나 condition을 이긴 샘플이 적으면 role split이 아직 부족한 것으로 판단.
5. 숫자와 별개로 comparison grid에서 fake texture가 줄었는지 확인.

## 현재 판단

- Stage1 VAE는 건드리지 않는다.
- Stage3로 되돌아가지 않는다.
- Stage2는 강한 base 역할을 이미 하고 있으므로 즉시 재학습하지 않는다.
- 우선순위는 Stage4의 역할 제한:
  - condition 저주파/구조 보존
  - 필요한 위치에서만 detail residual 허용
  - t25 이하의 작은 refinement
- role-split probe도 실패하면 다음은 architecture/parameterization 변경을 검토한다.
  예: U-Net이 full x0를 예측하지 않고 bounded residual 또는 gated residual만 예측하게 만들기.
