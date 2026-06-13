# LuSIR High-Frequency Residual Diffusion 설계 메모

## 배경

LuSIR의 deterministic 경로는 정식 clean-bicubic x4 benchmark에서 다음
수준까지 올라왔다.

```text
LuSIR detail v1d DIV2K Y PSNR/SSIM: 30.1602 / 0.83421
SwinIR classical x4:                31.0838 / 0.85228
remaining gap:                     0.9235 dB / 0.01807
```

일반 W&B `eval/decoded_psnr` 약 `25.05`는 task-specific degradation을 사용한
val100 학습 proxy다. 공개 classical SR 표의 full-image Y-channel PSNR과 직접
비교하지 않는다.

Stage2 clean-bicubic continuation은 원래 LR `5e-6`에서 step `15000`
`decoded_psnr=25.057`까지 완만하게 개선된 뒤 plateau했다. LR 변경 실험도
병목을 바꾸지 못했다.

```text
LR 20x, step 15500: decoded_psnr 15.72, 즉시 붕괴
LR 5x from-init, step 4000: 25.033
original LR, step 4000:     25.031
```

따라서 같은 Stage2 objective를 더 오래 또는 더 큰 LR로 학습하는 것은
clean-fidelity를 조금 다듬을 수는 있어도 사용자가 원하는 visible fine detail
생성 경로가 되기 어렵다.

## 목표 분리

다음 실험은 하나의 checkpoint가 PSNR과 perceptual detail을 동시에 최적화하도록
강요하지 않는다.

```text
Fidelity/base path:
  LR -> frozen Stage2 base -> Stage1 decoder

Stochastic detail path:
  frozen condition latent + noised condition latent + timestep
    -> gated high-frequency residual diffusion
    -> condition latent + bounded residual
    -> Stage1 decoder
```

성공 목표:

- base의 구조, 색, 저주파를 유지한다.
- 잎, 털, 잔디, 표면 질감처럼 LR에서 유일하게 복원할 수 없는 세부를
  여러 seed로 그럴듯하게 합성한다.
- PSNR 단독 승격 대신 LPIPS/DISTS, high-frequency metric, fixed visual review,
  seed diversity를 함께 본다.
- full-image를 다시 그려 생기는 색 변형, 흰 점, grid, cyan/green artifact를
  억제한다.

## 첫 실험 범위

기존 `train_diffusion.py`의 다음 구현을 재사용한다.

- condition-start diffusion
- `prediction_type: gated_residual_x0`
- bounded residual + learned gate
- residual highpass magnitude loss
- lowpass anchor
- detail gate anchor
- frozen Stage2 condition encoder와 Stage1 VAE

첫 실험은 구조를 완전히 새로 추가하기 전에 기존 검증된 경로를 강하게
고주파 전용으로 제한하는 probe다.

구현된 첫 설정:

```text
config:            configs/diffusion_photo130k_lsdir_highfreq_residual_v1_b8.yaml
condition encoder: frozen dual-context LSDIR Stage2 best98000
data:              photo_detail_mix
train init:        condition
prediction:        gated_residual_x0
timestep range:    5..125, sample/eval timestep 50
U-Net:             76.6M, output layer zero-init
batch:             8, grad accumulation 4, effective batch 32
decoded weight:    0.03
highpass/residual highpass magnitude: 1.0 / 2.0
lowpass anchor:    3.0
teacher:           첫 probe에서는 사용하지 않음
```

기존 XL Stage4 checkpoint를 무조건 이어받지 않는다. 기존 Stage4는
strong-cleanup objective 때문에 clean input을 과수정한 기록이 있다. 새 probe는
검증 가능한 중형 U-Net을 새로 학습한다.

스모크 결과:

- 출력층 zero-init 상태에서 `Condition`과 `PredX0` 저장 이미지가 byte-for-byte
  동일했다.
- initial `eval/latent_residual_l1=0`,
  `eval/decoded_vs_condition_mse=0`, PSNR delta `0.000 dB`다.
- 단일 L40S의 batch 8 peak VRAM은 약 `30.3GB / 46.1GB`로 장기 학습 여유가
  있다.
- 학습 로그에 condition PSNR, condition 대비 PSNR delta, residual L1,
  gate mean을 추가했고 샘플에는 `Condition`과 `AbsDelta4x`를 추가했다.

첫 v1 run:

```text
tmux: highfreq-residual-v1
W&B:  https://wandb.ai/jwheo/LuSIR/runs/q3t4hzms
log:  /home/ubuntu/scratch/sr-diffusion/runs/diffusion_photo130k_lsdir_highfreq_residual_v1_b8/train_console.log
```

초기 안정 구간은 약 `1.10 micro-step/s`, GPU util `99%`, VRAM 약
`30.3/46.1GB`였다. 그러나 이 run은 step `1650` 부근에서 중단했다.

```text
eval PSNR delta vs condition:
  step 500:  -0.002 dB
  step 1000: -0.017 dB
  step 1500: -0.037 dB

fixed sample:
  condition Laplacian energy: 0.005923
  step1500 Laplacian energy:  0.007660
  condition Laplacian L1 to GT: 0.009706
  step1500 Laplacian L1 to GT:  0.010581
```

고주파 에너지는 늘었지만 GT 방향의 detail 오차도 함께 악화했다. 샘플에서도
유효한 질감보다 미세한 반복 패턴이 증가했다. 따라서 “초반 tradeoff”로
간주하지 않고 실패 probe로 기록한다.

핵심 원인은 현재 v1이 condition latent를 노이즈화한 뒤 deterministic target
x0를 맞추며, 방향을 버린 residual magnitude loss를 강하게 사용하는 점이다.
다음 버전은 loss weight만 조정하지 않고 다음처럼 residual-space diffusion으로
구조를 바꾼다.

```text
target residual = target latent - frozen condition latent
noisy residual  = add_noise(target residual, noise, timestep)
model input     = noisy residual + frozen condition latent + timestep
model target    = noise
output          = frozen condition latent + bounded predicted residual
```

## Wavelet residual diffusion v2 구현

v2는 위 가설을 image-space signed Haar wavelet residual로 구현한다.

```text
frozen base:
  LR -> Stage2 dual-context -> Stage1 decoder -> detail v1d

diffusion target:
  HaarHigh(GT - detail v1d)

model:
  noisy signed high bands + Haar(detail v1d) + Haar(bicubic)
    -> 18.44M conditional U-Net
    -> predicted noise

reconstruction:
  detail v1d + inverse Haar(LL=0, predicted LH/HL/HH)
```

- config: `configs/wavelet_residual_diffusion_v2_probe.yaml`
- trainer: `tools/train/train_wavelet_residual_diffusion.py`
- Haar implementation: `src/sr_diffusion/wavelet.py`
- LL 채널은 모델 출력에 존재하지 않아 저주파 수정이 구조적으로 금지된다.
- residual magnitude가 아니라 signed residual과 diffusion noise를 학습한다.
- residual coefficient 표준편차는 약 `0.0473`이며 `0.08`로 정규화한다.
- wavelet coefficient는 실제 분포의 극단 outlier만 제거하도록 ±`0.16`
  (`normalized clip_x0=2.0`)으로 제한한다.

동일 val8의 clipped oracle은 v1d base 대비 평균 PSNR을
`28.7012 -> 31.4039`, Laplacian L1을 `0.018342 -> 0.010721`로 개선했다.
따라서 표현 공간 자체에는 충분한 복원 가능성이 있다. 첫 probe는 `2000`
micro-step이며 250 step마다 3 seed DDIM sampling을 평가한다.

첫 pure-noise sampling probe는 step1000까지 noise MSE와 signed residual
오차가 감소했지만, sampled residual energy가 target의 약 10배로 남아 있었다.
SR refinement에 pure noise `t=99` 시작은 지나치게 공격적이라 중단했다.

두 번째 probe는 같은 step1000 가중치를 이어받고 zero residual에
`start_timestep=50`만큼 noise를 넣어 복원한다. 학습 timestep도 `0..75`로
제한한다.

```text
config: configs/wavelet_residual_diffusion_v2_condition_start_probe.yaml
initial checkpoint: wavelet_residual_diffusion_v2_probe step1000
```

동일 checkpoint의 condition-start smoke는 full-noise sampler보다 sampled
PSNR을 약 `18 dB -> 22.85 dB`, residual energy ratio를 `10배대 -> 3.95배`,
lowpass drift를 `0.006대 -> 0.00157`로 개선했다.

condition-start probe는 step `3000`, 즉 `375` optimizer update에서 정상
종료했다. `start_timestep=50` 평가는 학습 내내 단조롭게 개선됐다.

```text
step 1250: PSNR 23.1287, signed wavelet L1 0.05855
step 2000: PSNR 23.9915, signed wavelet L1 0.05118
step 3000: PSNR 25.0289, signed wavelet L1 0.04331
```

그러나 step3000 checkpoint의 sampling 강도 비교 결과는 아직 모든 설정에서
v1d base보다 나빴다.

| start timestep | sampled PSNR | delta vs v1d | residual energy ratio | 판단 |
| ---: | ---: | ---: | ---: | --- |
| 15 | 28.1239 | -0.5768 dB | 0.708 | 시각적으로 안전하지만 명확한 유효 detail 없음 |
| 25 | 27.4752 | -1.2256 dB | 1.116 | 미세 입자 증가, 승격 불가 |
| 50 | 25.0289 | -3.6719 dB | 2.503 | 균일한 노이즈가 강함 |

구조와 저주파는 유지됐지만 Laplacian/highpass GT 오차도 모든 강도에서
악화했다. 표현 공간 실패로 단정하기에는 optimizer update가 너무 적고,
noise/signed-wavelet 오차가 계속 감소 중이다. 동일 구조를 step `20000`까지
이어 학습하고 `t=25`를 중간 stress test로 추적한다.

```text
config: configs/wavelet_residual_diffusion_v2_condition_start_long.yaml
resume: condition-start probe step3000
W&B:    https://wandb.ai/jwheo/LuSIR/runs/zh1fktq4
```

첫 장기-run `t=25` 평가는 step3000 대비 계속 개선됐다.

```text
step3000: PSNR 27.4752, delta -1.2256 dB, signed wavelet L1 0.02788
step4000: PSNR 27.7734, delta -0.9274 dB, signed wavelet L1 0.02605
```

장기 run은 step `20000`, `2500` optimizer update에서 정상 종료했다.
`t=25` val8은 step3000 대비 크게 안정화됐지만, 후반에는 residual과 diversity가
계속 줄며 zero residual 쪽으로 수렴했다.

```text
step3000:  PSNR delta -1.2256 dB, energy ratio 1.116, diversity 0.01846
step10000: PSNR delta -0.4845 dB, energy ratio 0.618, diversity 0.01028
step20000: PSNR delta -0.2475 dB, energy ratio 0.431, diversity 0.00724
```

최종 checkpoint의 val100 강도 비교:

| start timestep | PSNR delta vs v1d | SSIM delta | Laplacian gain | energy ratio |
| ---: | ---: | ---: | ---: | ---: |
| 15 | -0.0880 dB | -0.00647 | -0.000833 | 0.267 |
| 25 | -0.1392 dB | -0.01040 | -0.001229 | 0.336 |
| 50 | -0.3152 dB | -0.02433 | -0.002516 | 0.510 |

시각적으로 노이즈는 사라졌지만 v1d에 없던 의미 있는 질감도 만들지 못했다.
모든 강도에서 GT-aligned Laplacian/highpass 오차가 악화하고, 강도를 높이면
다양성과 함께 오류만 증가한다. 따라서 학습 부족 가설은 폐기하며 이 모델은
승격하지 않는다.

표현 공간의 oracle 가능성은 남아 있지만, 현재 조건과 noise-MSE objective는
불확실한 residual을 conditional mean, 즉 거의 zero residual로 축소한다.
다음 시도는 같은 설정의 continuation이나 단순 EMA가 아니라 다음 중 하나여야
한다.

- LR에서 근거가 있는 위치만 선택하는 learned uncertainty/detail mask
- stochastic branch에 대한 discriminator 또는 patch-level perceptual objective
- GT residual 직접 생성 대신 texture prior/teacher가 만든 detail target
- fidelity base와 생성 detail을 사용자 강도로 혼합하는 명시적 two-head 구조

## 평가

학습 중 확인:

- train loss가 아니라 fixed sample의 condition/base/detail/GT 비교
- `eval/decoded_mse`와 condition 대비 변화량
- residual/gate 크기
- lowpass drift
- highpass/Laplacian energy가 GT에 가까워지는지

checkpoint 비교:

1. deterministic base
2. deterministic detail v1d
3. residual diffusion seed 3개 이상
4. GT

승격 조건:

- fixed visual review에서 v1d보다 visible detail이 명확할 것
- low-frequency 구조와 색이 base에서 크게 움직이지 않을 것
- 여러 seed가 서로 다르되 의미 구조를 훼손하지 않을 것
- clean-bicubic PSNR 하락은 별도로 기록하고 숨기지 않을 것

## 하지 않을 것

- W&B val100 PSNR `+0.01 dB`만으로 승격하지 않는다.
- 생성형 결과를 classical SR SOTA PSNR과 같은 목표로 설명하지 않는다.
- full x0를 자유롭게 다시 그리는 장기 Stage4 continuation부터 시작하지 않는다.
- pretrained text-to-image model을 runtime dependency로 추가하지 않는다.
- wavelet v2를 같은 noise-MSE objective로 더 오래 continuation하지 않는다.
