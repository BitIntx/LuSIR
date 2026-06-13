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

장기 run:

```text
tmux: highfreq-residual-v1
W&B:  https://wandb.ai/jwheo/LuSIR/runs/q3t4hzms
log:  /home/ubuntu/scratch/sr-diffusion/runs/diffusion_photo130k_lsdir_highfreq_residual_v1_b8/train_console.log
```

초기 안정 구간은 약 `1.10 micro-step/s`, GPU util `99%`, VRAM 약
`30.3/46.1GB`다.

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
