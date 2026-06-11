# High-Frequency Detail Branch v1 설계 메모

## 왜 필요한가

최근 실험의 공통 결론은 다음과 같다.

- Stage 2 condition encoder는 구조, 색, 저주파 복원을 이미 꽤 잘한다.
- 남은 recoverable error는 대부분 high-frequency residual이다.
- Stage 4 diffusion이 full x0/image를 다시 예측하면 condition을 쉽게 손상한다.
- residual refiner v2는 안전하지만 visible detail 생성량이 작다.
- Stage 2 continuation, VGG feature continuation, LSDIR unique-data 확장은 모두
  PSNR을 조금 올렸지만 사용자 체감 detail 문제를 해결하지 못했다.

따라서 다음 구조는 전체 이미지를 다시 맞추는 모델이 아니라, condition 위에
필요한 고주파 residual만 제한적으로 합성하는 별도 branch여야 한다.

## 목표

```text
LR -> frozen Stage 2 condition -> frozen Stage 1 decoder -> base SR
   -> detail branch predicts bounded high-frequency residual + gate
   -> SR_detail = base SR + gate * bounded high-frequency residual
```

목표는 PSNR 단독 최고가 아니다. 성공 조건은 fixed review set에서 다음을 동시에
만족하는 것이다.

- baseline residual refiner v2 대비 visible texture/detail 개선이 blind A/B에서 보일 것.
- `photo_detail_mix`와 `mild`에서 PSNR/SSIM이 크게 후퇴하지 않을 것.
- `photo_v2`/`photo_v3_noise_mix` strong tail에서 흰 점, grid, cyan/green artifact가
  늘지 않을 것.
- Laplacian/highpass ratio가 GT 쪽으로 올라가되, highpass L1과 artifact review가 함께
  악화되지 않을 것.

## v1 구조

입력:

```text
LR upsampled to HR
base SR image from Stage 2 + Stage 1 decoder
condition latent
optional degradation/domain embedding
```

출력:

```text
residual_logits: RGB high-frequency residual
gate_logits: per-pixel or low-channel gate
```

합성:

```text
residual = residual_scale * tanh(residual_logits)
gate = sigmoid(gate_logits + gate_bias)
sr = clamp(base_sr + gate * highpass_project(residual), 0, 1)
```

`highpass_project`는 v1에서 residual의 local mean을 제거하는 간단한 blur-subtract로
시작한다. 이렇게 하면 branch가 색/밝기 전체를 바꾸는 경로를 줄이고 texture/detail
수정에 집중한다.

## 학습 목표

기본 loss:

```text
L1(sr, gt)
SSIM or Charbonnier reconstruction
highpass L1(sr, gt)
Laplacian L1(sr, gt)
gate sparsity
low-frequency anchor: lowpass(sr) ~= lowpass(base_sr)
```

선택 loss:

```text
LPIPS/DISTS-style perceptual loss
patch adversarial loss on residual/detail only
teacher feature distillation from a non-runtime restoration teacher
```

GAN/adversarial loss를 넣을 경우 v1에서는 full image discriminator보다 residual/detail
patch discriminator가 우선이다. full image GAN은 fake texture와 색 변형을 키울 수 있다.

## 평가 순서

1. `tools/eval/build_fixed_review_set.py`로 `detail_v1` fixed set 생성.
2. 현재 Colab default residual refiner v2를 `run_fixed_review_residual_refiner.py`로 평가.
3. `eval_fixed_review_outputs.py`로 PSNR/SSIM/detail metric/contact sheet/HTML 생성.
4. detail branch v1은 같은 fixed set에서 residual refiner v2와 비교한다.
5. 수치가 좋아도 HTML/contact sheet에서 texture가 fake처럼 보이면 실패로 기록한다.

## 하지 않을 것

- Stage 2/Stage 4 같은 checkpoint를 더 오래 continuation하는 것을 우선하지 않는다.
- PSNR `+0.01 dB` 개선만으로 모델을 승격하지 않는다.
- strong degradation을 train mix의 대부분으로 두지 않는다.
- pretrained T2I 모델을 runtime dependency로 넣지 않는다.

