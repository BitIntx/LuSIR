# LuSIR High-Frequency Detail Branch v1 설계 메모

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

## 현재 구현

구현 파일:

```text
tools/train/train_detail_branch.py
tools/eval/run_fixed_review_detail_branch.py
configs/detail_branch_v1_photo130k_lsdir.yaml
configs/detail_branch_v1b_aug_photo130k_lsdir.yaml
configs/detail_branch_v1c_condition_open_photo130k_lsdir.yaml
configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
configs/hf/detail_branch_v1b_aug_photo130k_lsdir.yaml
tests/test_detail_branch.py
```

현재 config는 최신 보존 Stage 2 dual-context LSDIR best98000을 frozen base로
사용한다.

```text
LR -> Stage 2 dual-context condition -> Stage 1 decoder -> base SR
base SR + bicubic LR upsample -> detail branch -> detail SR
```

이 실험은 Stage 3/4 diffusion sampling을 사용하지 않는다. Stage 2와 Stage 1은
frozen이고, image-space detail branch만 학습한다. branch output convolution은
zero-init이라 step 0 출력은 base SR과 정확히 같다.

Smoke 확인:

```text
command:
  python tools/train/train_detail_branch.py \
    --config configs/detail_branch_v1_photo130k_lsdir.yaml \
    --limit-steps 4 \
    --output-dir /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1_smoke_update \
    --disable-wandb

base/sr step 0:
  eval/base_psnr = eval/sr_psnr = 24.6188

after 4 micro-steps = 1 optimizer update:
  eval/sr_vs_base_psnr      +0.00005 dB
  eval/sr_vs_base_mean_psnr +0.00001 dB
  wins_vs_base              69/100
```

Smoke 수치는 성능 주장이 아니라 load/eval/backprop/update/checkpoint 경로가
정상 작동한다는 확인이다.

장기 학습:

```bash
python tools/train/train_detail_branch.py \
  --config configs/detail_branch_v1_photo130k_lsdir.yaml
```

주의: `train/max_steps`는 micro-step 기준이다. 현재 `grad_accum_steps: 4`이므로
`40000` micro-steps는 `10000` optimizer updates다.

v1b augmentation run:

```bash
python tools/train/train_detail_branch.py \
  --config configs/detail_branch_v1b_aug_photo130k_lsdir.yaml
```

v1b는 회전/affine/perspective/vertical flip을 쓰지 않는다. 추가되는 증강은
`hflip_prob: 0.5`, `texture_crop_retries: 4`, 약한 HR color jitter
(`[0.97, 1.03]`, probability `0.25`)뿐이다. 첫 v1 run은 step `7800`,
즉 `0.234 epoch`에서 멈추고 v1b로 전환했다.

## v1b 완료 결과

v1b augmentation run은 `40000` micro-steps에서 정상 종료됐다.
`grad_accum_steps: 4`이므로 이는 `10000` optimizer updates이고, train
`133450`장 기준 약 `1.199 epoch`다.

```text
run:       detail_branch_v1b_aug_photo130k_lsdir
config:    configs/detail_branch_v1b_aug_photo130k_lsdir.yaml
W&B:       https://wandb.ai/jwheo/LuSIR/runs/1o3aavi9
selected:  step 39500, eval/detail_score best
local ckpt:
  /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1b_aug_photo130k_lsdir/checkpoints/best_eval_detail.pt
HF path:
  checkpoints/detail_branch_v1b_aug_photo130k_lsdir_best39500.pt
```

선택 checkpoint의 val100 지표:

| metric | value |
| --- | ---: |
| base PSNR | 24.6188 |
| detail PSNR | 24.6649 |
| PSNR delta | +0.0461 dB |
| base SSIM | 0.80013 |
| detail SSIM | 0.80281 |
| SSIM delta | +0.00268 |
| mean PSNR delta | +0.0575 |
| wins vs base | 98/100 |
| detail wins vs base | 100/100 |

근접 checkpoint별 peak:

| selection | step | value |
| --- | ---: | ---: |
| best detail score | 39500 | 26.53945 |
| best PSNR delta | 38500 | +0.0489 dB |
| best SSIM delta | 37000 | +0.00336 |
| final | 40000 | +0.0444 dB PSNR, +0.00277 SSIM, 98/100 wins |

판단:

- v1 대비 augmentation + 장기 학습은 수치상 명확히 좋아졌다.
- residual은 여전히 작고 안정적이다. 흰 점, grid, cyan/green artifact를
  키우는 방향은 아니다.
- 라임/털/풀/건물 edge처럼 texture-heavy crop에서 얇은 고주파 보강이 보인다.
- 다만 base와 detail 차이는 작고, GT 수준의 표면 질감 복원에는 아직 못 미친다.
- 따라서 v1b는 최신 public detail artifact로 보존하되, Colab 기본값으로
  승격하려면 단일 이미지/tiled inference runner와 WebUI 통합이 먼저 필요하다.

fixed review set 평가:

```bash
python tools/eval/run_fixed_review_detail_branch.py \
  --config configs/detail_branch_v1b_aug_photo130k_lsdir.yaml \
  --checkpoint /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1b_aug_photo130k_lsdir/checkpoints/best_eval_detail.pt \
  --review-manifest /home/ubuntu/scratch/sr-diffusion/review_sets/detail_v1/review_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/review_outputs/detail_branch_v1b_aug_detail_v1

python tools/eval/eval_fixed_review_outputs.py \
  --review-manifest /home/ubuntu/scratch/sr-diffusion/review_sets/detail_v1/review_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/review_reports/detail_branch_v1b_aug_detail_v1 \
  --candidate base=/home/ubuntu/scratch/sr-diffusion/review_outputs/detail_branch_v1b_aug_detail_v1/samples/{id}/base.png \
  --candidate detail=/home/ubuntu/scratch/sr-diffusion/review_outputs/detail_branch_v1b_aug_detail_v1/samples/{id}/detail.png
```

## v1c와 v1d capacity 실험

V1b는 안정적이지만 residual/gate가 너무 보수적이었다. V1c는 v1b selected
checkpoint에서 시작해 frozen Stage 2 condition latent를 branch 입력에 직접
노출하고, residual scale과 초기 gate를 조금 더 열었다.

```text
v1c config: configs/detail_branch_v1c_condition_open_photo130k_lsdir.yaml
selected:   step 6000
PSNR delta: +0.0554 dB
SSIM delta: +0.00332
wins:       99/100
```

V1d는 v1c의 objective와 width를 유지하고 residual block만 `8 -> 18`로
늘려 branch 파라미터를 `1.35M -> 3.02M`으로 확장한다. 기존 block은 복사하고
추가 block은 identity-init하므로 시작 출력은 v1c와 정확히 같다.

```text
v1d config: configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
target:     100086 micro-steps = exactly 3 epoch
active W&B: https://wandb.ai/jwheo/LuSIR/runs/ctg4r7n9
```

technical report의 selected step `9500` snapshot:

| protocol | result |
| --- | ---: |
| `photo_detail_mix` aggregate PSNR delta | +0.0638 dB |
| `photo_detail_mix` SSIM delta | +0.00337 |
| `photo_detail_mix` wins | 100/100 |
| strict-bicubic DIV2K five-crop RGB PSNR | 31.8247 dB |
| strict-bicubic gain over v1c | +0.0093 dB |

판단:

- v1c와 v1d 모두 v1b보다 수치상 개선됐지만 변화는 여전히 작다.
- v1d의 추가 1.67M 파라미터가 만든 strict-bicubic 이득은 현재
  `+0.0093 dB`뿐이다.
- v1d가 계속 보수적이면 다음 단계는 용량 추가보다 perceptual/detail-only
  adversarial supervision과 blind visual comparison이다.

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
