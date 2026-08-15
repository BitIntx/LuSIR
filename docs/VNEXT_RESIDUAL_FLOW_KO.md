# LuSIR vNext: Stage2-anchored Residual Flow 연구 계획

작성일: `2026-08-15`

상태: **연구 제안 / 미구현**

이 문서는 2026년 8월 기준 최신 SR 연구를 현재 LuSIR의 실험 결과와
대조해 정리한 다음 연구 방향이다. 이 문서에 적힌 모델, config, checkpoint는
아직 구현되거나 학습된 것이 아니다. 검증된 현재 기본값과 기존 실험 결과는
`README.md`, `docs/HANDOFF_KO.md`, `docs/TRIAL_AND_ERROR_KO.md`를 기준으로 한다.

## 한 줄 결론

Stage 1과 현재 Stage 2를 보존하고, 기존 Stage 3/4 DDPM 계열의 다음 연구
후보로 **Stage2가 예측한 latent를 fidelity anchor로 사용하는 conditional
residual flow**를 추가한다.

```text
LR
  -> frozen Stage2 condition encoder
  -> z_base
  -> z_base + relative noise
  -> conditional residual flow (1-step 또는 few-step)
  -> z_sr
  -> frozen Stage1 decoder
  -> SR
```

새 detail branch를 직렬로 하나 더 추가하거나 현재 public 기본 checkpoint를
교체하는 계획이 아니다. 먼저 독립적인 research path로 검증한다.

## 반드시 보존할 현재 기준

- Public/Colab 기본값:
  `guarded-detail Stage2 v2 step10000 -> Stage1 decoder`.
- Conservative deterministic 옵션:
  `Stage2 XL step72000 -> residual refiner v2 step39000 -> Stage1 decoder`.
- Stage1은 factor-4, 16-channel latent VAE이며 decoder audit에서 병목이 아닌
  것으로 확인됐다.
- 현재 핵심 병목은 Stage2가 one-to-many HR 가능성의 조건부 평균을 예측하며
  질감을 평활화하는 것이다.
- 기존 Stage3/4, detail branch, masked residual-shift, wavelet residual
  diffusion, PatchGAN 실험은 삭제하지 않고 비교 baseline으로 유지한다.
- 코드 제약은 그대로 유지한다.
  - vision-only x4 SR
  - pretrained text-to-image diffusion model 미사용
  - PyTorch/ROCm 우선
  - custom CUDA/ROCm op 미사용
  - photo와 anime/illustration domain을 같은 codebase에서 지원

## 이 방향을 선택한 이유

### 기존 deterministic 경로의 한계

Stage2의 latent regression은 PSNR과 구조 보존에는 강하지만, 동일 LR에서
가능한 여러 HR texture의 평균에 수렴하기 쉽다. 기존 실험에서는 모델 규모,
attention window, hard/soft detail mask, deterministic detail branch를 바꿔도
missing texture가 크게 회복되지 않았다.

### 기존 생성형 실험과 다른 점

이 계획은 이전 masked latent residual-shift 또는 wavelet residual diffusion의
단순 continuation이 아니다.

- hard top-k mask로 생성 영역을 제한하지 않는다.
- Haar high-frequency band만 따로 확산하지 않는다.
- DDPM의 target/noise parameterization을 그대로 재사용하지 않는다.
- Stage2 output을 단순 condition으로만 넣지 않고 **flow source distribution의
  중심**으로 사용한다.
- 1-step endpoint 품질과 few-step trajectory를 같은 모델에서 함께 검증한다.

### 최신 연구에서 가져올 핵심 아이디어

- [VOSR](https://arxiv.org/abs/2604.03225): T2I generator 없이도
  vision-only generative SR가 가능하며, LR 구조 조건과 시각 semantic 조건을
  분리하고 restoration-oriented guidance를 사용한다.
- [RFMSR](https://arxiv.org/abs/2607.12753): 순수 Gaussian 대신 LQ latent
  주변에서 flow를 시작해 transport distance를 줄이고, velocity loss를
  유지하면서 1-step endpoint supervision을 추가한다.
- [MeanSR](https://arxiv.org/abs/2608.09405): LR-conditioned average velocity로
  one-step restoration trajectory를 직접 학습한다. 2026-08-15 현재 매우 최신
  프리프린트이고 공식 구현이 확인되지 않았으므로 1차 구현이 아니라 후속
  ablation 후보로만 둔다.
- [FiDeSR](https://diffusion-sr.github.io/FiDeSR/): hard binary mask 대신
  detail/error-aware continuous loss weighting과 latent residual refinement를
  사용한다.
- [DASP-SR](https://arxiv.org/abs/2604.11470): degradation token과
  edge-aware training noise로 복원 모호성과 구조 손실을 줄인다.

논문의 pretrained VAE, pretrained diffusion backbone, checkpoint를 그대로
가져오지 않는다. LuSIR에 필요한 formulation과 ablation idea만 이식한다.

## 핵심 formulation

HR latent와 Stage2 base latent를 각각 다음과 같이 둔다.

```text
z_hr   = frozen_stage1.encode(HR).mean
z_base = frozen_stage2(LR, domain_id)
```

LuSIR용 source latent는 Stage2 output 주변에 둔다.

```text
residual_scale = std(z_hr - z_base), per-channel EMA 또는 dataset statistic
z_1 = z_base + sigma_rel * residual_scale * epsilon
epsilon ~ N(0, I)
```

직선 conditional flow path와 target velocity는 다음과 같다.

```text
z_t = (1 - t) * z_hr + t * z_1
v_target = z_1 - z_hr
v_pred = flow_model(z_t, t, z_base, domain_id, optional_conditions)
loss_velocity = MSE(v_pred, v_target)
```

1-step 추론은 다음과 같다.

```text
z_sr = z_1 - v_pred(z_1, t=1, z_base, ...)
```

few-step 추론은 동일 velocity field를 Euler 또는 Heun으로 `t=1 -> 0` 적분한다.
초기 구현은 표준 PyTorch Euler를 사용하고, flow가 유효한 것이 확인되기 전에는
새 solver dependency를 추가하지 않는다.

### noise scale 원칙

RFMSR의 absolute `sigma=1.0`을 그대로 복사하지 않는다. LuSIR의 custom VAE
latent scale이 다르므로 `z_hr - z_base` 통계에 상대적인 값을 사용한다.

첫 sweep:

```text
sigma_rel: [0.0, 0.25, 0.5, 1.0]
```

- `0.0`: deterministic flow sanity baseline.
- `0.25`, `0.5`: 구조 보존 우선 후보.
- `1.0`: 생성 여유가 실제 detail로 이어지는지 확인하는 상한 후보.

절대 sigma, seed diversity, residual magnitude를 모두 로그에 남긴다.

## 구현 단계

### Phase 0: baseline과 evaluator 고정

상태: `[ ] 미구현`

목표는 새 objective의 효과를 기존 모델 변경과 혼동하지 않게 하는 것이다.

1. frozen Stage1과 frozen Stage2 checkpoint를 명시적으로 고정한다.
2. 동일한 val100, formal 219-image, degradation review set을 사용한다.
3. current Stage2 condition-only 결과를 새 evaluator로 다시 저장한다.
4. 다음 지표를 한 summary에 기록한다.
   - RGB/Y PSNR, SSIM
   - LPIPS, DISTS
   - highpass/laplacian energy ratio
   - missing/excess detail energy
   - seed 간 output diversity
   - runtime, peak VRAM, parameter count, FLOPs 가능 시 기록
5. CLIPIQA, MUSIQ, MANIQA 같은 no-reference 지표는 보조 지표로만 사용하고
   checkpoint selection 단독 기준으로 사용하지 않는다.

### Phase 1: residual conditional flow baseline

상태: `[ ] 미구현`

첫 구현은 현재 `ConditionalUNet`을 재사용해 objective만 비교한다.

예상 추가 파일:

```text
src/sr_diffusion/models/residual_flow.py
tools/train/train_residual_flow.py
tools/infer/infer_residual_flow.py
configs/residual_flow_stage2_anchor_v1_probe.yaml
tests/test_residual_flow.py
```

초기 조건:

- Stage1 frozen.
- Stage2 frozen.
- current U-Net 규모를 우선 사용.
- velocity MSE만으로 시작.
- EMA model weight 사용 여부는 기존 trainer 관례와 맞춰 별도 ablation.
- `sigma_rel` sweep과 1/4/8-step 추론을 같은 checkpoint에서 비교.
- GAN, DINO, degradation token, learned mask는 모두 끈 상태로 시작.

필수 unit/smoke test:

- `t=0`에서 `z_t == z_hr`.
- `t=1`에서 `z_t == z_1`.
- oracle velocity로 1-step endpoint가 `z_hr`를 복원.
- `sigma_rel=0` deterministic reproducibility.
- 고정 seed reproducibility.
- batch/tile inference shape consistency.
- bf16과 fp32에서 NaN/Inf 없음.

### Phase 2: soft detail/error weighting

상태: `[ ] 미구현`

Phase 1이 구조를 보존하지만 여전히 평활한 경우에만 추가한다. 기존 hard top-k
mask와 별도 generator 대신 continuous weight를 사용한다.

초기 형태:

```text
detail = normalized_sobel_or_laplacian(HR)
error = stopgrad(normalized_abs(HR - decoded_prediction))
weight = clamp(1 + lambda_detail * detail + lambda_error * error,
               min=1, max=weight_max)
```

weighted decoded Charbonnier와 spatial perceptual loss를 작은 weight로 추가한다.
전체 loss의 global mean과 weighted-region mean을 각각 로그한다.

첫 ablation:

```text
A: velocity only
B: velocity + decoded Charbonnier
C: B + soft detail weight
D: C + LPIPS
```

PatchGAN은 이 단계의 기본 구성에 넣지 않는다. 과거 LuSIR의 v3b/v5 실험에서
고주파 artifact와 fidelity 붕괴가 확인됐기 때문이다.

### Phase 3: degradation-aware conditioning

상태: `[ ] 미구현`

현재 `DegradationPipeline.apply()`는 sampled parameter를 버린다. 기존 호출을
깨지 않도록 optional metadata 반환 경로를 추가한다.

예상 interface:

```python
lr = pipeline.apply(hr, rng=rng, out_size=lr_size)
lr, degradation = pipeline.apply(
    hr,
    rng=rng,
    out_size=lr_size,
    return_metadata=True,
)
```

metadata 후보:

```text
downsample mode
HR/LR blur radius
Gaussian, sensor, chroma, Poisson noise flags/strength
JPEG/WebP quality
ringing, banding, sharpen/oversharpen strength
color jitter/shift
selected preset/domain
```

훈련에서는 known parameter를 normalized vector로 넣고 일부 항목에 dropout과
noise를 적용한다. 실제 입력에서는 다음 두 경로를 비교한다.

1. LR에서 계산한 경량 통계 descriptor.
2. 작은 learned degradation estimator.

처음에는 DASP-SR처럼 blur/noise/JPEG/edge/brightness/contrast의 경량
descriptor부터 시작한다. learned estimator는 이 경로가 유효한 뒤 추가한다.

SANI-style edge-aware noise는 독립 ablation으로 둔다. LR Sobel edge 영역의
noise를 줄이되, train/inference distribution mismatch와 flat-region artifact를
반드시 확인한다.

### Phase 4: optional visual semantic condition와 compact DiT

상태: `[ ] 미구현`

Phase 1-3이 유효할 때만 진행한다.

선택지 A: frozen DINOv3 feature

- LR에서 spatially grounded feature를 추출한다.
- bottleneck 또는 제한된 block에 cross-attention으로 주입한다.
- 구조 condition인 `z_base`는 계속 별도로 유지한다.
- feature cache 또는 small encoder를 사용해 학습 비용을 제한한다.
- 이 경로는 pretrained T2I model을 사용하지 않으므로 vision-only 목표는
  유지하지만, 프로젝트 설명에서 모든 component를 self-trained라고 표현하면
  안 된다.

선택지 B: strict self-trained semantic feature

- Stage1 encoder의 intermediate feature를 활용하거나,
- LuSIR 학습 이미지에 self-supervised vision encoder를 별도 학습한다.

backbone 교체는 objective 검증 뒤에 한다. 첫 DiT는 VOSR의 0.5B/1.4B 규모를
복사하지 않고 현재 U-Net과 비슷한 parameter/FLOP budget으로 맞춘다.

### Phase 5: MeanFlow/MeanSR objective

상태: `[ ] 보류`

일반 residual flow가 유효하고 1-step 성능이 multi-step보다 크게 낮을 때만
검토한다.

- average velocity `u(z_t, r, t, condition)` 학습.
- JVP 메모리와 처리량 측정.
- Stage-Aware Temporal Sampling 독립 ablation.
- DTM은 endpoint metric과 fixed visual review를 모두 통과해야 사용.
- 공식 구현 또는 독립 재현이 성숙하기 전에는 public 기본 경로로 승격하지
  않는다.

## 첫 실험 matrix

모델 구조와 데이터는 고정하고 한 번에 한 요소만 바꾼다.

| ID | Flow | Noise | Detail loss | Degradation cond | 목적 |
| --- | --- | --- | --- | --- | --- |
| `F0` | 없음 | 없음 | 기존 | 없음 | frozen Stage2 baseline |
| `F1` | residual CFM | `0.0` | velocity only | 없음 | formulation sanity |
| `F2` | residual CFM | `0.25` | velocity only | 없음 | low-noise 생성성 |
| `F3` | residual CFM | `0.5` | velocity only | 없음 | 주 후보 |
| `F4` | residual CFM | `1.0` | velocity only | 없음 | artifact 상한 확인 |
| `F5` | best F2-F4 | best | soft weighted | 없음 | detail supervision |
| `F6` | best F5 | best | soft weighted | known/meta token | robustness |
| `F7` | best F6 | best | soft weighted | inferred token | 실제 입력 |

각 ID는 최소 3개 seed의 짧은 probe를 거친 뒤 하나만 장기 run으로 보낸다.

## 승격 조건

새 경로는 다음을 모두 만족해야 research candidate로 승격한다.

1. current Stage2 대비 clean/formal Y PSNR 손실이 평균 `0.10 dB` 이하.
2. LPIPS 또는 DISTS가 clean과 real-degradation review에서 일관되게 개선.
3. highpass energy만 증가하고 GT-aligned highpass error가 악화되는 현상이 없음.
4. text, 얼굴, 건축선, anime line에서 새로운 왜곡이 없음.
5. seed를 바꿨을 때 구조가 변하지 않고, 차이는 plausible texture 범위에 머묾.
6. 1-step이 public 사용에 충분히 빠르고, few-step이 실제 품질 이득을 제공.
7. fixed contact sheet human review를 통과.

Public/Colab 기본값 교체는 formal benchmark와 HF artifact 재현까지 완료된 뒤
별도로 결정한다.

## 즉시 중단 조건

- `sigma_rel`을 낮추면 효과가 사라지고 높이면 과거 residual-shift와 같은
  ripple/grid/artificial texture가 나타남.
- PSNR 손실을 제한한 설정에서 LPIPS/DISTS 및 human review 개선이 없음.
- 여러 seed의 output이 동일해져 stochastic path가 다시 zero residual로 붕괴.
- seed에 따라 글자, 얼굴, 경계 구조가 달라짐.
- weighted detail loss가 missing detail이 아니라 excess energy만 증가시킴.
- degradation token이 clean 성능을 떨어뜨리면서 strong preset에만 과적합.

이 경우 같은 objective의 장기 continuation이나 단순 capacity 확대를 하지
않고 원인 분석 문서를 `docs/TRIAL_AND_ERROR_KO.md`에 추가한다.

## 권장 config 초안

실제 key 이름은 구현 시 기존 config style에 맞춰 확정한다.

```yaml
project:
  name: residual_flow_stage2_anchor_v1_probe

flow:
  objective: conditional_flow_matching
  source: stage2_anchor
  sigma_mode: residual_std
  sigma_rel: 0.5
  timestep_sampling: uniform
  prediction_type: velocity
  ema_rate: 0.999

condition:
  stage2_frozen: true
  domain_embedding: true
  degradation_token:
    enabled: false
  visual_semantic:
    enabled: false

loss:
  velocity_weight: 1.0
  decoded_weight: 0.0
  detail_weight: 0.0
  perceptual_weight: 0.0
  adversarial_weight: 0.0

eval:
  sigma_rel: [0.0, 0.25, 0.5, 1.0]
  steps: [1, 4, 8]
  seeds: [0, 1, 2]
```

## 재개 체크리스트

다음 작업자가 이 계획을 구현하기 전에 확인할 것:

- [ ] `README.md`의 current default와 checkpoint 선택을 다시 확인.
- [ ] `docs/HANDOFF_KO.md`의 최신 완료 실험을 확인.
- [ ] `docs/TRIAL_AND_ERROR_KO.md`에서 residual-shift, wavelet, GAN 실패를 확인.
- [ ] frozen Stage1/Stage2 checkpoint의 SHA256과 HF path를 기록.
- [ ] baseline val100/formal219 결과를 새 output directory에 재생성.
- [ ] Phase 1 외 기능이 모두 꺼진 최소 flow smoke config부터 구현.
- [ ] unit test와 four-step smoke를 통과한 뒤에만 GPU 장기 run 시작.
- [ ] 첫 결과를 이 문서의 `결정 로그`와 trial-and-error 문서에 기록.

## 결정 로그

| 날짜 | 결정 | 근거 | 상태 |
| --- | --- | --- | --- |
| 2026-08-15 | Stage1/Stage2를 보존하고 Stage2-anchored residual flow를 다음 생성형 연구 1순위로 선택 | decoder audit, deterministic smoothing, VOSR/RFMSR/MeanSR 검토 | 제안 |
| 2026-08-15 | MeanSR는 첫 구현이 아니라 Phase 5로 보류 | 매우 최신 프리프린트, 공식 구현 미확인, JVP 복잡도 | 보류 |
| 2026-08-15 | hard mask와 GAN을 Phase 1 기본값에서 제외 | 기존 mask의 작은 이득과 PatchGAN fidelity 붕괴 | 확정 |
| 2026-08-15 | DINOv3는 optional 경로로만 검토 | vision-only 조건에는 맞지만 strict self-trained 표현에는 영향 | 제안 |

새 실험을 실행할 때 이 표에 날짜, config, W&B URL, checkpoint, 최종 판정을
추가한다.

## 다음 대화/VM에서 사용할 짧은 프롬프트

```text
LuSIR 저장소에서 docs/VNEXT_RESIDUAL_FLOW_KO.md를 먼저 읽어줘.
현재 public 기본값과 기존 checkpoint는 바꾸지 말고, 문서의 Phase 0과 Phase 1만
진행해줘. Stage2 output을 source center로 쓰는 conditional residual flow를 현재
ConditionalUNet budget으로 구현하고, sigma_rel 0/0.25/0.5/1.0 및 1/4/8-step을
비교해줘. DINO, degradation token, GAN, MeanSR는 이번 단계에서 넣지 마.
```
