# LuSIR 시행착오 리포트

이 문서는 LuSIR의 직접 학습 x4 latent diffusion SR 실험에서 실패/부분 성공/다음 가설을
계속 누적하기 위한 기록이다. 최종 성능표가 아니라, 왜 다음 실험을 그렇게 잡았는지
추적하는 용도다.

## 2026-06-13 wavelet residual diffusion v2 장기 run 완료

pure-noise v2 step1000 checkpoint를 이어 condition-start 방식으로 step3000까지
학습했다. `start_timestep=50` sampled PSNR은 `22.85 -> 25.03 dB`,
signed wavelet L1은 `0.06111 -> 0.04331`로 계속 개선됐다. 따라서 학습 자체가
붕괴하거나 역행한 것은 아니다.

동일 step3000 checkpoint를 강도별로 다시 평가했다.

| start timestep | PSNR | delta vs v1d | Laplacian gain vs v1d | energy ratio |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 28.1239 | -0.5768 dB | -0.003906 | 0.708 |
| 25 | 27.4752 | -1.2256 dB | -0.007562 | 1.116 |
| 50 | 25.0289 | -3.6719 dB | -0.022338 | 2.503 |

`t=15`는 시각적으로 base와 가까워졌지만 유효한 새 detail보다 약한 stochastic
입자 변화에 가깝다. `t=25/50`은 생성 강도와 diversity가 커지는 대신
GT-aligned high-frequency 오차가 악화했다. 현재 결론은 다음과 같다.

- signed Haar residual과 LL 차단은 구조/색 보존에는 유효하다.
- step3000 시점에는 덜 수렴한 residual noise prediction이 주된 문제였다.
- step20000까지 학습하면 노이즈는 사라지지만 residual/diversity도 함께 줄었다.
- 저주파 과수정은 해결했으나 실제 missing detail 생성에는 실패했다.

장기 run:
`configs/wavelet_residual_diffusion_v2_condition_start_long.yaml`,
<https://wandb.ai/jwheo/LuSIR/runs/zh1fktq4>.
첫 `t=25` 중간 평가인 step4000은 PSNR `27.7734`, v1d 대비
`-0.9274 dB`, signed wavelet L1 `0.02605`로 step3000의
`27.4752 / -1.2256 / 0.02788`보다 개선됐다.

장기 run은 step `20000`, `2500` optimizer update에서 정상 종료했다. 최종
val100은 다음과 같다.

| start timestep | PSNR delta | SSIM delta | Laplacian gain | diversity |
| ---: | ---: | ---: | ---: | ---: |
| 15 | -0.0880 dB | -0.00647 | -0.000833 | 0.00574 |
| 25 | -0.1392 dB | -0.01040 | -0.001229 | 0.00722 |
| 50 | -0.3152 dB | -0.02433 | -0.002516 | 0.01093 |

판단:

- oracle은 유효하지만 현재 condition/noise-MSE 조합은 불확실한 signed residual을
  conditional mean인 zero에 가깝게 예측한다.
- 장기 학습은 grain 제거에는 성공했지만 사용자 체감 detail을 만들지 못했다.
- 모든 강도에서 v1d보다 GT-aligned metric이 낮으므로 승격하지 않는다.
- 동일 objective continuation은 종료한다. 다음은 learned detail mask와
  patch-level perceptual/adversarial supervision을 우선 검토한다.

## 2026-06-13 high-frequency residual diffusion v1 조기 중단

`configs/diffusion_photo130k_lsdir_highfreq_residual_v1_b8.yaml`은 frozen
dual-context Stage2 위에서 zero-init gated residual로 시작했다. 초기 output은
condition과 정확히 같았지만, 학습 후 다음처럼 일관되게 역행했다.

| step | PSNR delta vs condition | residual L1 | gate mean |
| ---: | ---: | ---: | ---: |
| 500 | `-0.002 dB` | `0.0163` | `0.203` |
| 1000 | `-0.017 dB` | `0.0482` | `0.256` |
| 1500 | `-0.037 dB` | `0.0688` | `0.292` |

fixed sample의 Laplacian energy는 `0.005923 -> 0.007660`으로 증가했지만,
GT와의 Laplacian L1 오차도 `0.009706 -> 0.010581`로 악화했다. 즉 유효한
detail보다 반복 패턴과 고주파 에너지만 추가했다. run
<https://wandb.ai/jwheo/LuSIR/runs/q3t4hzms>은 step `1650` 부근에서 중단했다.

다음 시도는 magnitude loss 가중치 조정이 아니라 target residual을 직접
노이즈화하고 noise를 예측하는 residual-space diffusion 구조로 바꾼다.

구현된 v2는 `GT - detail v1d`의 signed Haar LH/HL/HH 대역만 diffusion한다.
LL 대역은 출력할 수 없다. val8에서 동일한 ±0.16 clipping을 적용한 oracle은
v1d 평균 PSNR을 `28.7012 -> 31.4039`, Laplacian L1을
`0.018342 -> 0.010721`로 개선해 표현 공간의 가능성을 확인했다.

첫 v2 pure-noise sampler는 step1000에서 residual energy가 target의 약 10배로
남아 중단했다. 동일 checkpoint를 `start_timestep=50` condition-start로
평가하면 sampled PSNR이 약 `18 dB -> 22.85 dB`, lowpass drift가
`0.006대 -> 0.00157`로 개선됐다. 다음 probe는 이 checkpoint에서 이어
train timestep을 `0..75`로 제한한다.

## 2026-06-13 정식 full-image x4 benchmark

five-crop RGB PSNR 진단만으로는 공개 SR 결과와 비교할 수 없어서 DIV2K
validation, Set5, Set14, Urban100의 공식 x4 pair를 복구하고 정식 evaluator를
추가했다. 평가는 MATLAB-compatible BT.601 Y, scale=4 border shave,
MATLAB-style SSIM을 사용하며, candidate 크기가 HR과 다르면 자동 resize 없이
실패한다.

| path | DIV2K | Set5 | Set14 | Urban100 |
| --- | ---: | ---: | ---: | ---: |
| dual-context base | 29.9575 | 31.6621 | 28.2441 | 25.4816 |
| detail v1d | **30.1602** | **31.8892** | **28.4123** | **25.8755** |
| v1d gain | +0.2027 | +0.2271 | +0.1682 | +0.3939 |

표는 Y PSNR이다. V1d는 네 dataset 모두에서 base의 PSNR과 SSIM을 개선했고,
texture-heavy Urban100에서 가장 큰 이득을 냈다. 따라서 v1d 재설계와 3-epoch
학습은 정식 protocol에서도 의미가 있었다.

같은 evaluator에서 RealESRNet/RealESRGAN보다 clean fidelity가 높았지만, 두
모델은 real-world/perceptual 목적이라 이것을 SOTA 주장으로 해석하면 안 된다.
다음 병목 판단에는 SwinIR 같은 classical baseline, LPIPS/DISTS,
real-degradation 평가, blind human review가 필요하다.

공식 SwinIR classical x4 checkpoint를 DIV2K validation에 실행하고 같은
evaluator로 다시 계산한 결과는 `31.0838 / 0.85228`이다. V1d보다 Y PSNR
`+0.9235 dB`, Y SSIM `+0.01807` 높다. 따라서 detail branch는 제 역할을
하지만, 다음 clean-fidelity 병목은 branch 용량이 아니라 Stage2/base
reconstruction 경로다.

## 2026-06-13 Stage2 clean-fidelity LR 실험과 목표 분리

clean-bicubic fidelity continuation은 원래 LR `5e-6`에서 다음처럼 완만하게
올랐다.

```text
step 1000:  eval/decoded_psnr 25.019
step 4000:  eval/decoded_psnr 25.031
step 9000:  eval/decoded_psnr 25.045
step 15000: eval/decoded_psnr 25.057, current best
step 17000: eval/decoded_psnr 25.054
step 17825: 수동 종료, best 갱신 없음
```

이 값은 task-specific val100 학습 proxy이며 정식 full-image Y-channel
benchmark와 직접 비교하지 않는다. 정식 DIV2K 비교는 LuSIR detail v1d
`30.1602`, SwinIR `31.0838`, gap `0.9235 dB`다.

LR이 너무 낮아서 plateau한 것인지 확인하기 위해 원본 step15000을 보존하고
별도 실험을 돌렸다.

| experiment | result | 판단 |
| --- | ---: | --- |
| LR `20x` continuation | first eval step15500 `15.72 dB` | 즉시 붕괴, 폐기 |
| LR `5x` continuation | step15500 `24.963`, step16000 `25.012` | 원본보다 낮음 |
| LR `5x` from-init | step4000 `25.033` | 원래 LR step4000 `25.031`과 동률 |

결론:

- LR 증가는 clean-fidelity gap이나 visible detail 병목을 바꾸지 못했다.
- 같은 objective에서 `+0.01 dB`씩 움직이는 것은 모델이 현재 구조/손실의
  근처 최적점에 들어갔다는 신호다.
- deterministic base는 fidelity 경로로 보존한다.
- 다음 별도 학습은 전체 x0를 다시 그리는 Stage4 continuation이 아니라,
  frozen base 위에 bounded/gated high-frequency residual만 생성하는 stochastic
  diffusion 경로로 분리한다.
- 성공 판단은 PSNR 단독이 아니라 LPIPS/DISTS, fixed visual review,
  high-frequency metric, lowpass drift, seed diversity를 함께 사용한다.

설계 메모:
`docs/HIGH_FREQUENCY_RESIDUAL_DIFFUSION_KO.md`.

## 2026-06-13 Stage2 clean-bicubic fidelity continuation 준비

SwinIR gap을 줄이기 위해 새 Stage2 continuation을 별도 config로 분리했다.
목표는 real-world denoise가 아니라 공식 x4 bicubic LR pair와 같은 조건에서
Stage2/base reconstruction 자체를 끌어올리는 것이다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue.yaml
init:   checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
data:   manifest_photo130k_lsdir.csv, degradation_preset=benchmark_bicubic
loss:   decoded_weight 1.5, edge 0.5, highpass 0.75, highpass_magnitude 0.25
train:  batch 8, grad_accum 4, max 60000 micro steps
eval:   val100 every 1000 micro steps, run_at_start=false
```

같이 고친 점:

- `train_latent_pretrain.py`가 `hflip_prob`, `texture_crop_retries`,
  `hr_color_jitter_*`를 `ManifestImageDataset`으로 전달하도록 수정했다.
  Dataset에는 이미 구현돼 있었지만 Stage2 train helper가 쓰지 않고 있었다.
- `run_sr_benchmark.py`에 `stage2_base` variant를 추가했다. 새 Stage2
  checkpoint가 나오면 detail branch 없이 바로 full-image benchmark에 넣을 수
  있다.
- smoke: best98000 checkpoint를 `stage2_base` variant로 Set5 한 장 tiled
  inference했고 정상 출력됐다.
- 첫 launch에서 run-at-start val100 eval이 4분 이상 step 1을 잡아 학습을
  시작하지 못했다. 해당 run은 끊고 `eval.every=1000`, `run_at_start=false`로
  조정했다.
- 당시 초기 launch 기록:
  - tmux `stage2-bicubic-fidelity`
  - W&B <https://wandb.ai/jwheo/LuSIR/runs/6cvkm4cc>
  - log
    `/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue/train_console.log`
  - initial interrupted step1 val100 decoded PSNR `24.9600`
  - restarted run reached step `125` at about `1.15 micro-step/s`, GPU util
    `100%`, VRAM `37.8/46.1GB`

이후 continuation은 원래 LR에서 step `15000` val100 proxy `25.057`까지
올랐지만 plateau했고, LR probe도 병목을 바꾸지 못했다. 현재 판단과 다음
stochastic residual diffusion 방향은 바로 위
`Stage2 clean-fidelity LR 실험과 목표 분리` 항목에 기록한다.

## 2026-06-13 detail branch v1d 3 epoch 완료

V1b는 안정적이지만 visible residual이 너무 작았다. 같은 branch를 단순히 더
오래 학습하는 대신, v1c에서는 frozen Stage 2 condition latent를 직접 입력하고
gate/residual 범위를 조금 열었다.

```text
v1c selected step: 6000
photo_detail_mix PSNR delta: +0.0554 dB
SSIM delta:                 +0.00332
wins:                       99/100
```

V1c가 빠르게 plateau한 뒤, 용량 자체가 병목인지 확인하기 위해 v1d를 만들었다.
V1d는 width/objective를 유지한 채 residual block을 `8 -> 18`, branch
파라미터를 `1.35M -> 3.02M`으로 늘린다. 새 block은 identity-init하여 시작
출력을 v1c와 동일하게 유지했다.

```text
config: configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
completed: 100086 micro-steps = exactly 3 epoch
selected: step 99500
W&B:    https://wandb.ai/jwheo/LuSIR/runs/ctg4r7n9
```

선택 step `99500`은 ordinary fixed `photo_detail_mix`에서 aggregate PSNR
`+0.1646 dB`, mean PSNR `+0.1888 dB`, SSIM `+0.00647`, wins `99/100`,
detail wins `100/100`이다. strict-bicubic DIV2K five-crop에서는 `31.9513 dB`,
base 대비 `+0.2102 dB`, v1c 대비 `+0.1358 dB`, `5/5` wins를 기록했다.
선택 checkpoint는 단일 이미지/tiled inference runner와 Colab WebUI의
`Latest detail research - Detail Branch v1d` 옵션으로 노출했다. public 기본값은
T4와 사용자 안정성을 위해 residual refiner v2를 유지한다.

final step `100086`의 strict-bicubic mean PSNR은 `31.9516 dB`로 선택
checkpoint와 사실상 같지만, ordinary val aggregate PSNR/SSIM/highpass/detail
score는 step `99500`이 더 좋다. 따라서 best99500을 공식 선택하고 동일
objective continuation은 종료한다.

시각적으로 흰 점, grid, 과도한 sharpening은 없고 early v1d보다 residual이
강해졌지만 여전히 GT fine texture에는 못 미친다. 결론은 “용량/장기 학습이
무의미했다”가 아니라 “안정적 수치 개선은 만들었지만 perceptual-detail
돌파는 아니었다”이다.

## 2026-06-13 strict-bicubic 모델 규모 비교

기존 validation preset은 blur/noise/compression 등이 섞여 있어 공개
bicubic-only SR 논문의 PSNR과 직접 비교할 수 없었다. 또한 기존 `clean`
preset도 downsample kernel을 bicubic/bilinear/lanczos 중 무작위로 골랐다.
따라서 추가 훼손 없이 PIL bicubic x4만 쓰는 `benchmark_bicubic` preset을
추가하고 DIV2K val `0801-0805` center crop 5장으로 진단했다.

| path | loaded params | RGB PSNR |
| --- | ---: | ---: |
| bicubic | n/a | 29.5999 |
| Stage2 XL condition-only | 40.040M | 30.5677 |
| Stage2 multiscale | 76.591M | 31.6068 |
| Stage2 dual-context | 140.334M | 31.7411 |
| dual + detail v1c | 141.689M | 31.8154 |
| dual + detail v1d step99500 | 143.354M | 31.9513 |
| Stage4 XL edge sampled | 509.658M | 29.5487 |

판단:

- LuSIR의 `24 dB`대 ordinary 수치는 강한 degradation 조건의 난이도를 크게
  반영한다. clean bicubic에서는 deterministic 경로가 `31~32 dB`에 도달한다.
- Stage2 XL에서 multiscale/dual-context로 확장한 것은 실제 reconstruction
  이득을 만들었다.
- 509.658M Stage4 XL은 clean input을 과수정해 bicubic보다도 낮다. 파라미터
  수가 곧 품질 순위는 아니다.
- v1d는 v1c보다 `+0.1358 dB` 좋아져 3 epoch 장기 학습은 의미가 있었다.
  다만 시각적 변화는 여전히 보수적이므로 capacity만 더 늘리는 것은 다음
  우선순위가 아니다.
- 이 결과는 RGB, 5 center crops, PIL bicubic, border shave 없음 조건이므로
  정식 SOTA benchmark 주장이 아니다.

## 2026-06-11 detail branch v1 구현

배경:

- Stage 2 dual-context LSDIR scale-up은 `+0.10 dB`대의 수치 개선을 만들었지만
  사용자가 지적한 missing fine detail/뭉개짐은 계속 남았다.
- VGG feature continuation은 `+0.01~0.03 dB` 수준의 개선에 그쳤고 fixed
  sample에서 눈으로 거의 구분되지 않았다.
- residual refiner v2는 안전하지만 visible texture 생성량이 작다.

따라서 또 하나의 full-image/latent continuation 대신, decoded base SR 위에
고주파 residual만 더하는 deterministic detail branch를 구현했다.

```text
config: configs/detail_branch_v1_photo130k_lsdir.yaml
train:  tools/train/train_detail_branch.py
eval:   tools/eval/run_fixed_review_detail_branch.py
base:   Stage 2 dual-context LSDIR best98000 + Stage 1 decoder, frozen
path:   LR -> Stage2 -> Stage1 decoder -> base SR -> image-space detail branch
```

구조상 output conv는 zero-init이다. step 0은 base SR과 정확히 같아야 하고,
branch는 `highpass_project(residual)`과 gate를 통해 색/저주파 구조 변경 경로를
제한한다.

Smoke:

```text
4 micro-steps = 1 optimizer update
eval/base_psnr step0: 24.6188
eval/sr_psnr step0:   24.6188
step4 sr_vs_base_psnr: +0.00005 dB
step4 wins_vs_base:    69/100
```

판단:

- smoke는 품질 주장이 아니라 load/eval/backprop/update/checkpoint 경로 확인이다.
- 장기 run은 `40000` micro-steps = `10000` optimizer updates로 시작한다.
- 승격 여부는 `detail_v1` fixed review set에서 residual refiner v2 baseline과
  contact sheet/HTML/LPIPS/DISTS까지 보고 판단한다.

### 2026-06-11 detail branch v1 조기 중단 및 v1b augmentation 전환

v1 장기 run은 초반 수치가 올라갔지만 residual이 매우 보수적으로 작고 시각적
detail 생성량이 아직 약했다. train set이 `133450`장, batch `4`이므로 중단 시점
`7800` micro-steps는:

```text
7800 * 4 / 133450 = 0.234 epoch
```

즉 아직 1 epoch의 1/4도 지나지 않았다. 회전/affine처럼 SR alignment나 현실감을
깨는 augmentation은 넣지 않고, 다음 안전한 증강만 추가한 v1b로 재시작한다.

```text
config: configs/detail_branch_v1b_aug_photo130k_lsdir.yaml
changes:
  hflip_prob: 0.5
  texture_crop_retries: 4
  texture_crop_downsample: 128
  hr_color_jitter_prob: 0.25
  hr_color_jitter: [0.97, 1.03]
excluded:
  rotation, vertical flip, affine/perspective, random erasing, mixup
```

목표는 모델/optimizer/loss를 바꾸지 않고 데이터 노출만 바꿔서, detail branch가
하늘/벽 같은 smooth crop보다 털, 잎, 직물, 글자, 표면 질감이 있는 crop을 더 자주
보게 하는 것이다.

### 2026-06-11 detail branch v1b 완료

v1b augmentation run은 `40000` micro-steps에서 정상 종료됐다.
`grad_accum_steps: 4` 기준 `10000` optimizer updates이며, train `133450`장 기준
약 `1.199 epoch`다.

```text
run:    detail_branch_v1b_aug_photo130k_lsdir
config: configs/detail_branch_v1b_aug_photo130k_lsdir.yaml
W&B:    https://wandb.ai/jwheo/LuSIR/runs/1o3aavi9
local:
  /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1b_aug_photo130k_lsdir
best:
  checkpoints/best_eval_detail.pt
```

val100 결과:

| selection | step | PSNR delta | SSIM delta | mean delta | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| best detail score | 39500 | +0.0461 dB | +0.00268 | +0.0575 | 98/100 |
| best PSNR delta | 38500 | +0.0489 dB | +0.00317 | +0.0531 | 96/100 |
| best SSIM delta | 37000 | +0.0438 dB | +0.00336 | +0.0460 | 95/100 |
| final | 40000 | +0.0444 dB | +0.00277 | +0.0529 | 98/100 |

이전 v1 조기 중단 run의 최고가 PSNR `+0.0230 dB`, SSIM `+0.00239`였으므로,
augmentation + 장기 학습은 수치상 진전이 있었다. 특히 wins가 final/selected에서
`98/100`까지 유지되고, detail score best 기준 detail wins가 `100/100`으로 나온다.

눈으로 본 판단:

- 변화는 안정적이고 artifact-light다.
- 라임 표면, 털, 풀잎, 원거리 건물 edge에서 얇은 detail 보강은 보인다.
- 하지만 base와 detail 차이가 작고, GT의 실제 미세 질감에는 아직 크게 못 미친다.
- SSIM 개선은 `+0.002~+0.003`대라 아쉽지만, 현재 branch가 작은 high-frequency
  residual만 허용하기 때문에 큰 SSIM 도약을 기대하기 어렵다.
- SSIM만 키우는 방향은 다시 smoothing을 보상할 수 있으므로 주의가 필요하다.

결론:

- v1b step `39500`을 이전 비교용 public detail artifact로 보존한다.
- final `40000`이 아니라 `best_eval_detail.pt`를 기준으로 문서/HF/리뷰를 맞춘다.
- public Colab 기본값은 아직 residual refiner v2다. detail branch v1d는 비교용
  WebUI 옵션으로 제공하며, 사용자 기본 경로 승격은 정식 benchmark와 human
  review 이후에 판단한다.
- 다음 ablation은 무작정 더 길게 돌리기보다 약한 SSIM/MS-SSIM loss, gate/residual
  개방, 또는 degradation-aware detail gate를 비교하는 편이 낫다.

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
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/nog04fwr>

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
python tools/eval/eval_condition_samples.py \
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
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/xyvqg0n6>

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

- W&B: <https://wandb.ai/jwheo/LuSIR/runs/lrb6nco9>
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

- W&B: <https://wandb.ai/jwheo/LuSIR/runs/edfko8e8>
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

- `tools/analysis/diagnose_stage2_residuals.py`

실행:

```bash
python tools/analysis/diagnose_stage2_residuals.py \
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

- `tools/train/train_residual_refiner.py`
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

- `tools/eval/eval_residual_refiner.py`
- `tools/infer/infer_residual_refiner.py`

실행:

```bash
python tools/eval/eval_residual_refiner.py \
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
  - step 0-2000: <https://wandb.ai/jwheo/LuSIR/runs/6h0124us>
  - step 2000-8000: <https://wandb.ai/jwheo/LuSIR/runs/0p3lfqt7>
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
- `tools/analysis/analyze_degradation_presets.py`
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
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/so0lbyte>
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
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/3v6wmf5o>
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
W&B: https://wandb.ai/jwheo/LuSIR/runs/hgr8ilhk
```

초기 실행은 L40S 한 장에서 VRAM 약 `32.9/46.1GB`, GPU util `100%`,
steady 약 `1.31 micro-step/s`였다. 판정 기준은 decoded PSNR만이 아니라
Laplacian detail ratio와 고정 샘플의 실제 질감 복원이다. 이 probe에서
Condition 선명도가 움직이지 않으면 단일 해상도 residual CNN을 멀티스케일
Stage 2 구조로 교체한다.

### Probe 결과와 구조 교체 결정

4000 step까지 고정 샘플을 확인했지만 사용자 체감 선명도는 좋아지지 않았다.
decoded PSNR은 소폭 올랐으나, 3500/4000 step의 Laplacian detail ratio는
초기값보다 오히려 낮았다.

```text
step 1:    decoded PSNR 23.7387, detail ratio 0.28167
step 2500: decoded PSNR 24.1447, detail ratio 0.31115
step 4000: decoded PSNR 24.2792, detail ratio 0.27731
stop:      decoded PSNR 24.2941, detail ratio 0.28655
```

결론:

- 동일한 단일 해상도 trunk에 decoded/detail loss만 추가하면 distortion 수치는
  개선돼도 실제 detail 복원은 안정적으로 좋아지지 않는다.
- 기존 manifest의 train 103,450장 중 COCO가 100,000장이고,
  DIV2K/Flickr2K는 3,450장뿐이라 고품질 복원 신호가 지나치게 희석된다.
- loss weight 추가 실험을 계속하기보다 Stage 2의 문맥 범위와 데이터 노출 비율을
  동시에 바꾼다.

## 2026-06-07 Stage 2 multiscale-context + HQ-balanced long run

공개 복원 연구의 공통 방향을 참고했다.

- [SwinIR](https://arxiv.org/abs/2108.10257), [HAT](https://arxiv.org/abs/2309.05239):
  SR에서 넓은 문맥과 장거리 상호작용의 중요성.
- [NAFNet](https://arxiv.org/abs/2204.04676): 복원 모델에서 효율적인
  multi-scale encoder-decoder 경로의 유효성.
- [Real-ESRGAN](https://arxiv.org/abs/2107.10833): 실제 복원 성능에서
  degradation/data 구성의 중요성.

기존 19M flat Stage 2 trunk의 이름과 가중치는 보존하고, 128 -> 64 -> 32
해상도의 multiscale-context 분기를 추가했다. 마지막 context projection을
zero-init하여 step 72000 체크포인트를 partial-init하면 최초 출력이 기존 모델과
정확히 같고, 이후에만 넓은 문맥을 학습한다.

데이터는 새 이미지를 무작정 추가하는 대신 기존 고품질 원본의 학습 노출을 먼저
교정했다. `scripts/build_hq_mix_manifest.py --hq-repeat 30`으로 train manifest를
다음처럼 구성했다.

```text
train rows:       203,500
COCO rows:        100,000
DIV2K/Flickr2K:   103,500
HQ train ratio:    50.86%
```

장기 run 설정:

```text
config: configs/latent_pretrain_photo100k_multiscale_hqmix_long.yaml
model params: 55.50M
degradation: photo_detail_mix
loss: latent 0.25 + decoded 1.0 + edge 1.0 + highpass 2.0
      + highpass residual magnitude 1.0
batch: 8, grad accumulation: 4, effective batch: 32
max micro-steps: 50,000
```

L40S smoke 결과 batch 8 forward/backward와 val100 eval이 정상 통과했다.
GPU util은 `100%`, VRAM은 약 `34.8/46.1GB`였고 batch 4와 micro-step 속도가
거의 같아 batch 8을 선택했다. 전체 테스트는 `25 passed`.

장기 run:

```text
W&B: https://wandb.ai/jwheo/LuSIR/runs/6zt2do4v
initial val100: decoded PSNR 23.7387, detail ratio 0.28167
steady: about 1.25 micro-step/s, GPU util 100%, VRAM 34.8/46.1GB
```

### 50k 완료 결과

학습은 `50000` micro step까지 정상 완료됐다. train eval의 global decoded PSNR은
`23.7387 -> 24.4870`으로 `+0.7482 dB` 상승했지만, 마지막 Laplacian detail
ratio는 `0.2817 -> 0.2783`으로 감소했다. 고정 샘플에서도 큰 윤곽, 색, 노이즈
정리는 개선됐지만 털, 과일 표면, 글자, 천, 먼 구조의 실제 고주파는 복원하지
못했다.

step 41000/46000/50000을 기존 Stage 2 XL step 72000과 같은 val100에서 직접
비교한 결과 step 46000을 선택했다. step 50000은 PSNR 이득이 거의 없고
clean/mild detail ratio가 다시 하락했다.

| degradation | 기존 PSNR | step 46000 PSNR | delta / wins | 기존 detail | step 46000 detail |
| --- | ---: | ---: | ---: | ---: | ---: |
| `photo_detail_mix` | 25.3103 | 26.3450 | +1.0348 / 99 | 0.2891 | 0.2977 |
| `mild` | 25.0449 | 25.9678 | +0.9228 / 97 | 0.2864 | 0.2878 |
| `photo_v2` | 22.9271 | 23.8712 | +0.9441 / 81 | 0.2876 | 0.2210 |
| `photo_v3_noise_mix` | 22.9014 | 23.8664 | +0.9650 / 81 | 0.2992 | 0.2257 |

판단:

- multiscale context + HQ-balanced data는 clean/mild base reconstruction과
  denoising 정확도에는 명확히 성공했다.
- 그러나 강한 degradation에서는 실제 detail까지 노이즈로 판단해 기존보다 더
  많이 제거한다. 사용자 체감상 smoothing 문제는 해결되지 않았다.
- 따라서 step 46000은 새 condition 후보로 보존하지만, perceptual detail 목표의
  완성 모델로 취급하지 않는다.
- 다음 구조 변경은 deterministic regression loss를 더 조정하는 수준이 아니라
  perceptual/feature-space supervision 또는 별도 detail synthesis 경로가 필요하다.

```text
selected checkpoint:
  checkpoints/stage2_photo100k_multiscale_hqmix_step_0046000.pt
metrics:
  metrics/stage2_multiscale_hqmix_step46000_cross_preset_summary.json
W&B:
  https://wandb.ai/jwheo/LuSIR/runs/6zt2do4v
```

### 다음 continuation 준비: frozen VGG feature supervision

같은 pixel/edge/highpass weight 조정만 반복하지 않고, ImageNet pretrained
VGG16의 얕은/중간 feature를 비교하는 optional perceptual loss를 구현했다.
pretrained T2I나 생성 모델은 사용하지 않지만, 외부 pretrained vision feature
supervision을 도입하는 실험이라는 점은 명시한다.

```text
config:
  configs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue.yaml
initialization:
  selected multiscale Stage 2 step 46000
feature layers:
  VGG16 indices 3 / 8 / 15, resized to 256
batch:
  4 x grad_accum 8 = effective 32
lr:
  5e-6
max:
  12000 micro steps
selection:
  eval/decoded_psnr + 5 * eval/laplacian_energy_ratio
```

CUDA smoke:

- batch 4 forward/backward와 val100 eval 정상 통과.
- VRAM 약 `20.6/46.1GB`, GPU util `99~100%`.
- steady 약 `2.62 micro-step/s`.
- perceptual loss는 초기 약 `0.0394`이며 전체 loss를 압도하지 않았다.
- 장기 학습 시작:
  - W&B: <https://wandb.ai/jwheo/LuSIR/runs/nrqhw05u>
  - 초기 val100: decoded PSNR `24.4835`, detail ratio `0.2907`,
    perceptual `0.03218`, PSNR-detail score `25.937`.
  - steady 약 `2.62 micro-step/s`, GPU util `99~100%`, VRAM 약 `20.6GB`.

### 2026-06-08 perceptual continuation 판단 프로토콜

Stage 번호는 학습 순서이며 실제 추론 직렬 경로가 아니다.

```text
Colab 기본 deterministic:
  LR -> Stage2 XL step72000 -> residual refiner v2 step39000 -> Stage1 decoder

generative 비교:
  LR -> Stage2 -> Stage3 또는 Stage4 중 하나 -> Stage1 decoder
```

현재 perceptual continuation은 public Colab 기본 모델을 즉시 교체하는 학습이
아니라, multiscale Stage2 step46000을 새 condition 후보로 검증하는 실험이다.

- step `2500`: decoded PSNR `24.50`, detail ratio `0.292`, perceptual
  `0.03148`, shortlist score `25.966`.
- step `3000`: decoded PSNR `24.49`, detail ratio `0.301`, perceptual
  `0.03144`, shortlist score `26.000`.
- 초기 score `25.937`보다 상승했지만 아직 승격 근거로는 부족하다.

최종 승격 조건:

1. `photo_detail_mix`와 `mild`에서 초기 step46000 대비 PSNR/detail이 함께
   유지 또는 개선될 것.
2. `photo_v2`와 `photo_v3_noise_mix`에서 기존 detail collapse와
   cyan/white artifact가 악화되지 않을 것.
3. LPIPS/DISTS 계열 perceptual metric과 고정 sample blind A/B가 개선될 것.
4. `decoded_psnr + 5 * detail_ratio`는 shortlist에만 사용한다. 이 score는
   실제 detail이 아니라 인공 고주파/노이즈 증가에도 상승할 수 있다.
5. 동일 objective의 장기 continuation은 위 조건을 만족하는 중간 checkpoint가
   있을 때만 정당화한다.

### 2026-06-08 perceptual continuation 완료 및 판단

학습은 계획대로 `12000` micro steps에서 정상 종료됐다. 자동
`PSNR + 5 * detail ratio` 기준 best는 step `8000`이다.

| checkpoint | photo_detail delta | mild delta | photo_v2 delta | photo_v3 delta |
| --- | ---: | ---: | ---: | ---: |
| step 8000 | +0.0101 dB | +0.0121 dB | +0.0136 dB | +0.0256 dB |
| step 11000 | +0.0244 dB | +0.0243 dB | +0.0255 dB | -0.0063 dB |
| step 12000 | +0.0180 dB | +0.0170 dB | +0.0208 dB | +0.0107 dB |

`photo_detail_mix` 학습 eval:

| checkpoint | decoded PSNR | detail ratio | VGG perceptual | shortlist score |
| --- | ---: | ---: | ---: | ---: |
| 초기 step 46000 | 24.4835 | 0.2907 | 0.03218 | 25.9370 |
| step 8000 | 24.4936 | 0.3031 | 0.03134 | 26.0092 |
| step 11000 | 24.5080 | 0.2976 | 0.03127 | 25.9962 |
| step 12000 | 24.5015 | 0.2998 | 0.03125 | 26.0003 |

시각 판단:

- step 8000/11000/12000과 초기 step46000의 차이를 고정 contact sheet에서
  거의 구분하기 어렵다.
- 라임 표면, 털, 잎, 셔츠 패턴, 원거리 구조의 missing detail은 복구되지 않았다.
- 강한 입력에서 새 artifact가 크게 증가하지는 않았지만 smoothing도 유지됐다.

결론:

- frozen VGG supervision은 latent MSE, PSNR, VGG metric을 작게 개선한
  부분 성공이다.
- 사용자 체감 fine-detail 목표는 달성하지 못했으므로 public/default Stage2로
  승격하지 않는다.
- step `8000`은 네 preset에서 모두 후퇴하지 않은 가장 안전한 실험 후보로
  보존한다. step `11000`은 clean/mild PSNR 후보지만 strong preset 후퇴 때문에
  기본 후보로 선택하지 않는다.
- 다음에는 같은 Stage2 regression continuation보다 별도 detail synthesis 경로,
  degradation-aware high-frequency branch, 또는 실제 perceptual/preference
  objective를 검토해야 한다.

### 2026-06-08 `+0.01 dB` 판단과 unique-data dual-context 장기 run

perceptual continuation의 cross-preset 개선은 `+0.0101~+0.0256 dB`였지만
고정 sample에서 거의 구분되지 않았다. 이 정도 변화는 평가 노이즈와 checkpoint
선택 편차에 가깝고, 사용자 체감 업스케일 품질 개선으로 해석하지 않는다.

데이터 점검 결과:

- HQ-balanced manifest: `203600` rows.
- 실제 고유 이미지: `103550`장.
- COCO `100000`, DIV2K `900`, Flickr2K `2650` 고유 이미지이며 HQ 데이터는
  반복 노출로만 비중을 높였다.
- 다음 run은 LSDIR 고유 이미지 `30000`장을 받아 최종
  `133450` unique train + `100` val로 구성한다.

구조 변경:

- selected multiscale Stage2 step `46000`의 55.50M 파라미터를 모두 유지한다.
- 두 번째 multiscale context branch를 추가해 총 `119238352` 파라미터로 늘린다.
- 새 branch 출력 convolution은 zero-init이므로 partial init 직후 출력은 기존
  selected checkpoint와 정확히 같다.
- 기존 VGG perceptual objective는 시각 효과가 없었으므로 다시 사용하지 않고,
  decoded/edge/highpass objective를 유지한다.

CUDA smoke:

- config:
  `configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml`
- batch `8`, grad accumulation `4`, effective batch `32`.
- 초기 val100 decoded PSNR `24.48`, detail ratio `0.291`.
- L40S VRAM 약 `37.8/46.1GB`, GPU util `99%`, 약 `0.75 micro-step/s`.
- max `100000` micro steps = `25000` optimizer updates.
- 일반 checkpoint는 디스크 보호를 위해 `5000` micro-step마다 저장한다.
- raw LSDIR 데이터는 GitHub/HF에 올리지 않는다.
- 장기 학습 시작:
  - tmux: `stage2-lsdir-dual`
  - W&B: <https://wandb.ai/jwheo/LuSIR/runs/4akqckxu>
  - log:
    `/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_multiscale_long/train.log`
  - 초기 eval 이후 step 50~75 steady 약 `1.15 micro-step/s`, GPU `100%`,
    VRAM `37.8/46.1GB`, 약 `306W`, `58°C`.

### 2026-06-09 dual-context LSDIR 장기 run 완료 판단

학습은 `100000` micro steps에서 정상 종료됐고 W&B sync도 완료됐다.

같은 compare tool 기준 selected multiscale step46000 대비:

| preset | old step46000 | best step98000 | delta | final step100000 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `photo_detail_mix` | 24.4835 | 24.6197 | +0.1362 | 24.6091 | +0.1256 |
| `mild` | 24.2496 | 24.3583 | +0.1086 | 24.3522 | +0.1025 |
| `photo_v2` | 22.7186 | 22.7726 | +0.0540 | 22.7854 | +0.0668 |
| `photo_v3_noise_mix` | 22.4401 | 22.4044 | -0.0356 | 22.4533 | +0.0132 |

고정 sample grid 통계:

- best step98000은 initial 대비 PSNR `+0.1055 dB`, Laplacian ratio
  `+0.0140`.
- final step100000은 initial 대비 Laplacian ratio `+0.0284`로 더 높지만
  PSNR은 best보다 낮다.
- 흰색 artifact/클리핑 증가는 샘플 통계상 보이지 않았다.

판단:

- 30k LSDIR 추가와 119M dual-context 확장은 `+0.01 dB` 수준의 이전
  perceptual continuation보다 분명한 수치 개선을 만들었다.
- 다만 개선폭은 clean/mild에서 `+0.10~0.14 dB`, strong preset에서
  `+0.01~0.07 dB` 수준이다. 사용자 체감 fine-detail 복원 돌파로 보기는
  어렵다.
- step98000은 clean/mild용 자동 best, step100000은 strong preset tail에
  조금 더 안전한 후보로 보존한다.
- public/default 승격 전에는 contact sheet human review가 필요하다.

### 2026-06-13 detail-need mask target/proxy 진단

Residual diffusion v2가 zero residual로 수렴한 뒤, 전체 이미지에 detail을
생성하는 대신 실제로 detail이 부족한 위치를 먼저 찾는 실험을 시작했다.

구현:

- `src/sr_diffusion/detail_mask.py`
- `tools/analysis/diagnose_detail_need_mask.py`
- `docs/DETAIL_NEED_MASK_KO.md`

GT target은 GT high-frequency magnitude가 base보다 큰 missing-detail만 사용한다.
base high-frequency가 GT보다 큰 excess-detail은 별도로 측정해 생성 target에서
제외한다.

photo-detail val100:

| selector | top20 missing capture | concentration | excess capture |
| --- | ---: | ---: | ---: |
| GT target | `0.4878` | `2.4389x` | `0.3796` |
| highpass disagreement proxy | `0.3252` | `1.6262x` | `0.4838` |
| base/bicubic gap proxy | `0.3201` | `1.6005x` | `0.4762` |

판단:

- GT target은 무작위 top20 대비 missing-detail을 충분히 집중한다.
- grid에서도 라임 껍질, 털, 잎맥, 의류 무늬 등 실제 누락 texture를 찾는다.
- hand-crafted proxy는 target과 상관이 있지만 excess-detail도 많이 선택하므로
  그대로 gate로 쓰지 않는다.
- 다음은 작은 learned mask predictor가 correlation `0.5403`, top20 capture
  `0.3252`, excess capture `0.4838` baseline을 넘는지 짧게 검증한다.
- predictor가 실패하면 masked generator나 adversarial 장기 학습을 시작하지 않는다.
