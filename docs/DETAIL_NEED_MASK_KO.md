# Detail-Need Mask 진단

## 목적

기존 deterministic detail branch는 전체 이미지에서 보수적인 residual을 예측했고,
residual diffusion은 장기 학습 시 residual과 seed diversity가 함께 zero 쪽으로
수렴했다. 다음 실험은 fidelity base를 유지하면서 실제로 detail이 부족한 위치만
선택해야 한다.

첫 단계에서는 mask predictor를 바로 학습하지 않는다. GT를 사용할 수 있는 학습
샘플에서 `어디에 detail이 추가로 필요한가`라는 target이 의미 있는지 먼저
측정한다.

## 현재 target 정의

`src/sr_diffusion/detail_mask.py`의 target은 다음 원칙을 따른다.

- GT와 base를 같은 highpass filter로 분해한다.
- GT high-frequency magnitude가 base보다 큰 위치만 `missing detail`로 본다.
- base의 high-frequency가 GT보다 큰 위치는 `excess detail`로 분리한다.
- missing magnitude와 GT/base highpass mismatch를 patch 단위로 결합한다.
- image별 95% quantile로 정규화해 `detail-need score`를 만든다.

따라서 단순히 edge가 있거나 base가 GT와 다르다는 이유만으로 detail 추가 target이
되지 않는다. 과도한 texture나 artifact는 추가 생성이 아니라 correction 대상이다.

## 실행

기본 val100 진단:

```bash
source /home/ubuntu/venvs/cuda/bin/activate
python tools/analysis/diagnose_detail_need_mask.py
```

짧은 smoke:

```bash
python tools/analysis/diagnose_detail_need_mask.py \
  --limit 8 \
  --num-workers 0 \
  --output-dir /home/ubuntu/scratch/sr-diffusion/runs/diagnose_detail_need_mask_smoke
```

Docker:

```bash
bash scripts/docker_lusir.sh run \
  python tools/analysis/diagnose_detail_need_mask.py --limit 8 --num-workers 0
```

출력:

```text
summary.json
detail_need_mask_grid.png
```

## 판단 지표

- `top0.20_missing_capture`: 상위 20% mask가 전체 missing-detail energy를 얼마나
  포함하는지. 무작위 선택 기대값은 약 `0.20`이다.
- `top0.20_missing_concentration`: 선택 영역의 missing-detail 밀도가 전체 평균의
  몇 배인지. `1.0`보다 충분히 높아야 한다.
- `top0.20_excess_capture`: 과도한 base texture까지 선택한 비율. 낮을수록 좋다.
- `top0.20_oracle_psnr_gain`: 선택 영역에만 실제 GT highpass correction을 넣었을
  때 base 대비 PSNR 이득.
- `proxy_*_corr`: 추론 때 관측 가능한 base/bicubic proxy와 GT target의 pixelwise
  상관. 낮으면 hand-crafted gate가 아니라 learned mask predictor가 필요하다.
- `proxy_*_top0.20_missing_capture`: 해당 proxy가 직접 선택한 상위 20% 영역의
  missing-detail 포착률. 작은 learned predictor가 넘어야 할 실제 baseline이다.

## 다음 gate

1. val100과 real-degradation 고정 review set에서 target grid를 눈으로 확인한다.
2. target이 missing texture가 아니라 일반 edge/noise를 고르면 정의를 수정한다.
3. target이 유효하면 base, bicubic, condition feature를 입력으로 하는 작은 mask
   predictor를 target에 supervision한다.
4. predictor가 target보다 충분히 낮은 성능이면 detail generator를 붙이지 않는다.
5. predictor가 통과한 뒤에만 masked detail head와 patch perceptual loss를 짧게
   ablation한다.

## 2026-06-13 photo-detail val100 결과

실행:

```bash
python tools/analysis/diagnose_detail_need_mask.py \
  --output-dir /home/ubuntu/scratch/sr-diffusion/runs/diagnose_detail_need_mask_val100
```

결과:

| selector | top 20% missing capture | missing concentration | excess capture |
| --- | ---: | ---: | ---: |
| GT-supervised target | `0.4878` | `2.4389x` | `0.3796` |
| highpass disagreement proxy | `0.3252` | `1.6262x` | `0.4838` |
| base/bicubic gap proxy | `0.3201` | `1.6005x` | `0.4762` |

- target 상위 20%는 무작위 기대값 `0.20`의 약 `2.44x` missing-detail 밀도를
  가지므로 위치 선택 signal은 유효하다.
- grid에서 라임 껍질, 털, 잎맥, 의류 무늬처럼 base에서 약해진 texture가
  집중적으로 선택됐다.
- 최고 hand-crafted proxy의 target correlation은 `0.5403`이다. target과 proxy
  사이에 학습할 여지는 있지만 proxy를 그대로 mask로 쓰기에는 excess 선택이 많다.
- top 20% oracle highpass correction은 base 대비 평균 `+5.0102 dB`다. 실제
  모델이 달성 가능한 수치는 아니지만, 위치를 제한한 detail correction의
  충분한 상한이 있음을 보여준다.

보존 결과:

```text
metrics/detail_need_mask_photo_detail_val100_summary.json
/home/ubuntu/scratch/sr-diffusion/runs/diagnose_detail_need_mask_val100/detail_need_mask_grid.png
```

작은 learned predictor의 1차 합격선:

```text
pixel correlation:       > 0.5403
top20 missing capture:   > 0.3252
top20 excess capture:    < 0.4838
```

## Learned predictor v1 결과

구현:

- `DetailMaskPredictor`: base, bicubic, frozen Stage2 condition latent, observable
  proxy 4종을 입력으로 받는 `460,545` parameter residual predictor.
- GT-supervised detail-need score 회귀와 pixelwise correlation을 학습하고,
  excess-detail 위치의 prediction을 별도로 억제한다.
- config: `configs/detail_mask_predictor_v1_probe.yaml`
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/2im6o5rn>
- best: step `3250`, `best_eval_mask.pt`

photo-detail val100:

| selector | correlation | top20 missing capture | excess capture | selection score |
| --- | ---: | ---: | ---: | ---: |
| highpass disagreement baseline | `0.5403` | `0.3252` | `0.4838` | `0.3817` |
| learned predictor best3250 | `0.7456` | `0.3861` | `0.4304` | `0.7013` |

- learned predictor는 사전에 정한 세 합격선을 모두 통과했다.
- selection score는 baseline보다 `+0.3196` 높다.
- step `250`부터 baseline을 넘었고, step `2250~4000` 구간에서도 성능이
  유지돼 짧은 우연이나 초기 spike로 보이지 않는다.
- 다음 단계는 predictor를 frozen spatial gate로 사용해 기존 v1d detail
  branch를 제한하는 masked branch다. mask가 없는 기존 v1d 동작은 그대로
  유지한다.

보존 요약:

```text
metrics/detail_mask_predictor_v1_val100_summary.json
```

## Masked detail branch v2 장기 run 완료

`configs/detail_branch_v2_masked_long_20ep.yaml`은 v1d best99500과 predictor
best3250에서 시작한다. predictor는 frozen이고, soft mask에 `0.05` floor를 둬
기존 correction을 완전히 차단하지 않는다.

```text
batch size:       4 (L40S 46GB에서 batch 5는 OOM)
grad accumulation: 1
learning rate:    1.25e-5
max steps:        667240 = 20 epochs upper bound
eval interval:    2000 steps
W&B:              https://wandb.ai/jwheo/LuSIR/runs/lyo21m9r
log:              /home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v2_masked_long_20ep/train.log
```

batch 4, grad accumulation 4에서 쓰던 `5e-5`를 그대로 사용하면 샘플당 optimizer
update 강도가 4배가 된다. 따라서 grad accumulation을 제거하면서 LR도 4분의
1로 낮춰 기존 update 강도를 보존했다.

20 epoch는 상한으로만 두고, 두 독립 continuation을 관찰해 plateau에서
조기 종료했다. W&B:

- <https://wandb.ai/jwheo/LuSIR/runs/lyo21m9r>
- <https://wandb.ai/jwheo/LuSIR/runs/oiyutuds>

선택 기준은 config에 고정한 `eval/detail_score`다. 동일한 val100 evaluator로
다시 비교한 결과:

| checkpoint | detail score | PSNR delta | mean PSNR delta | SSIM delta | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 36000 | `26.69528` | `+0.18744 dB` | `+0.20521 dB` | `+0.00721` | `100/100` |
| **step 38000** | **`26.69601`** | `+0.18177 dB` | `+0.20432 dB` | **`+0.00755`** | `100/100` |

- step 38000은 detail score가 `+0.00073` 높아 selected v2 checkpoint다.
- step 36000은 aggregate PSNR delta가 `+0.00567 dB` 높다. 차이는 평가/선택
  편차 수준이며 두 grid는 눈으로 거의 구분되지 않는다.
- step 38000 이후 step 50000까지 detail score best를 갱신하지 못했다.
- step 34000/38000/48000 grid도 거의 동일해 조기 중단 판단이 맞다.
- v1d ordinary val100 대비 수치는 소폭 좋아졌지만, GT에 없는 가짜 texture나
  흰 점 없이 보수적인 correction을 유지한 정도다. 실제 missing fine texture
  생성 문제는 해결하지 못했다.

W&B의 `samples/eval_grid`는 왼쪽부터 `LR`, `bicubic`, `base`, `detail`,
`residual`, `detail mask`, `GT`다. 해석 기준은 다음과 같다.

- `detail`이 base보다 선명해지되 흰 점, ringing, 가짜 무늬가 늘지 않는지.
- `detail mask`가 털, 잎맥, 표면 무늬에 열리고 평탄한 하늘/피부에는 과도하게
  열리지 않는지.
- `residual`이 점점 강해지기만 하지 않고 실제 missing-detail 위치에 정렬되는지.
- `eval/sr_vs_base_psnr`, `eval/sr_vs_base_ssim`은 양수를 유지해야 한다.
- `eval/sr_vs_base_highpass_l1`, `eval/sr_vs_base_laplacian_l1`도 양수가
  좋다. 이 둘은 GT 대비 error 감소량이다.
- `eval/wins_vs_base`, `eval/detail_wins_vs_base`가 감소하면서 residual/gate만
  커지면 중단한다.
- `eval/detail_score`는 checkpoint shortlist용이며 샘플 grid보다 우선하지 않는다.

보존 artifact:

```text
checkpoints/detail_branch_v2_masked_photo130k_lsdir_best38000.pt
configs/hf/detail_branch_v2_masked_photo130k_lsdir.yaml
metrics/detail_branch_v2_masked_photo130k_lsdir_summary.json
samples/detail_branch_v2_masked_photo130k_lsdir_best38000_grid.png
```

중요한 재현성 수정:

- 초기 구현은 학습/val에서만 learned mask를 적용하고 사용자 추론과 정식
  benchmark에서는 mask를 누락했다.
- `tools/infer/infer_detail_branch.py`와 `tools/eval/run_sr_benchmark.py`가
  config의 predictor checkpoint와 floor를 로드하도록 수정했다.
- mask가 없는 v1d config는 기존 동작을 그대로 유지한다.

결론:

- learned predictor는 `어디를 고칠지` 찾는 prerequisite로 유효하다.
- 그러나 기존 L1/highpass 중심 deterministic branch를 mask로 제한하는 것만으로
  `무엇을 생성할지` 문제가 해결되지는 않았다.
- 같은 objective의 추가 continuation이나 단순 branch 증량은 하지 않는다.
- 다음 detail 실험은 frozen fidelity base와 learned mask를 유지하면서,
  mask-weighted patch perceptual/adversarial supervision을 작은 bounded head에
  적용하고 lowpass drift/PSNR/real-image artifact guardrail로 제한한다.
