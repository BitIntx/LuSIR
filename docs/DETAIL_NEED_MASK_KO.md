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
