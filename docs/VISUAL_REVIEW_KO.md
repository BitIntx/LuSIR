# LuSIR Residual Refiner v2 시각 검토

## 비교 리포트 생성

같은 검증 샘플을 `LR / Bicubic / Condition / Conservative / Balanced /
Full / GT` 순서로 비교한다. 각 이미지를 클릭하면 원본 크기로 열린다.

```bash
python tools/analysis/compare_residual_strengths.py \
  --config configs/residual_refiner_stage2_xl_photo_detail_v2_continue_40k.yaml \
  --checkpoint /path/to/residual_refiner_stage2_xl_photo_detail_v2_best39000.pt \
  --output-dir outputs/residual_strength_visual_review \
  --presets photo_detail_mix mild photo_v2 photo_v3_noise_mix \
  --strengths 0.5 0.75 1.0 \
  --sample-count 8
```

생성 후 `outputs/residual_strength_visual_review/index.html`을 연다.

## 2026-06-07 대표 샘플 관찰

- `photo_detail_mix`와 `mild`에서는 Refiner가 Condition의 구조와 색을
  거의 보존하면서 경계와 일부 질감을 조금 개선한다.
- `Full`이 항상 육안상 최고는 아니다. 일부 잎, 털, 고주파 경계에서는
  `Balanced` 또는 `Conservative`가 GT에 더 자연스럽게 가깝다.
- `photo_v2`와 `photo_v3_noise_mix`에서는 Condition 단계가 강한 노이즈를
  제거하지만 실제 세부도 함께 크게 잃는다. Refiner 강도 차이는 이 손실에
  비해 작다.
- 청록/흰 격자형 점과 작은 색 번짐은 Refiner가 새로 만드는 현상보다는
  Condition 출력에 이미 존재하는 경우가 많다. 현재 Refiner는 이를
  제거하지 못한다.
- 글자, 동물 털, 잎맥, 먼 나뭇가지와 건물 표면은 GT보다 확실히 부드럽다.
  마지막 Refiner만 더 학습해서 해결하기 어려운 앞단 표현 병목이다.

## 현재 시각적 위치

LuSIR는 pretrained text-to-image 모델 없이 직접 학습한 vision-only,
deterministic x4 SR 연구 모델이다. 구조 보존, 예측 가능성, 낮은 환각 위험,
T4 실행 가능성은 장점이다. 반면 공개 선도 생성형 복원 모델들보다 실제로
보이는 미세 질감과 선명도는 아직 크게 부족하다.

비교 대상으로 참고할 공개 프로젝트:

- [SUPIR](https://github.com/Fanghua-Yu/SUPIR): 대형 생성 사전학습을 활용한
  photo-realistic restoration.
- [SeeSR](https://github.com/cswry/SeeSR): semantic-aware real-world SR.
- [OSEDiff](https://github.com/cswry/OSEDiff): one-step diffusion real-world SR.
- [PiSA-SR](https://github.com/csslc/PiSA-SR): pixel/semantic 품질을 조절하는
  diffusion SR.
- [DP2O-SR](https://github.com/cswry/DP2O-SR): perceptual preference 최적화
  기반 real-world SR.
- [AdcSR](https://github.com/Guaishou74851/AdcSR): adversarial diffusion
  compression 기반 real-world SR.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): 실용 복원 기준선.

현재 평가는 직접 동일 입력으로 실행한 SOTA 벤치마크가 아니라 구조와 공개
결과, 내부 검증 샘플을 바탕으로 한 정성 판단이다. SOTA를 주장하려면 동일
입력 blind A/B와 표준 벤치마크에서 LPIPS/DISTS/MANIQA/MUSIQ 및 사용자
선호도를 함께 측정해야 한다.

## 2026-06-14 masked detail v2 판단

- learned detail-mask predictor는 라임 표면, 털, 잎맥처럼 실제 detail이
  부족한 위치를 hand-crafted proxy보다 잘 찾는다.
- masked branch step 34000/38000/48000의 고정 grid는 거의 구분되지 않았다.
- selected step38000은 v1d보다 ordinary val100과 정식 219장 benchmark를
  일관되게 조금 개선했지만, missing texture가 새로 복구됐다고 볼 시각적
  차이는 없다.
- 흰 점/격자/과도한 sharpening이 늘지 않은 것은 장점이지만, learned mask만
  붙여서는 기존 deterministic branch의 보수적인 출력을 바꾸지 못했다.
- 다음 시각 검토는 같은 branch continuation이 아니라 mask-weighted patch
  perceptual/adversarial head가 실제 texture를 만들면서 artifact guardrail을
  지키는지 판단하는 데 사용한다.

## 판단 기준

1. 선명해 보이는지가 아니라 GT의 선, 질감, 물체 경계에 실제로 가까워졌는지 본다.
2. 흰 점, 청록 점, halo, ringing, 격자, 가짜 털과 잎맥이 생기면 실패로 본다.
3. 글자와 얼굴처럼 의미가 중요한 부분은 그럴듯한 환각보다 구조 보존을 우선한다.
4. `Full`이 더 날카롭지만 GT와 다르면 `Balanced`를 기본 사용자 모드로 본다.
5. 강한 열화 입력에서는 Refiner보다 Condition 출력 자체를 먼저 점검한다.
