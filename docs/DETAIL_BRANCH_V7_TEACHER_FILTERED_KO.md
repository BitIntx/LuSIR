# Detail Branch v7 Teacher-Filtered Hinge Probe

## 목적

v4/v6 RealESRGAN teacher 실험은 teacher highpass 신호를 보조 loss로 넣었지만,
visible texture breakthrough를 만들지 못했다. v7은 teacher를 전체 정답으로
모방하지 않고, GT highpass 기준으로 teacher가 frozen base보다 locally near/better인
위치만 선택해 직접 학습 신호로 사용한다.

핵심 가설:

- Stage2/Stage1 base는 fidelity가 안정적이지만 high-frequency detail이 부족하다.
- RealESRGAN teacher는 전체적으로는 GT보다 나쁠 수 있으나, 일부 patch에서는 base보다
  GT highpass에 가까운 detail prior를 제공한다.
- teacher-positive patch만 residual/highpass target과 GT highpass hinge로 쓰고,
  나머지는 guard/negative loss로 막으면 가짜 texture 확산을 줄일 수 있다.

## 구현

Config:

```text
configs/detail_branch_v7_teacher_filtered_hinge_probe.yaml
```

Run:

```text
/home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v7_teacher_filtered_hinge_probe
wandb: https://wandb.ai/jwheo/LuSIR/runs/4ysx4nk0
log:   /home/ubuntu/scratch/sr-diffusion/detail_branch_v7_teacher_filtered_hinge_probe.log
```

주요 변경:

- `teacher_improvement_mask`: teacher/base/GT highpass local error를 비교해
  teacher-positive mask를 만든다.
- `gt_highpass_hinge_losses`: teacher-positive 위치에서는 SR highpass error가 base보다
  낮아지도록 압력을 주고, teacher-negative 위치에서는 base보다 나빠지는 것을 막는다.
- learned noise-negative detail mask는 hard top10에서 top20 + floor `0.05`로 완화했다.
  기존 hard top10과 teacher-positive 영역의 겹침이 너무 작아 teacher signal이 거의
  차단됐기 때문이다.

## 초기 확인

4-step smoke:

```text
step0 val100:
  sr_psnr delta      +0.0967 dB
  mean_psnr delta    +0.1133 dB
  SSIM delta         +0.00364
  highpass ratio     +0.0129
  lowpass drift      0.000211
  outside mask L1    0.000497

step1 train:
  teacher_w          0.0372
  teacher_hinge      0.00076
  teacher_guard      0.00008
```

이 smoke는 성공 판정이 아니다. 다만 v6의 hard top10 설정에서
`teacher_w`가 약 `0.001-0.002` 수준으로 거의 막히던 문제가 풀렸고, teacher signal이
실제로 loss에 들어간다는 것을 확인했다.

## 판정 기준

5k probe에서 다음을 함께 본다:

- `eval/sr_vs_base_psnr`: 최소 `0 dB` 이상 유지, 가능하면 `+0.10 dB` 근처 유지.
- `eval/sr_vs_base_ssim`: 음수로 꺾이면 실패.
- `eval/sr_vs_base_highpass_ratio`: 오르되 grid에서 노이즈/가짜 texture가 보이면 실패.
- `eval/lowpass_drift_l1`: step0보다 크게 증가하면 실패.
- `eval/outside_mask_residual_l1`: floor 때문에 0은 아니지만 계속 작아야 한다.
- W&B train 지표:
  - `train/teacher_weight`가 너무 낮으면 teacher signal 차단.
  - `train/teacher_hinge`가 계속 0이면 GT highpass 개선 압력이 사라진 것.
  - `train/teacher_guard`, `train/negative_residual`이 커지면 artifact 위험 증가.

## 조기 중단

본 run은 step `500` 평가 후 중단했다.

```text
step250:
  sr_psnr delta      +0.0883 dB
  mean_psnr delta    +0.1101 dB
  SSIM delta         +0.00201
  highpass ratio     +0.0008
  laplacian ratio    -0.0141

step500:
  sr_psnr delta      +0.0821 dB
  mean_psnr delta    +0.1044 dB
  SSIM delta         +0.00158
  highpass ratio     -0.0021
  laplacian ratio    -0.0167
  lowpass drift      0.000146
  outside mask L1    0.000562
```

grid에서는 붕괴나 강한 artifact는 보이지 않았지만, visible detail 개선도 거의 없었다.
오히려 step0의 top20+floor inference gain 이후 학습이 진행될수록 highpass/laplacian
detail 지표가 줄었다. 이는 v6와 masked latent residual-shift v2에서 봤던
"안전해지지만 디테일이 사라지는" 패턴에 가깝다.

판정:

- v7 teacher-filtered hinge objective는 구현/로깅/캐시 경로 검증에는 성공했다.
- 그러나 현재 loss balance로는 promotion 가치가 없다.
- RealESRGAN teacher가 전체적으로 GT highpass보다 나쁜 배치가 많고, 선택 patch도
  학습을 지속하면 detail을 밀어 올리기보다 conservative residual로 수렴한다.
- 다음 실험은 teacher loss를 더 키우는 단순 continuation이 아니라, teacher 후보의
  patch selection 품질 자체를 재진단하거나 GT-aligned detail target을 별도 head로
  더 직접 예측하는 구조로 가야 한다.

## Teacher patch-quality 진단

후속 진단은 `tools/analysis/diagnose_teacher_patch_quality.py`로 수행했다.
RealESRGAN teacher cache 첫 256장에서 teacher는 base를 한 번도 이기지 못했다.

```text
teacher - base PSNR mean:      -2.5596 dB
teacher PSNR wins:             0 / 256
teacher highpass-L1 wins:      0 / 256
teacher-selected area:         0.2036
v7 effective teacher weight:   0.0177
selected patch improvement:    -0.000632
effective HP oracle gain:      +0.0131 dB
```

결론은 명확하다. RealESRGAN teacher를 더 세게 쓰거나 v7을 오래 돌리는 것은
우선하지 않는다. 상세 내용은 `docs/TEACHER_PATCH_QUALITY_KO.md`에 기록했다.
다음 active follow-up은 teacher를 제거하고 학습시에만 GT detail-need mask를 쓰는
`configs/detail_branch_v8_gtmask_training_probe.yaml`이다.
