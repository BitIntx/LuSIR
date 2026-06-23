# RealESRGAN Teacher Patch Quality Diagnostic

## 목적

v7 teacher-filtered detail branch는 RealESRGAN teacher를 전체 정답으로 모방하지 않고,
GT highpass 기준으로 teacher가 base보다 locally near/better인 patch만 쓰도록
설계했다. 그러나 step500에서 highpass/laplacian detail 지표가 줄었다. 이 문서는
teacher signal 자체가 학습에 쓸 만한지 진단한 결과다.

## 실행

```bash
python -u tools/analysis/diagnose_teacher_patch_quality.py \
  --config configs/detail_branch_v7_teacher_filtered_hinge_probe.yaml \
  --limit 256 \
  --batch-size 4 \
  --num-workers 4 \
  --output-dir /home/ubuntu/scratch/sr-diffusion/runs/diagnose_teacher_patch_quality_v7_train256
```

출력:

```text
summary: /home/ubuntu/scratch/sr-diffusion/runs/diagnose_teacher_patch_quality_v7_train256/summary.json
grid:    /home/ubuntu/scratch/sr-diffusion/runs/diagnose_teacher_patch_quality_v7_train256/teacher_patch_quality_grid.png
```

대상은 RealESRGAN teacher cache 첫 256장이다:

```text
/home/ubuntu/scratch/sr-diffusion/teacher_cache/realesrgan_x4plus_photo_detail_mix_2048
```

## 결과

요약:

```text
base PSNR mean:               26.1580
teacher PSNR mean:            23.5984
teacher - base PSNR:          -2.5596 dB
teacher PSNR wins:            0 / 256
teacher highpass-L1 wins:     0 / 256

teacher selected area:        0.2036
v7 effective teacher weight:  0.0177
selected patch improvement:   -0.000632

base PSNR:                    26.1580
effective HP oracle PSNR:     26.1711
effective HP oracle gain:     +0.0131 dB
```

해석:

- RealESRGAN teacher는 이 cache/protocol에서 base보다 전역 fidelity가 훨씬 낮다.
- teacher highpass-L1도 모든 샘플에서 base를 이기지 못했다.
- v7 filter는 약 20% 영역을 teacher-positive로 고르지만, 그 영역의 평균
  local highpass improvement도 음수다. 즉 filter가 margin 때문에 "덜 나쁜"
  영역을 통과시키는 경우가 많다.
- learned detail mask까지 곱한 실제 v7 effective teacher weight는 약 1.8%에
  불과하다. teacher signal이 너무 약하고, 그마저 평균적으로 GT-aligned가 아니다.
- teacher highpass residual을 oracle처럼 직접 넣어도 PSNR gain은 약 `+0.013 dB`로
  작다. visible texture breakthrough를 기대하기 어렵다.

## 결론

RealESRGAN teacher를 단순히 더 강하게 쓰거나 v7을 오래 돌리는 것은 우선하지 않는다.
다음 실험은 teacher를 제거하고, 학습시에만 GT detail-need mask를 사용해 branch가
정확한 위치에서 residual/highpass target을 보도록 하는 v8 probe로 진행한다. inference와
eval은 여전히 learned noise-negative mask를 사용하므로 런타임 GT 누수는 없다.

Active follow-up:

```text
config: configs/detail_branch_v8_gtmask_training_probe.yaml
wandb:  https://wandb.ai/jwheo/LuSIR/runs/099kwayk
log:    /home/ubuntu/scratch/sr-diffusion/detail_branch_v8_gtmask_training_probe.log
```
