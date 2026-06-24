# Stage2 clean-bicubic generalization v1

deterministic512 probe는 같은 train512에서 detail metric을 올릴 수 있었지만,
held-out val100에서는 PSNR/SSIM이 하락하고 excess/highpass error가 증가했다.
V1의 목적은 fixed-subset memorization이 아니라 detail signal의 전이 여부를
직접 평가하는 것이다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_bicubic_generalization_v1.yaml
init:   checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
data:   benchmark_bicubic, full photo130k+LSDIR train split
wandb:  https://wandb.ai/jwheo/LuSIR/runs/yr815agn
```

설계:

- train은 stochastic crop, horizontal flip, texture-aware crop retry를 복원한다.
- 회전/기하 변형은 사용하지 않는다.
- 약한 HR color jitter(`[0.98, 1.02]`, probability `0.15`)만 추가한다.
- det512의 강한 highpass balance를 완화하고 원래 LR `5e-6`을 쓴다.
- GT보다 강한 local highpass energy만 soft hinge로 벌주는
  `artifact_excess_weight: 2.0`을 추가한다.
- primary eval은 held-out deterministic clean-bicubic val100이며 checkpoint도
  `eval/decoded_mean_psnr`로 선택한다.
- 동일 시점에 deterministic train512를 `eval_train512/*`로 함께 기록한다.
  train만 오르고 val이 떨어지는 순간을 W&B에서 바로 확인하기 위함이다.
- batch 8, gradient accumulation 1, 12000 micro-step/update로 실행한다.
- 중복 저장을 피하기 위해 best와 latest만 남기고 numbered step checkpoint는
  만들지 않는다.

중단/승격 기준:

- val mean PSNR과 SSIM이 best98000보다 개선되거나 최소한 유지돼야 한다.
- highpass ratio 상승만으로 성공 판정하지 않는다.
- val highpass L1 또는 excess energy가 초기값보다 지속 증가하면 실패다.
- train512와 val100 간 PSNR gap이 벌어지고 fixed grid에 ripple/grid texture가
  나타나면 조기 중단한다.
- 이 run은 Stage2 연구 probe이며 결과 확인 전 public/HF/Colab에 승격하지 않는다.

## 실행 상태

2026-06-24에 장기 run을 시작했다. step1 dual eval은 기존 best98000 기준을
재현했다.

```text
held-out val100: mean PSNR 26.92, SSIM 0.82143, highpass ratio 0.824,
                    highpass L1 0.03113, missing 0.01799, excess 0.00678
fixed train512:  mean PSNR 26.95, SSIM 0.80987, highpass ratio 0.791,
                    highpass L1 0.02935, missing 0.01743, excess 0.00539
```

L40S에서 batch 8은 약 `37.8 / 46.1 GiB`를 사용하고, eval/checkpoint 구간을
제외한 학습 속도는 약 `1.13 step/s`다.
