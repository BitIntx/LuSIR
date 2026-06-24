# Stage2 clean-bicubic deterministic512 probe

overfit64에서는 고정 train64에서 PSNR, highpass ratio, missing energy가 모두
개선됐다. 이 결과는 Stage2가 clean detail을 표현 자체로 못 하는 구조는
아니라는 신호다. 다음 질문은 subset을 키웠을 때도 같은 detail-preserving
signal이 유지되는지다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_bicubic_det512_probe.yaml
init:   checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
data:   benchmark_bicubic, train split first 512 samples
run:    /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_bicubic_det512_probe
log:    /home/ubuntu/scratch/sr-diffusion/latent_pretrain_photo130k_lsdir_dual_bicubic_det512_probe.log
wandb:  https://wandb.ai/jwheo/LuSIR/runs/0wzx4xzy
```

설계:

- overfit64 best에서 이어받지 않고 dual-context best98000에서 다시 시작한다.
- `max_items: 512`, `deterministic_train: true`, `hflip_prob: 0.0`,
  `texture_crop_retries: 1`로 train subset을 고정한다.
- eval은 같은 train512에 걸린다. 따라서 이 역시 deployable 성능이 아니라
  subset-scale upper-bound 진단이다.
- 64장 대비 epoch 수가 1/8로 줄어드므로 `max_steps: 6000`으로 잡는다.
- 디스크가 빠듯하므로 `save_every: 6000`으로 두고 중간 checkpoint를 남기지
  않는다. best checkpoint는 overwrite되는 단일 파일만 유지한다.

판정 기준:

- 좋은 신호: overfit64보다 느리더라도 `mean_psnr`, `highpass_ratio`,
  `missing`이 같은 방향으로 개선된다.
- 일반화 병목 신호: PSNR은 오르지만 highpass ratio가 `0.82-0.84` 부근에서
  멈추거나 missing이 거의 내려가지 않는다.
- 다음 단계: 512에서도 detail metric이 유지되면 2048 deterministic subset 또는
  train512/heldout-val dual eval로 넘어간다. 512에서 바로 약해지면 data/crop
  curriculum과 loss regularization을 먼저 조정한다.

## 최종 결과

run은 2026-06-24에 step 6000까지 정상 완료됐다. 같은 train512에서는 모든
주요 지표가 끝까지 개선됐다.

| step | decoded PSNR | mean PSNR | detail ratio | highpass ratio | missing | PSNR detail score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.20 | 26.95 | 0.326 | 0.791 | 0.01743 | 25.850 |
| 500 | 25.33 | 27.14 | 0.325 | 0.800 | 0.01687 | 25.984 |
| 1000 | 25.56 | 27.37 | 0.341 | 0.806 | 0.01639 | 26.242 |
| 2000 | 25.91 | 27.77 | 0.397 | 0.825 | 0.01537 | 26.706 |
| 3000 | 26.14 | 28.01 | 0.418 | 0.833 | 0.01490 | 26.971 |
| 4000 | 26.30 | 28.20 | 0.431 | 0.839 | 0.01448 | 27.163 |
| 5000 | 26.42 | 28.32 | 0.441 | 0.840 | 0.01432 | 27.297 |
| 6000 | 26.50 | 28.40 | 0.453 | 0.845 | 0.01410 | 27.400 |

그러나 같은 checkpoint를 학습에 사용하지 않은 deterministic clean-bicubic
val100에 평가하면 반대 결과가 나온다.

| candidate | mean PSNR | SSIM | highpass ratio | highpass L1 | missing | excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| init best98000 | 26.9198 | 0.82143 | 0.8244 | 0.03113 | 0.01799 | 0.00678 |
| det512 step6000 | 26.2488 | 0.80582 | 0.8601 | 0.03366 | 0.01742 | 0.00877 |
| delta | -0.6710 | -0.01561 | +0.0357 | +0.00253 | -0.00057 | +0.00199 |

시각적으로도 train512 prediction은 step1보다 윤곽과 미세 대비가 강해졌지만,
나뭇잎, 잔디, 얼룩말 주변에 반복 격자와 잔물결 질감이 생겼다. 이는 GT-aligned
detail 복원보다 고정 subset의 고주파 패턴에 맞춘 결과와 일치한다.

최종 판단:

- 64장에서 512장으로 subset을 키워도 current Stage2가 clean detail을 외우는
  표현 능력은 유지된다.
- 하지만 그 신호는 held-out val100에 전이되지 않는다. PSNR/SSIM이 크게
  떨어지고 highpass L1 및 excess energy가 악화했다.
- highpass ratio 단독 상승은 성공 기준으로 쓸 수 없다. highpass error,
  excess energy, held-out PSNR/SSIM, fixed grid를 함께 봐야 한다.
- 이 checkpoint는 public/HF/Colab 후보로 승격하지 않고 진단용으로만 보존한다.
- 다음 Stage2 실험은 더 큰 fixed-subset memorization이 아니라 stochastic
  crop/degradation, held-out dual eval, artifact-negative regularization을
  포함한 일반화 실험이어야 한다.
