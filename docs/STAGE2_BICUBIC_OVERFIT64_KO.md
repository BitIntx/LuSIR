# Stage2 clean-bicubic overfit64 probe

목적은 성능 향상 run이 아니라 원인 분리다. 기존 clean-bicubic continuation은
val100 proxy에서 `25.057 dB` 부근에 plateau했고, LR 증가는 병목을 바꾸지
못했다. 그래서 이번에는 일반화 문제를 제거하고, 현재 Stage2 구조와 loss가
고정된 소수 clean-bicubic 샘플을 실제로 외울 수 있는지 확인한다.

```text
config: configs/latent_pretrain_photo130k_lsdir_dual_bicubic_overfit64_probe.yaml
init:   checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
data:   benchmark_bicubic, train split first 64 samples
run:    /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_bicubic_overfit64_probe
log:    /home/ubuntu/scratch/sr-diffusion/latent_pretrain_photo130k_lsdir_dual_bicubic_overfit64_probe.log
wandb:  https://wandb.ai/jwheo/LuSIR/runs/12ui2qg0
```

구현 변경:

- `ManifestImageDataset`에 `max_items` 옵션을 추가했다.
- Stage2 `make_dataset`은 `data.deterministic_train: true`일 때 train split도
  deterministic crop/degradation으로 읽는다.
- overfit config는 `max_items: 64`, `deterministic_train: true`,
  `hflip_prob: 0.0`, `texture_crop_retries: 1`을 사용한다.
- eval도 같은 train 64장에 걸린다. 즉 이 metric은 일반화 성능이 아니라
  표현/최적화 upper-bound 진단이다.

4-step smoke는 정상이다.

```text
eval step1:
  decoded_psnr      24.30
  mean_psnr         26.42
  detail_ratio      0.325
  highpass_ratio    0.791
  missing           0.01971
  psnr_detail_score 24.954
```

판정 기준:

- 성공 신호: train64에서 `mean_psnr`이 빠르게 오르면서 `highpass_ratio`도
  `0.80`대 초반을 넘어 올라가고, `missing`이 확실히 내려간다.
- 부분 성공: PSNR만 오르고 highpass/missing이 그대로면 현재 loss가 여전히
  평균 복원 쪽으로 작동한다.
- 실패 신호: train64에서도 highpass/missing을 개선하지 못하면 데이터셋 확대나
  장기 continuation보다 target parameterization 또는 architecture 변경이 먼저다.

이 probe 결과가 좋더라도 그대로 public/default로 승격하지 않는다. 고정 64장
overfit은 “가능한가”를 확인하는 진단이며, 다음 단계는 같은 판단 기준을 val set
일반화로 옮기는 것이다.

## 완료 결과

3000 micro-step까지 정상 완료됐다. W&B run은
<https://wandb.ai/jwheo/LuSIR/runs/12ui2qg0>이다.

| step | decoded PSNR | mean PSNR | highpass ratio | missing | mean-PSNR detail score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.3048 | 26.4246 | 0.7908 | 0.01971 | 28.0062 |
| 500 | 25.1762 | 27.5347 | 0.8325 | 0.01682 | 29.1997 |
| 1000 | 25.7465 | 28.1343 | 0.8603 | 0.01520 | 29.8549 |
| 1500 | 26.1845 | 28.5935 | 0.8636 | 0.01448 | 30.3207 |
| 2000 | 26.4810 | 28.8921 | 0.8765 | 0.01370 | 30.6451 |
| 2500 | 26.6982 | 29.1026 | 0.8759 | 0.01349 | 30.8544 |
| 3000 | 26.8461 | 29.2477 | 0.8812 | 0.01314 | 31.0100 |

Best checkpoint:

```text
/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_bicubic_overfit64_probe/checkpoints/best_eval_mean_psnr_detail.pt
```

판정:

- 고정 train64에서는 PSNR, highpass ratio, missing energy가 모두 뚜렷하게
  좋아졌다.
- 따라서 현재 dual-context Stage2 구조가 clean-bicubic detail을 표현 자체로
  못 하는 것은 아니다.
- active bottleneck은 더 좁게 보면 구조 절대용량보다 일반 train/val 조건에서
  high-frequency target을 안정적으로 일반화시키는 방법이다.
- 다음 실험은 같은 64장 overfit continuation이 아니라, 고정 subset에서 확인된
  detail-preserving signal을 larger train/val split으로 옮기는 data/curriculum
  또는 regularization probe여야 한다.
