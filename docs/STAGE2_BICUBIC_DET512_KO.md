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

## 중간 결과

2026-06-23 기준 run은 계속 진행 중이며, step 1000까지는 붕괴 없이 같은 방향의
개선을 보였다.

| step | decoded PSNR | mean PSNR | detail ratio | highpass ratio | missing | PSNR detail score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.20 | 26.95 | 0.326 | 0.791 | 0.01743 | 25.850 |
| 500 | 25.33 | 27.14 | 0.325 | 0.800 | 0.01687 | 25.984 |
| 1000 | 25.56 | 27.37 | 0.341 | 0.806 | 0.01639 | 26.242 |

step 1과 step 1000 sample grid의 평균 절대 RGB 차이는 약 `0.0086`로, 눈으로
보이는 변화는 아직 작다. 그러나 PSNR, highpass ratio, missing energy가 모두
같은 방향으로 움직였기 때문에 "64장에서는 가능하지만 512장에서는 즉시 사라지는
신호"는 아니다. 현재 판단은 다음과 같다.

- det512는 overfit64보다 훨씬 느리다.
- 그래도 subset을 키웠을 때 high-frequency 개선 방향은 유지된다.
- 이 run은 그대로 step 6000까지 두고, 최종 표에서 highpass ratio가 `0.82` 이상
  올라가는지와 missing이 계속 내려가는지를 본다.
- step 6000에서도 highpass ratio가 `0.81` 근처에 머물면, 다음은 단순 장기
  continuation이 아니라 train/heldout 분리와 crop/curriculum regularization
  쪽으로 옮긴다.
