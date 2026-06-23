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
