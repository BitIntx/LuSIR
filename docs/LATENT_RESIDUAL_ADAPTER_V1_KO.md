# Stage2 Latent Residual Adapter v1

## 목적

v6 no-GAN detail branch는 v5처럼 붕괴하지 않았지만, best checkpoint가 step 0에
머물렀다. image-space detail branch에서 너무 안전하게 residual을 제한하면
artifact는 줄어도 새로운 texture가 거의 생기지 않는다.

다음 가설은 Stage2 conditional latent 자체가 너무 평균화되어 있고,
image-space 후처리 branch가 없는 정보를 복원하려고 하기 때문에 한계가 있다는
것이다. v1 adapter는 기존 Stage2 fidelity base를 보존한 채, latent 공간에서
작은 residual만 추가로 학습한다.

## 구조

```text
LR -> frozen dual-context Stage2 base best98000 -> base latent
LR + base latent -> zero-init latent residual adapter -> residual latent
base latent + bounded residual latent -> Stage1 decoder -> SR
```

핵심 제약:

- 기존 Stage2 base는 frozen이다.
- adapter output conv는 zero-init이다.
- 시작 출력은 기존 Stage2 base와 정확히 같다.
- optimizer는 adapter 파라미터만 잡는다.
- residual은 `residual_scale * tanh(logits)`로 bounded 된다.

## 설정

```text
config: configs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1.yaml
base: checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
adapter params: 3.75M
max steps: 12000 micro-steps
batch: 8 x grad_accum 4
loss: decoded + edge/highpass + small latent anchor
best metric: eval/mean_psnr_detail_score
```

## 성공 조건

- step 0이 기존 dual-context base와 같은 수치를 보여야 한다.
- `eval/decoded_mean_psnr` 또는 `eval/mean_psnr_detail_score`가 guarded v2
  수준 이상으로 올라가야 한다.
- `eval/highpass_energy_ratio`가 올라가도 `eval/missing_energy`가 같이 줄어야
  한다.
- sample grid에서 base보다 texture가 살아야 하고, v5처럼 scratch artifact가
  생기면 안 된다.

중단 기준:

- decoded/global PSNR과 mean PSNR이 base보다 계속 낮아짐
- highpass ratio만 오르고 missing energy나 sample grid가 개선되지 않음
- fixed samples가 더 날카롭기보다 noisy/dirty해짐

## 2026-06-22 smoke

4-step smoke는 정상이다.

```text
optimizer params: 3,745,296
loaded base checkpoint: stage2_photo130k_lsdir_dual_multiscale_best98000.pt
eval step=1 decoded_psnr=24.62
eval step=1 mean_psnr=26.49
eval step=1 highpass_ratio=0.789
eval step=1 missing=0.01968
```

이 값은 기존 base 시작점과 일치한다. 실제 판정은 장기 run의 500-step 단위
eval과 sample grid로 한다.

## 2026-06-22 장기 run 시작

```text
wandb: https://wandb.ai/jwheo/LuSIR/runs/o7tsc4mo
tmux:  lusir_latent_adapter_v1
log:   /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1/train.log
```

확인 명령:

```bash
tail -f /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1/train.log
```

초기 상태:

```text
step 1 eval decoded_psnr=24.62
step 1 eval mean_psnr=26.49
step 1 eval highpass_ratio=0.789
step 1 eval missing=0.01968
GPU: L40S 100%, about 27.4GB VRAM after startup
```
