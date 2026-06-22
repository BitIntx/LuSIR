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

## 2026-06-22 완료 및 판정

장기 run은 `12000` micro-step까지 완료됐다.

```text
wandb:     https://wandb.ai/jwheo/LuSIR/runs/o7tsc4mo
best ckpt: /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1/checkpoints/best_eval_mean_psnr_detail.pt
best step: 11000
latest:    12000
```

val100 composite 기준 best step `11000`:

| metric | value |
| --- | ---: |
| decoded PSNR | `24.62017` |
| mean PSNR | `26.48841` |
| SSIM | `0.80135` |
| highpass ratio | `0.80192` |
| missing energy | `0.019230` |
| mean PSNR detail score | `28.09225` |

해석:

- step 1 대비 highpass ratio와 missing energy는 좋아졌다.
- 하지만 mean PSNR은 초기 `26.49146`보다 낮고, decoded PSNR도 사실상
  변하지 않았다.
- sample grid에서 6500/9000/11000/12000의 육안 차이는 작았다.
- 붕괴는 없었지만, visible texture를 새로 만든 결과도 아니었다.

같은 219장 clean-bicubic formal x4 benchmark에 투입해 frozen Stage2 base,
guarded Stage2 v2, masked detail v2와 비교했다. SSIM 계산을 위해
`opencv-python-headless`를 설치하고 기존 evaluator의 MATLAB-compatible Y
PSNR/SSIM protocol을 그대로 사용했다.

| candidate | mean Y PSNR | mean Y SSIM | mean RGB PSNR | mean RGB SSIM |
| --- | ---: | ---: | ---: | ---: |
| bicubic | `25.7170` | `0.71773` | `24.2697` | `0.69205` |
| stage2 base | `27.8431` | `0.79742` | `26.3131` | `0.77340` |
| latent adapter v1 | `27.8294` | `0.79836` | `26.3031` | `0.77430` |
| guarded Stage2 v2 | `27.8539` | `0.79945` | `26.3263` | `0.77555` |
| guarded Stage2 v2 x8 | `27.9496` | `0.80175` | `26.4303` | `0.77844` |
| masked detail v2 | `28.1429` | `0.80797` | `26.6097` | `0.78473` |

Pairwise 판정:

```text
latent adapter v1 vs stage2 base:
  Y PSNR -0.0138 dB, Y SSIM +0.00094
  Y PSNR wins 80/219, Y SSIM wins 191/219

latent adapter v1 vs guarded Stage2 v2:
  Y PSNR -0.0246 dB, Y SSIM -0.00109
  Y PSNR wins 67/219, Y SSIM wins 20/219

latent adapter v1 vs masked detail v2:
  Y PSNR -0.3135 dB, Y SSIM -0.00960
  Y PSNR wins 1/219, Y SSIM wins 0/219
```

보존 파일:

```text
metrics/formal_x4_benchmark_stage2_latent_adapter_v1_value_compare_summary.json
metrics/formal_x4_benchmark_stage2_latent_adapter_v1_value_compare_metrics.csv
samples/stage2_latent_adapter_v1_value_compare_selected.jpg
samples/stage2_latent_adapter_v1_value_compare_contact_sheet.jpg
```

최종 결론:

- `latent_residual_adapter_v1`은 연구 기록으로는 의미가 있다. frozen base 위에
  bounded latent residual을 붙여도 붕괴하지 않고 SSIM을 아주 조금 올릴 수
  있다는 것을 확인했다.
- 그러나 public/default 또는 research-best 후보로 승격하지 않는다.
- Stage2 base 대비 Y PSNR을 잃고, guarded Stage2 v2와 masked detail v2보다
  명확히 낮다.
- 다음 visible-detail 연구는 plain latent residual adapter continuation이 아니라
  learned mask/detail head 또는 patch-level perceptual/artifact-negative
  supervision 쪽으로 가는 것이 맞다.
