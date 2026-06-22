# Masked Latent Residual-Shift Diffusion v1

## 목적

기존 deterministic Stage 2는 PSNR/SSIM은 안정적이지만 conditional-mean
smoothing 때문에 없는 texture를 만드는 힘이 약했다. plain latent residual
adapter, image-space detail branch, noise-MSE Haar residual diffusion은 각각
안정적인 fidelity correction 또는 고주파 증가는 만들었지만 눈에 보이는
GT-aligned texture 개선으로 이어지지 않았다.

이번 실험은 public 경로를 바꾸지 않고, frozen Stage 2 latent에서 GT latent로
이동하는 **masked residual-shifting process**만 새로 학습한다.

```text
LR
  -> frozen dual-context Stage 2 best98000 -> base latent
  -> frozen Stage 1 decoder                -> base SR
base SR + bicubic + base latent
  -> frozen noise-negative mask best1500   -> hard top10 detail mask

noisy masked latent + base latent + LR + latent mask + timestep
  -> 19.19M ConditionalUNet
  -> masked latent correction
  -> frozen Stage 1 decoder
  -> SR
```

Stage 1, Stage 2, detail-mask predictor는 모두 frozen이다. 학습되는 것은
zero-init residual-shift U-Net뿐이다. mask 밖 latent correction은 모델 loss에
맡기지 않고 수식에서 0으로 제한한다.

## Residual-shift 정의

`z_b`는 frozen Stage 2 base latent, `z_gt`는 Stage 1 HR latent, `m`은 latent
해상도로 줄인 detail mask다.

```text
z_0 = z_b + m * (z_gt - z_b)
z_t = (1 - eta_t) * z_0 + eta_t * z_b + kappa * sqrt(eta_t) * epsilon
```

U-Net은 noise가 아니라 `z_b`에 더할 correction을 직접 예측한다. sampling은
같은 residual-shift trajectory에서 DDIM-style deterministic update를 사용한다.
현재 설정은 train timestep 100, inference 8 step, `kappa=0.15`다.

## Loss와 guardrail

- masked latent Charbonnier
- decoded image Charbonnier
- masked highpass/Laplacian Charbonnier
- small masked VGG feature loss
- mask 밖 base fidelity anchor
- lowpass base anchor
- latent correction L1

zero-init이므로 step 0 sampling 결과는 Stage 2 base와 정확히 같아야 한다.
실제 val20 smoke에서 이 조건을 확인했다.

```text
step 0 decoded PSNR: 27.6208
base decoded PSNR:   27.6208
PSNR delta:          +0.0000 dB
SSIM:                0.81938
detail ratio:        0.7965
outside-mask drift:  0.000000
```

## Probe

```text
config: configs/masked_latent_residual_shift_v1_probe.yaml
run:    https://wandb.ai/jwheo/LuSIR/runs/ldo6yzfu
steps:  5000
GPU:    1x L40S 48 GB
batch:  12, grad accumulation 1
peak:   약 32.5 GiB
speed:  smoke 약 0.68 step/s, 8.2 image/s
ETA:    eval 포함 약 2.1시간
```

```bash
tail -f /home/ubuntu/scratch/sr-diffusion/masked_latent_residual_shift_v1_probe.log
tmux attach -t residual_shift_v1
```

W&B에서 우선 볼 항목:

- `samples/eval_grid`: LR / mask / Stage 2 base / residual-shift / GT
- `eval/decoded_psnr`, `eval/psnr_delta_vs_base`
- `eval/decoded_ssim`, `eval/ssim_delta_vs_base`
- `eval/detail_ratio`, `eval/highpass_gain_vs_base`
- `eval/outside_mask_drift`, `eval/lowpass_drift`
- `eval/diversity_l1`

## 판정 기준

5k probe를 장기 run으로 확장하려면 다음을 함께 만족해야 한다.

- PSNR delta가 대략 `-0.05 dB`보다 나쁘지 않다.
- SSIM이 지속적으로 무너지지 않는다.
- detail ratio가 base `0.7965`보다 최소 약 `+0.02` 오르면서 highpass error도
  줄어든다. 단순 노이즈로 ratio만 올리는 경우는 실패다.
- mask 밖 drift와 lowpass drift가 작게 유지된다.
- fixed grid blind review에서 base 대비 detail 선호가 60%를 넘고 scratch,
  ringing, white speckle이 늘지 않는다.

현재 public Colab/HF 기본값은 바꾸지 않는다. probe가 위 gate를 통과한 뒤에만
40k-80k 장기 학습과 formal 219-image benchmark를 검토한다.

## v1 완료와 inference sweep

v1은 5000 step까지 완료됐고 best detail-score checkpoint는 step 3500이다.
full trajectory 결과는 다음과 같아 승격 기준을 크게 벗어났다.

```text
decoded PSNR:       26.8651 (base 대비 -0.7557 dB)
decoded SSIM:       0.80858 (base 대비 -0.01080)
detail ratio:       0.82718 (base 0.79652)
highpass gain:     -0.001725
outside-mask drift: 0.000915
```

detail ratio는 올랐지만 GT highpass error도 악화됐다. grid에서는 과일 표면,
동물 털, 잎 가장자리 등에 거친 가짜 질감이 보였다.

같은 best3500에서 correction strength를 낮춘 결과:

| strength | PSNR delta | SSIM delta | detail ratio | highpass gain |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | -0.0006 | +0.000003 | 0.79665 | -0.0000004 |
| 0.20 | -0.0019 | -0.000006 | 0.79684 | -0.0000019 |
| 0.35 | -0.0083 | -0.000074 | 0.79729 | -0.0000135 |
| 0.50 | -0.0287 | -0.000362 | 0.79830 | -0.0000582 |

strength를 줄이면 안전하지만 visible detail도 거의 사라졌다. full correction에서
start timestep을 낮춘 sweep도 같은 trade-off였다. timestep 10은 PSNR delta
`-0.0903 dB`, detail ratio `0.80099`, highpass gain `-0.000162`였다.

따라서 v1은 “full noise만 과했다”가 아니라 masked GT latent loss 자체가 decoded
fidelity보다 강했던 것으로 판단한다. v2 continuation은 best3500 optimizer
state에서 시작해 latent weight `1.0 -> 0.1`, image/highpass/outside anchor를
강화하고 timestep 10에서 평가한다.

```text
v2 config: configs/masked_latent_residual_shift_v2_fidelity_continue.yaml
v2 W&B:   https://wandb.ai/jwheo/LuSIR/runs/6e463dc0
range:    v1 best3500 -> step6500
```
