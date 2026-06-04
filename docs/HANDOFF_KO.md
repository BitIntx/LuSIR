# VM Handoff / 대화 기반 인수인계

이 문서는 원문 채팅 로그가 아니라, 현재 대화에서 결정하고 실행한 내용을
다른 VM의 Codex/작업자가 바로 이어받을 수 있게 정리한 공개용 요약입니다.

최신 실패/부분 성공/다음 가설 기록은 `docs/TRIAL_AND_ERROR_KO.md`에 누적합니다.

## 목표

직접 학습하는 x4 vision-only latent diffusion super-resolution 모델.

- T2I pretrained diffusion 모델을 사용하지 않음.
- `LR 128x128 -> HR 512x512`가 현재 기본 목표.
- `LR 192x192 -> HR 768x768`은 이후 목표.
- photo / anime domain conditioning을 한 코드베이스에서 처리.
- PyTorch first.
- GPU/ROCm 우선, TPU/XLA는 나중에 고려.
- custom CUDA/ROCm op 없이 표준 PyTorch 위주로 유지.

## 모델 구조

```text
HR image
  -> factor-4 VAE / AutoencoderKL
  -> HR latent

LR image
  -> LR-to-latent condition encoder
  -> condition latent

noisy HR latent + condition latent + timestep + domain embedding
  -> conditional diffusion U-Net
  -> denoised HR latent
  -> VAE decoder
  -> x4 SR output
```

현재 파라미터 수:

```text
Stage 1 VAE:                  21.096M
Stage 2 LR-to-latent encoder:  2.388M
Stage 3 diffusion U-Net:      76.610M
Full inference path:         100.094M
```

500M급 확장 config:

```text
XL Stage 1 VAE:                  21.096M
XL Stage 2 LR-to-latent encoder: 18.944M
XL Stage 4 diffusion U-Net:     469.618M
XL full inference path:         509.658M
Stage 2 XL config: configs/latent_pretrain_photo100k_v3_noise_xl.yaml
Stage 4 XL config: configs/diffusion_photo100k_xl_stage4_condition_v3.yaml
```

## 라이선스 / 공개 상태

- GitHub: <https://github.com/BitIntx/sr-diffusion>
- Hugging Face: `jwheo/sr-diffusion`
- GitHub repo는 public.
- HF model repo도 public.
- code license: PolyForm Noncommercial 1.0.0.
- checkpoint/artifact license: CC BY-NC 4.0.
- 상업적 이용은 금지.
- raw training data는 repo/HF에 올리지 않음.

## 현재 구현된 것

- 프로젝트 scaffold.
- config system.
- manifest 기반 dataset loader.
- x4 degradation pipeline.
- Stage 1 VAE training/eval/inference.
- Stage 2 deterministic LR-to-HR-latent pretrain.
- Stage 3 conditional latent diffusion training.
- Stage 4 condition-start fine-tune prototype.
- W&B online logging.
- fixed sample image logging: LR / GT / Pred.
- sampled validation/eval tooling.
- HF artifact upload/download scripts.
- Colab demo.
- tiled inference:
  - `--tile`
  - 128x128 LR tiles
  - overlap feather blending
  - arbitrary-size LR image to x4 output

## 데이터 상태

Scratch root:

```text
/home/jwheojjang/scratch/sr-diffusion
```

현재 주요 manifests:

```text
data/manifest_photo10k.csv
data/manifest_photo100k.csv
```

`manifest_photo100k.csv`:

```text
photo/train: 103450
photo/val:   100
```

구성:

- DIV2K
- Flickr2K
- deterministic COCO train2017 subset

COCO train2017은 `min_size>=480` 조건으로는 45,897장밖에 안 나와서,
photo100k는 `min_size>=320` 기준으로 구성했다.

Scratch는 VM restart나 VM 이동 시 날아갈 수 있으므로 데이터는 복구
스크립트로 다시 받는 전제로 운영한다.

## 완료된 학습

### Stage 1: VAE / Autoencoder

```text
config: configs/autoencoder_photo10k.yaml
run: autoencoder_photo10k_b16_eval_online
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt
finished step: 50000
eval/recon: 0.01198
eval/kl:    9.38684
eval/psnr:  40.19
```

HF:

```text
checkpoints/stage1_autoencoder_best_eval_recon.pt
```

### Stage 2: 10k LR-to-latent

```text
config: configs/latent_pretrain_photo10k.yaml
run: latent_pretrain_photo10k_b16
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
finished step: 50000
best eval/latent_loss: step 48000, 0.21775
best decoded PSNR proxy: step 47000, 23.89
```

HF:

```text
checkpoints/stage2_latent_pretrain_best_eval_latent.pt
```

### Stage 3: 10k diffusion

```text
config: configs/diffusion_photo10k_b32.yaml
run: diffusion_photo10k_b32
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
finished step: 25000
best eval/noise_mse checkpoint: step 24000
sampled val100 t50 32-step:
  SR PSNR:      25.2216
  bicubic PSNR: 24.4778
  delta:        +0.7438
```

HF:

```text
checkpoints/stage3_diffusion_b32_best_eval_noise.pt
```

### Stage 4: 10k condition-start prototype

```text
config: configs/diffusion_photo10k_b32_stage4_condition.yaml
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt
best checkpoint step: 1000
val100 t25 32-step sampled SR PSNR: 25.2930
```

HF:

```text
checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt
metrics/stage4_condition_val100_t25_32step_summary.json
```

### Stage 2: photo100k scale-up

```text
config: configs/latent_pretrain_photo100k.yaml
run: latent_pretrain_photo100k_b64
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 30000
best eval/latent_loss: step 28000, 0.21230
best decoded PSNR proxy: step 22000, 23.93
final eval: step 30000, eval/latent_loss 0.21267, decoded_psnr 23.88
```

HF:

```text
checkpoints/stage2_photo100k_b64_best_eval_latent.pt
metrics/stage2_photo100k_b64_summary.json
```

### Stage 3: photo100k scale-up

```text
config: configs/diffusion_photo100k_b32.yaml
run: diffusion_photo100k_b32
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
condition encoder:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 60000
best eval/noise_mse: step 60000, 0.00680
best decoded PSNR proxy: step 53000, 24.57
final eval: step 60000, decoded_psnr 24.56, eval/x0_mse 0.08052
```

HF:

```text
checkpoints/stage3_photo100k_b32_best_eval_noise.pt
metrics/stage3_photo100k_b32_summary.json
```

Stage3 sampled eval:

```text
output:
  /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_photo100k_val100_t50_32step
val100, 32 DDIM steps:
  SR PSNR:      25.3745
  bicubic PSNR: 24.4778
  delta:        +0.8967
```

### Stage 4: photo100k condition-start

```text
config: configs/diffusion_photo100k_b32_stage4_condition.yaml
run: diffusion_photo100k_b32_stage4_condition
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt
finished step: 5000
best decoded checkpoint: step 1500
sampled val100 t25 32-step:
  SR PSNR:      25.4072
  bicubic PSNR: 24.4778
  delta:        +0.9294
  vs Stage3:    +0.0327, wins 68 / losses 32
```

HF:

```text
checkpoints/stage4_photo100k_condition_b32_best_eval_condition_decoded.pt
metrics/stage4_photo100k_condition_val100_t25_32step_summary.json
metrics/stage4_photo100k_condition_compare_stage3_summary.json
```

### Stage 2: photo100k degradation v2 fine-tune

```text
config: configs/latent_pretrain_photo100k_v2.yaml
run: latent_pretrain_photo100k_v2_b64
degradation preset: photo_v2
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 20000
best eval/latent_loss: step 19000, 0.28528
best decoded PSNR proxy: step 15000, 21.91
final eval: step 20000, eval/latent_loss 0.28704, decoded_psnr 21.60
```

`photo_v2`는 `mild`보다 훨씬 강한 LR degradation이므로 Stage2 mild의
decoded PSNR 23.9대와 직접 비교하면 안 된다. 이 checkpoint는 v2 LR 입력을
diffusion condition latent로 안정적으로 넘기기 위한 기준 checkpoint다.

HF:

```text
checkpoints/stage2_photo100k_v2_b64_best_eval_latent.pt
metrics/stage2_photo100k_v2_b64_summary.json
```

### Stage 3: photo100k degradation v2 fine-tune

```text
config: configs/diffusion_photo100k_b32_v2.yaml
run: diffusion_photo100k_b32_v2
degradation preset: photo_v2
condition encoder:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/checkpoints/best_eval_noise.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt
finished step: 20000
best eval/noise_mse: step 19000, 0.00821
best decoded PSNR proxy: step 19000, 23.44
sampled val100 t50 32-step:
  SR PSNR:      22.6699
  bicubic PSNR: 22.4103
  delta:        +0.2595
  wins/losses:  63 / 37
```

정성 확인:

- 강한 noise/JPEG 계열에서는 bicubic보다 denoise가 되는 샘플이 있다.
- 일부 샘플에서는 색/대비 과보정, 녹색 점 artifact, 과한 texture smoothing이
  보인다.
- 따라서 Stage3 v2는 최종 품질 checkpoint가 아니라 Stage4 v2
  condition-start 안정화의 시작점으로 본다.

HF:

```text
checkpoints/stage3_photo100k_v2_b32_best_eval_noise.pt
metrics/stage3_photo100k_v2_b32_summary.json
metrics/stage3_photo100k_v2_val100_t50_32step_summary.json
```

### Stage 4: photo100k degradation v2 condition-start

```text
config: configs/diffusion_photo100k_b32_stage4_condition_v2.yaml
run: diffusion_photo100k_b32_stage4_condition_v2
degradation preset: photo_v2
condition encoder:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints/best_eval_condition_decoded.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/checkpoints/best_eval_noise.pt
finished step: 5000
best decoded checkpoint: step 1000
best decoded PSNR proxy: step 1000, 22.12
sampled val100 t25 32-step:
  SR PSNR:      22.8426
  bicubic PSNR: 22.4103
  delta:        +0.4323
  wins/losses:  70 / 30 vs bicubic
  vs Stage3 v2: +0.1727, wins 81 / losses 19
```

정성 확인:

- Stage3 v2보다 평균/승률은 개선됐고, 일부 overshoot가 완화됐다.
- 하지만 어두운 영역과 강한 artifact 샘플에서 cyan/green 점 artifact가
  아직 보인다.
- 다음 단계는 더 긴 v2 condition-start보다 artifact 억제 loss/샘플링 조정,
  A/B review가 우선이다.

HF:

```text
checkpoints/stage4_photo100k_condition_v2_b32_best_eval_condition_decoded.pt
metrics/stage4_photo100k_condition_v2_b32_summary.json
metrics/stage4_photo100k_condition_v2_val100_t25_32step_summary.json
metrics/stage4_photo100k_condition_v2_compare_stage3_v2_summary.json
```

### Stage 2: photo100k v3 noise XL condition encoder

```text
config: configs/latent_pretrain_photo100k_v3_noise_xl.yaml
run: latent_pretrain_photo100k_v3_noise_xl_b64
degradation preset: photo_v3_noise_mix
params: 18.944M
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/best_eval_latent.pt
finished step: 80000
best eval/latent_loss: step 66000, 0.27230
best decoded PSNR proxy: step 72000, 21.52
final eval: step 80000, eval/latent_loss 0.27592, decoded_psnr 21.51
```

비교:

```text
small v3 best:
  step 11000, eval/latent_loss 0.28187, decoded_psnr 21.30
XL best:
  step 66000, eval/latent_loss 0.27230, decoded_psnr 21.38
XL best PSNR proxy:
  step 72000, eval/latent_loss 0.27940, decoded_psnr 21.52
```

판단:

- XL condition encoder는 small v3를 확실히 넘었다.
- Stage4 XL을 바로 시작하지 않았다.
- Stage4 전에 `best_eval_latent.pt`, `step_0072000.pt`, `latest.pt` 세 후보의
  condition-only decoded sample을 같은 validation 이미지에서 비교하는 것이 좋다.
- `best_eval_latent.pt`는 latent loss 최선이고, `step_0072000.pt`와
  `latest.pt`는 decoded PSNR/latent MSE 측면에서 볼 가치가 있다.

HF:

```text
checkpoints/stage2_photo100k_v3_noise_xl_b64_best_eval_latent.pt
checkpoints/stage2_photo100k_v3_noise_xl_b64_step_0072000.pt
checkpoints/stage2_photo100k_v3_noise_xl_b64_latest.pt
metrics/stage2_photo100k_v3_noise_xl_b64_summary.json
configs/latent_pretrain_photo100k_v3_noise_xl.yaml
configs/diffusion_photo100k_xl_stage4_condition_v3.yaml
```

## 현재 관찰 / 판단

- 기본 x4 복원력은 잡혔다.
- Stage4 condition-start는 Stage3 sampled eval 대비 소폭 개선됐다.
- 디노이즈와 선명화 능력은 아직 `mild` degradation 기준으로 제한적이다.
- `photo_v2` degradation과 Stage2 v2 condition encoder는 구현 및 20k
  fine-tune까지 완료됐다.
- Stage3/Stage4 v2 fine-tune과 sampled eval까지 완료됐다.
- Stage4 v2는 Stage3 v2보다 낫지만, 사용자가 체감할 artifact 억제가 아직
  남아 있다.
- 강한 noise/color-noise curriculum을 가진 Stage2 v3 small run은 step
  12700에서 중단했다. eval이 9k~12k에서 `eval/latent_loss` 약 0.282로
  횡보해, 더 태우는 것보다 500M급 확장으로 넘어가는 판단을 했다.
- Stage2 XL condition encoder는 80k step까지 완료됐고 small v3를 넘었다.
- 현재 실행 중인 학습은 없다. Stage4 XL은 아직 시작하지 않았다.
- 다음 개선축은 Stage2 XL 후보 비교 후 Stage4 XL condition-start diffusion
  U-Net이다.

## 다음 작업

우선순위:

1. Stage2 XL condition encoder 후보 비교:
   - `best_eval_latent.pt`: step 66000, latent loss 최선
   - `step_0072000.pt`: decoded PSNR proxy 최선
   - `latest.pt`: step 80000, final eval PSNR proxy가 높고 latent MSE가 낮음
   - 같은 val 이미지에서 LR / GT / decoded condition output contact sheet를 만든다.
2. Stage4 photo100k XL condition-start:
   - config: `configs/diffusion_photo100k_xl_stage4_condition_v3.yaml`
   - params: U-Net 469.618M, full path 509.658M
   - init diffusion: Stage4 v2 `best_eval_condition_decoded.pt`
   - use `--partial-init` because architecture is deeper/wider
   - condition encoder: Stage2 XL 후보 비교 후 선택
   - 아직 시작하지 않았음
3. A/B review sheet 정리:
   - mild Stage4 vs Stage3 v2 vs Stage4 v2
   - denoise, sharpening, artifact, naturalness 기준
4. artifact 억제 실험:
   - cyan/green dot artifact를 줄이는 sampled eval 기준 마련
   - color/contrast overshoot penalty 또는 lower start timestep 검토
5. perceptual/fidelity fine-tune:
   - `x0_weight`를 켠 condition-start training
   - LPIPS/VGG perceptual loss 검토
   - GAN은 나중에, A/B eval 기반으로 조심스럽게
6. few-step distillation.
7. A/B Elo preference eval.

## 새 VM에서 Codex에게 줄 짧은 프롬프트

```text
이 repo는 /home/.../sr-diffusion 의 x4 latent diffusion SR 프로젝트다.
docs/HANDOFF_KO.md 와 docs/VM_RECOVERY_KO.md 를 먼저 읽고 이어서 작업해줘.
현재 Stage4 photo100k condition-start와 Stage2/Stage3/Stage4 photo100k
degradation v2 fine-tune 및 sampled eval까지 완료됐고, Stage2 photo100k
v3 noise XL condition encoder도 80k step까지 완료됐다. Stage4 XL은 아직
시작하지 않았다. 다음은 Stage2 XL 후보(best/step72000/latest)를 같은 val
이미지에서 condition-only decoded output으로 비교한 뒤,
configs/diffusion_photo100k_xl_stage4_condition_v3.yaml 을 Stage4 v2
checkpoint에서 --partial-init으로 시작할지 결정하는 것이다.
상업적 이용은 금지이고, raw dataset은 GitHub/HF에 올리지 않는다.
```
