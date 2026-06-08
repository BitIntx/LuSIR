# 새 VM 복구 가이드

이 문서는 기존 scratch가 사라졌거나 다른 VM으로 옮길 때 현재 프로젝트를
복구하는 절차다.

## 1. Repo clone

```bash
git clone https://github.com/BitIntx/sr-diffusion
cd sr-diffusion
```

## 2. Python / PyTorch 설치

ROCm 7.2 VM:

```bash
python3 -m venv /home/$USER/venvs/rocm
source /home/$USER/venvs/rocm/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install -e .
```

CUDA/Colab은 Colab notebook 또는 해당 CUDA PyTorch wheel을 사용한다.

## 3. Hugging Face checkpoint 복구

프로토타입 추론만 필요하면:

```bash
python scripts/download_hf_checkpoints.py
```

photo100k handoff checkpoint까지 모두 받으려면:

```bash
python scripts/download_hf_checkpoints.py --preset photo100k
```

Stage2 XL 후보 checkpoint까지 받아서 새 VM에서 비교하려면:

```bash
python scripts/download_hf_checkpoints.py --preset photo100k_xl_candidates
```

최신 Stage4 XL edge artifact와 residual diagnostic/refiner probe까지 받으려면:

```bash
python scripts/download_hf_checkpoints.py --preset residual_refiner_stage2_xl_mild
```

다운로드 위치:

```text
checkpoints/
configs/
metrics/
```

주의: `--preset photo100k`는 Stage3 photo100k checkpoint가 포함되어
다운로드 용량이 크다.

`--preset photo100k_xl_candidates`는 Stage2 XL 후보 checkpoint 3개까지
포함하므로 더 크다.

`--preset residual_refiner_stage2_xl_mild`는 XL edge checkpoint, residual
refiner checkpoint, diagnostic metrics, sample grids를 포함한다.

## 4. 추론만 해보기

단일 128x128 LR 입력:

```bash
python tools/infer/infer_diffusion.py \
  --input-lr /path/to/lr_128.png \
  --output-dir outputs/demo
```

큰 LR 입력 타일 추론:

```bash
python tools/infer/infer_diffusion.py \
  --input-lr /path/to/larger_lr.png \
  --output-dir outputs/tiled_demo \
  --tile \
  --tile-overlap 32
```

Colab:

```text
https://colab.research.google.com/github/BitIntx/sr-diffusion/blob/main/notebooks/sr_diffusion_colab_demo.ipynb
```

## 5. Scratch/data 복구

기존 VM에서는 scratch root를 다음처럼 사용했다:

```text
/home/jwheojjang/scratch/sr-diffusion
```

새 VM에서 같은 경로를 쓸 수 있으면 가장 편하다. 다른 유저명이라면
`SRD_SCRATCH` 또는 `SRD_SCRATCH_PROJECT` 환경변수로 조정한다.

데이터 전체 복구:

```bash
bash scripts/recover_scratch.sh --coco-count 100000
```

이 명령은 다음을 복구한다:

```text
DIV2K
Flickr2K
COCO train2017 deterministic subset
manifest_df2k_photo.csv
manifest_photo100k.csv
```

예상 manifest:

```text
/home/.../scratch/sr-diffusion/data/manifest_photo100k.csv
photo/train: 103450
photo/val: 100
```

## 6. Training config가 기대하는 checkpoint 경로

현재 학습 config들은 기존 scratch 절대경로를 참조한다. 새 VM에서 같은
경로가 아니라면 두 방법 중 하나를 선택한다.

방법 A: config의 checkpoint path를 새 경로로 수정.

방법 B: HF에서 받은 checkpoint를 기존 구조와 같은 scratch path에 배치.

예시:

```bash
mkdir -p /home/$USER/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints
mkdir -p /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints
mkdir -p /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints
mkdir -p /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints
mkdir -p /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints
mkdir -p /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/checkpoints
mkdir -p /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition/checkpoints
mkdir -p /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints

cp checkpoints/stage1_autoencoder_best_eval_recon.pt \
  /home/$USER/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt

cp checkpoints/stage2_photo100k_b64_best_eval_latent.pt \
  /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt

cp checkpoints/stage2_photo100k_v2_b64_best_eval_latent.pt \
  /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt

cp checkpoints/stage2_photo100k_v3_noise_xl_b64_best_eval_latent.pt \
  /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/best_eval_latent.pt

cp checkpoints/stage2_photo100k_v3_noise_xl_b64_step_0072000.pt \
  /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/step_0072000.pt

cp checkpoints/stage2_photo100k_v3_noise_xl_b64_latest.pt \
  /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/latest.pt

cp checkpoints/stage3_photo100k_b32_best_eval_noise.pt \
  /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt

cp checkpoints/stage3_photo100k_v2_b32_best_eval_noise.pt \
  /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/checkpoints/best_eval_noise.pt

cp checkpoints/stage4_photo100k_condition_b32_best_eval_condition_decoded.pt \
  /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt

cp checkpoints/stage4_photo100k_condition_v2_b32_best_eval_condition_decoded.pt \
  /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints/best_eval_condition_decoded.pt
```

기존 config가 `/home/jwheojjang/...`를 하드코딩하고 있다면, 새 VM의 유저명에
맞게 config를 수정하거나 같은 경로로 symlink를 만든다.

## 7. 이어서 할 작업

현재 이어서 할 작업은 Stage2 XL condition encoder 후보를 비교한 뒤
Stage4 XL condition-start를 시작할지 결정하는 것이다. Stage4 v2 sampled
eval 재현은 baseline 확인용으로 필요할 때만 실행한다:

```bash
python tools/eval/eval_diffusion_samples.py \
  --config configs/diffusion_photo100k_b32_stage4_condition_v2.yaml \
  --checkpoint /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints/best_eval_condition_decoded.pt \
  --output-dir /home/$USER/scratch/sr-diffusion/runs/eval_diffusion_photo100k_stage4_condition_v2_val100_t25_32step \
  --split val \
  --limit 100 \
  --steps 32 \
  --seed 1337
```

그 다음:

```text
Stage2 XL best/step72000/latest condition-only decoded output 비교
Stage4 XL condition-start를 Stage4 v2 checkpoint에서 --partial-init으로 시작
denoise/sharpening A/B review against Stage3 v2, Stage4 v2, and mild baseline
cyan/green dot artifact and color/contrast overshoot mitigation experiments
```

Stage4 XL 시작 명령 예시:

```bash
python tools/train/train_diffusion.py \
  --config configs/diffusion_photo100k_xl_stage4_condition_v3.yaml \
  --init-checkpoint /home/$USER/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints/best_eval_condition_decoded.pt \
  --partial-init
```

주의: 이 명령은 Stage4 XL 학습을 시작한다. VM 인수인계 직후에는 먼저
condition encoder 후보 비교를 하는 것이 권장된다.

## 8. tmux / 모니터링

학습 실행:

```bash
tmux new-session -d -s sr_stage3_photo100k 'cd /path/to/sr-diffusion && env PYTHONUNBUFFERED=1 python tools/train/train_diffusion.py --config configs/diffusion_photo100k_b32.yaml > /path/to/train_tmux.log 2>&1'
```

로그:

```bash
tail -f /path/to/train_tmux.log
```

ROCm GPU:

```bash
watch -n 1 rocm-smi --showuse --showmemuse --showtemp --showpower
```

## 9. 업로드 정책

- GitHub에는 code/docs/config만 올린다.
- HF에는 selected checkpoints/configs/metrics만 올린다.
- raw datasets, private validation images, W&B local cache는 올리지 않는다.
