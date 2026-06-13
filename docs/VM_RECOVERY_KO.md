# LuSIR 새 VM 복구 가이드

이 문서는 기존 scratch가 사라졌거나 다른 VM으로 옮길 때 LuSIR 프로젝트를
복구하는 절차다.

GitHub/HF repo id는 LuSIR로 바뀌었다. 기존 scratch 경로와 일부 과거 실험
artifact 경로는 아직 `sr-diffusion` 이름을 쓴다.

## 1. Repo clone

```bash
git clone https://github.com/BitIntx/LuSIR
cd LuSIR
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

완료된 dual-context LSDIR Stage2 best98000 checkpoint와 contact sheet까지
받으려면:

```bash
python scripts/download_hf_checkpoints.py --preset stage2_photo130k_lsdir_dual
```

선택된 high-frequency detail branch v1d best99500 checkpoint와 일반/strict-bicubic
review grid까지
받으려면:

```bash
python scripts/download_hf_checkpoints.py --preset detail_branch_v1d
```

V1d step `99500`은 최신 public detail artifact다. V1b도 이전 비교용
`detail_branch_v1b` preset으로 보존되어 있다.

정식 x4 benchmark dataset과 manifest를 scratch에 복구하려면:

```bash
python scripts/download_sr_benchmarks.py
```

protocol과 재현 명령은 `docs/SR_BENCHMARK.md`에 있다. dataset 원본은 GitHub나
HF artifact에 재배포하지 않는다.

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
https://colab.research.google.com/github/BitIntx/LuSIR/blob/main/notebooks/sr_diffusion_colab_demo.ipynb
```

현재 Colab은 Gradio WebUI를 실행한다. 유저 업로드가 기본이고 correction strength,
tile overlap, tile batch size, diffusion steps는 slider로 조정한다. 결과 화면은
bicubic/Stage 2 condition/Input LR nearest 중 하나와 SR output을 before/after
slider로 비교한다. residual refiner v2가 public 기본값이며, 선택된 detail branch
v1d는 단일 이미지/tiled inference 연구 옵션으로 선택할 수 있다.

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

현재 dual-multiscale Stage 2 장기 run용 LSDIR 30k subset과 병합 manifest:

```bash
python scripts/download_lsdir_hf.py \
  --output-dir /home/$USER/scratch/sr-diffusion/datasets/photo/lsdir \
  --manifest /home/$USER/scratch/sr-diffusion/data/manifest_lsdir_photo.csv \
  --target-count 30000

python scripts/merge_manifests.py \
  --inputs /home/$USER/scratch/sr-diffusion/data/manifest_photo100k.csv \
           /home/$USER/scratch/sr-diffusion/data/manifest_lsdir_photo.csv \
  --output /home/$USER/scratch/sr-diffusion/data/manifest_photo130k_lsdir.csv
```

예상 병합 결과는 `133450` unique train + `100` val이다. 다운로더는 parquet
shard 하나씩만 보관하고 추출 후 삭제하므로 전체 parquet 복제본이 필요하지
않다. raw LSDIR 데이터는 GitHub/HF에 업로드하지 않는다.

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

3.02M detail branch v1d capacity run은 완료됐다.

```text
config:    configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml
W&B:       https://wandb.ai/jwheo/LuSIR/runs/ctg4r7n9
completed: 100086 micro-steps = exactly 3 epoch
selected:  step 99500
HF:        checkpoints/detail_branch_v1d_deep3m_photo130k_lsdir_best99500.pt
```

선택 step `99500`:

```text
photo_detail_mix PSNR delta:      +0.1646 dB
photo_detail_mix mean PSNR delta: +0.1888 dB
photo_detail_mix SSIM delta:      +0.00647
wins:                             99/100
strict-bicubic five-crop:         31.9513 dB
```

다음은 정식 DIV2K/Set5/Set14/Urban100 benchmark와 public baseline/blind human
비교다. 같은 objective의 추가 continuation이나 단순 capacity 증가는 우선하지
않는다. 현재 public Colab 기본 경로는 여전히 residual refiner v2이며, v1d는
선택 가능한 연구 비교 옵션이다.

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
