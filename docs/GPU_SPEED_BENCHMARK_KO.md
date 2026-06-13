# LuSIR GPU 속도 벤치마크

이 문서는 다른 VM/GPU에서 현재 Stage 2 clean-bicubic continuation의 학습
속도를 같은 조건으로 비교하기 위한 절차다.

기준 run은 L40S 단일 GPU, PyTorch `2.12.0+cu130`, cuDNN `9.20.0` 환경에서
`1.15 micro-step/s`로 안정화됐다. 여기서 step은 `train_latent_pretrain.py`
로그의 micro-batch step이다. 현재 config는 `batch_size: 8`,
`grad_accum_steps: 4`이므로 optimizer update 속도는 `micro-step/s / 4`이고,
micro-batch 이미지 처리량은 `micro-step/s * 8`이다.

## 전제 조건

repo와 Python 환경:

```bash
git clone https://github.com/BitIntx/LuSIR
cd LuSIR
python3 -m venv /home/$USER/venvs/cuda
source /home/$USER/venvs/cuda/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -e .
```

Stage 2 benchmark에 필요한 public checkpoint:

```bash
python scripts/download_hf_checkpoints.py --preset stage2_photo130k_lsdir_dual
```

config가 Stage 1 autoencoder를 scratch run 경로에서 찾으므로, HF에서 받은
checkpoint를 같은 위치에 둔다:

```bash
mkdir -p /home/$USER/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints
cp checkpoints/stage1_autoencoder_best_eval_recon.pt \
  /home/$USER/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt
```

Stage 2 init checkpoint는 repo의 `checkpoints/` 아래 상대경로를 사용하므로
위 HF preset 다운로드만 되어 있으면 된다:

```text
checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
```

실제 end-to-end benchmark는 dataloader까지 포함하므로
`manifest_photo130k_lsdir.csv`와 이미지 파일이 필요하다. 복구 절차:

```bash
bash scripts/recover_scratch.sh --coco-count 100000

python scripts/download_lsdir_hf.py \
  --output-dir /home/$USER/scratch/sr-diffusion/datasets/photo/lsdir \
  --manifest /home/$USER/scratch/sr-diffusion/data/manifest_lsdir_photo.csv \
  --target-count 30000

python scripts/merge_manifests.py \
  --inputs /home/$USER/scratch/sr-diffusion/data/manifest_photo100k.csv \
           /home/$USER/scratch/sr-diffusion/data/manifest_lsdir_photo.csv \
  --output /home/$USER/scratch/sr-diffusion/data/manifest_photo130k_lsdir.csv
```

AWS Ubuntu image처럼 user가 `ubuntu`이면 config의 절대 scratch 경로와 바로
맞는다. user명이 다르면 config 안의 `/home/ubuntu/scratch`를 현재 scratch
경로로 바꾸거나 같은 경로를 symlink로 맞춘다.

## Stage 2 학습 속도 벤치마크

W&B를 끄고 250 micro-step만 실행한다. 첫 50 step은 warmup으로 제외하고
나머지 `steps_per_sec`의 평균/중앙값을 출력한다. 임시 output directory는
checkpoint 때문에 수 GB가 될 수 있으므로 기본적으로 자동 삭제하고 log만
남긴다.

```bash
source /home/$USER/venvs/cuda/bin/activate
python tools/bench/benchmark_stage2_speed.py
```

다른 Python/venv를 명시하려면:

```bash
python tools/bench/benchmark_stage2_speed.py \
  --python /home/$USER/venvs/cuda/bin/python \
  --steps 250 \
  --warmup-step 50
```

결과 예시:

```text
throughput_summary:
  stable_points: 9
  mean_steps_per_sec: 1.1500
  median_steps_per_sec: 1.1500
  min_steps_per_sec: 1.1500
  max_steps_per_sec: 1.1500
  images_per_sec_microbatch: 9.2000
  optimizer_updates_per_sec: 0.2875
```

현재 L40S 기준값:

```text
torch 2.12.0+cu130, CUDA runtime 13.0, cuDNN 9.20.0: 1.1500 step/s
torch 2.12.0+cu132, CUDA runtime 13.2, cuDNN 9.20.0: 1.1500 step/s
torch 2.12.0+cu132, CUDA runtime 13.2, cuDNN 9.23.1: 1.1489 step/s
```

따라서 현재 기준으로는 cu132 또는 최신 cuDNN만 올려도 속도 이득은 없었다.

## Dataloader 병목 확인

학습 속도가 낮게 나오면 dataloader만 따로 잰다:

```bash
python tools/bench/benchmark_dataloader.py --workers 0 2 4
```

`persistent_workers`와 prefetch도 확인하려면:

```bash
python tools/bench/benchmark_dataloader.py \
  --workers 4 \
  --persistent-workers \
  --prefetch-factor 2
```

현재 L40S VM의 dataloader-only 기준값:

```text
workers=0: 5.95 batch/s
workers=2: 9.98 batch/s
workers=4: 12.00 batch/s
workers=4 persistent_workers=True prefetch_factor=2: 12.43 batch/s
```

학습은 약 `1.15 batch/s`만 소비하므로 이 VM에서는 dataloader 병목이 아니다.
다른 VM에서 dataloader-only가 학습 속도와 비슷하거나 더 낮으면 local disk,
network filesystem, CPU worker 수, 이미지 decode 성능을 먼저 의심한다.

## 실제 장기 학습 시작

benchmark 후 실제 run은 W&B를 켜고 tmux에서 실행한다:

```bash
tmux new-session -d -s stage2-bicubic-fidelity \
  "cd $PWD && source /home/$USER/venvs/cuda/bin/activate && \
   python -u tools/train/train_latent_pretrain.py \
     --config configs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue.yaml \
     2>&1 | tee /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue/train_console.log"
```

로그 확인:

```bash
tail -f /home/$USER/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue/train_console.log
```

GPU 상태:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits
```
