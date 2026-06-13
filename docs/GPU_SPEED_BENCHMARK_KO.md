# LuSIR GPU Quick Benchmark

새 VM이 LuSIR Stage 2 학습에 적합한지 빠르게 판단하기 위한 quick benchmark다.
기본 실행은 PyTorch가 볼 수 있는 모든 CUDA GPU를 자동으로 사용하며, GPU가
2장 이상이면 실제 `torchrun` DDP 경로를 탄다. 1장이면 단일 GPU 학습 경로로
실행된다.

## One Command

일반 Ubuntu user:

```bash
curl -fsSL https://raw.githubusercontent.com/BitIntx/LuSIR/main/scripts/bootstrap_stage2_speed_benchmark.sh | bash
```

root 또는 sudo 환경:

```bash
curl -fsSL https://raw.githubusercontent.com/BitIntx/LuSIR/main/scripts/bootstrap_stage2_speed_benchmark.sh | sudo bash
```

이 한 줄은 repo clone/update, venv 생성, PyTorch/LuSIR 설치, 필요한 checkpoint
2개 다운로드, synthetic 512px dataset 생성, DDP quick benchmark 실행까지
처리한다.

## 옵션

기본값은 모든 visible CUDA GPU 사용이다. 단일 GPU만 재고 싶으면:

```bash
curl -fsSL https://raw.githubusercontent.com/BitIntx/LuSIR/main/scripts/bootstrap_stage2_speed_benchmark.sh \
  | env LUSIR_NPROC_PER_NODE=1 bash
```

더 길게 재려면:

```bash
curl -fsSL https://raw.githubusercontent.com/BitIntx/LuSIR/main/scripts/bootstrap_stage2_speed_benchmark.sh \
  | env LUSIR_BENCH_STEPS=500 bash
```

주요 환경변수:

```bash
LUSIR_NPROC_PER_NODE=auto      # auto = 모든 visible CUDA GPU
LUSIR_BENCH_STEPS=250
LUSIR_BENCH_WARMUP_STEP=50
LUSIR_BENCH_BATCH_SIZE=auto    # VRAM에 따라 8/4/2/1 자동 선택
LUSIR_BENCH_GRAD_ACCUM_STEPS=auto
LUSIR_WORKDIR=$HOME/LuSIR
LUSIR_SCRATCH=$HOME/scratch
LUSIR_VENV=$HOME/venvs/lusir-bench
PYTORCH_INDEX_URL=...          # 특정 PyTorch wheel index를 강제할 때
```

## VRAM

현재 실제 장기 학습 config는 L40S 48GB에서 `batch_size=8`로 약 `37.8GB`
VRAM을 사용한다. 그래서 24GB급 GPU에서는 기본 학습 설정 그대로는 OOM이 날
가능성이 높다.

one-command quick benchmark는 VRAM을 보고 batch size를 자동으로 낮춘다:

```text
>=44GB: batch_size=8, grad_accum=4
>=28GB: batch_size=4, grad_accum=8
>=18GB: batch_size=2, grad_accum=16
<18GB:  batch_size=1, grad_accum=32
```

즉 48GB GPU가 아니어도 benchmark는 시도할 수 있다. 다만 batch가 작아지면
속도 수치는 `batch_size=8` 기준 L40S와 직접 비교하면 안 되고, 결과 블록의
`global images/s`, `updates/s`, `eff_batch`를 같이 봐야 한다. `batch=1`에서도
OOM이 나면 그 VM은 현재 Stage 2 XL 학습용으로는 맞지 않다.

강제로 batch를 지정하려면:

```bash
curl -fsSL https://raw.githubusercontent.com/BitIntx/LuSIR/main/scripts/bootstrap_stage2_speed_benchmark.sh \
  | env LUSIR_BENCH_BATCH_SIZE=2 LUSIR_BENCH_GRAD_ACCUM_STEPS=16 bash
```

## 결과 해석

마지막에 색상 있는 결과 블록이 나온다:

```text
RESULT LuSIR Stage 2 Quick Benchmark
  mode        DDP
  world_size  2
  mean        2.08 step/s
  median      2.09 step/s
  images      33.28 img/s global
  local       16.64 img/s per GPU
  updates     0.52 optimizer updates/s
  eff_batch   64
```

여기서 `step/s`는 `train_latent_pretrain.py`의 global micro-step/s다. config의
기본값은 `batch_size=8`, `grad_accum_steps=4`이므로:

```text
global images/s = step/s * 8 * world_size
optimizer updates/s = step/s / 4
effective batch = 8 * 4 * world_size
```

현재 단일 L40S 기준 quick/real 학습 속도는 warmup 이후 약 `1.15 step/s`다.
2장 DDP VM이면 통신 overhead 때문에 완전 2배는 아니어도, global images/s와
optimizer updates/s가 단일 GPU보다 충분히 올라가는지를 보면 된다.

## Manual Rerun

bootstrap 이후 같은 VM에서 다시 돌리려면:

```bash
cd ~/LuSIR
source ~/venvs/lusir-bench/bin/activate
python tools/bench/benchmark_stage2_speed.py \
  --config ~/scratch/sr-diffusion/configs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_benchmark.yaml \
  --python ~/venvs/lusir-bench/bin/python \
  --nproc-per-node auto \
  --color always
```
