#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${LUSIR_REPO_URL:-https://github.com/BitIntx/LuSIR.git}"
BRANCH="${LUSIR_BRANCH:-main}"
MODE="${LUSIR_BENCH_MODE:-quick}"
STEPS="${LUSIR_BENCH_STEPS:-250}"
WARMUP_STEP="${LUSIR_BENCH_WARMUP_STEP:-50}"
SYNTH_COUNT="${LUSIR_SYNTH_COUNT:-512}"
SYNTH_SIZE="${LUSIR_SYNTH_SIZE:-512}"
NUM_WORKERS="${LUSIR_NUM_WORKERS:-4}"
SKIP_APT="${LUSIR_SKIP_APT:-0}"

USER_NAME="$(id -un 2>/dev/null || echo root)"
HOME_DIR="${HOME:-}"
if [[ -z "${HOME_DIR}" || ! -d "${HOME_DIR}" ]]; then
  HOME_DIR="$(getent passwd "${USER_NAME}" | cut -d: -f6 || true)"
fi
if [[ -z "${HOME_DIR}" || ! -d "${HOME_DIR}" ]]; then
  HOME_DIR="/root"
fi

WORKDIR="${LUSIR_WORKDIR:-${HOME_DIR}/LuSIR}"
SCRATCH="${LUSIR_SCRATCH:-${HOME_DIR}/scratch}"
SCRATCH_PROJECT="${LUSIR_SCRATCH_PROJECT:-${SCRATCH}/sr-diffusion}"
VENV="${LUSIR_VENV:-${HOME_DIR}/venvs/lusir-bench}"

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_stage2_speed_benchmark.sh

One-command bootstrap for the LuSIR Stage 2 speed benchmark on a new VM.

Default mode is quick:
  - clone/update LuSIR
  - create a Python venv
  - install PyTorch + LuSIR
  - download only the two required checkpoints
  - generate a synthetic 512px dataset
  - run the 250-step Stage 2 speed benchmark

Environment:
  LUSIR_BENCH_MODE=quick|full   Default: quick. full downloads photo100k+LSDIR.
  LUSIR_WORKDIR=PATH            Default: $HOME/LuSIR
  LUSIR_SCRATCH=PATH            Default: $HOME/scratch
  LUSIR_VENV=PATH               Default: $HOME/venvs/lusir-bench
  LUSIR_BENCH_STEPS=N           Default: 250
  LUSIR_BENCH_WARMUP_STEP=N     Default: 50
  LUSIR_SYNTH_COUNT=N           Default: 512, quick mode only
  LUSIR_NUM_WORKERS=N           Default: 4
  PYTORCH_INDEX_URL=URL         Optional pip index-url for torch/torchvision.
  LUSIR_SKIP_APT=1              Skip apt-get package install.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${MODE}" != "quick" && "${MODE}" != "full" ]]; then
  echo "Unsupported LUSIR_BENCH_MODE=${MODE}. Use quick or full." >&2
  exit 2
fi

run_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Need root or sudo to install system packages." >&2
    exit 1
  fi
}

echo "== LuSIR Stage 2 speed benchmark bootstrap =="
echo "mode=${MODE}"
echo "user=${USER_NAME}"
echo "workdir=${WORKDIR}"
echo "scratch=${SCRATCH}"
echo "venv=${VENV}"

if [[ "${SKIP_APT}" != "1" && -f /etc/debian_version ]] && command -v apt-get >/dev/null 2>&1; then
  echo "== install system packages =="
  run_sudo env DEBIAN_FRONTEND=noninteractive apt-get update
  run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-venv
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required. Install git or rerun without LUSIR_SKIP_APT=1." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install python3 or rerun without LUSIR_SKIP_APT=1." >&2
  exit 1
fi

echo "== clone/update LuSIR =="
if [[ -d "${WORKDIR}/.git" ]]; then
  git -C "${WORKDIR}" fetch origin "${BRANCH}"
  git -C "${WORKDIR}" checkout "${BRANCH}"
  git -C "${WORKDIR}" pull --ff-only origin "${BRANCH}"
else
  mkdir -p "$(dirname "${WORKDIR}")"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
fi
cd "${WORKDIR}"

echo "== create Python venv =="
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip

echo "== install PyTorch and LuSIR =="
if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
  python -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"
else
  python -m pip install torch torchvision
fi
python -m pip install -e .

echo "== Python/Torch environment =="
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "CUDA GPU is not available to PyTorch; cannot run the Stage 2 GPU benchmark." >&2
  exit 1
fi

mkdir -p \
  "${SCRATCH_PROJECT}/configs" \
  "${SCRATCH_PROJECT}/data" \
  "${SCRATCH_PROJECT}/datasets/photo" \
  "${SCRATCH_PROJECT}/runs/autoencoder_photo10k_b16_eval_online/checkpoints" \
  "${SCRATCH_PROJECT}/runs/benchmarks"

echo "== download minimal checkpoints =="
python scripts/download_hf_checkpoints.py \
  --file checkpoints/stage1_autoencoder_best_eval_recon.pt \
  --file checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt
cp -f checkpoints/stage1_autoencoder_best_eval_recon.pt \
  "${SCRATCH_PROJECT}/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt"

if [[ "${MODE}" == "quick" ]]; then
  echo "== create synthetic benchmark dataset =="
  SYNTH_DIR="${SCRATCH_PROJECT}/data/stage2_speed_synth_${SYNTH_COUNT}"
  python scripts/make_toy_dataset.py \
    --output "${SYNTH_DIR}" \
    --count "${SYNTH_COUNT}" \
    --size "${SYNTH_SIZE}"
  MANIFEST="${SYNTH_DIR}/manifest.csv"
else
  echo "== recover full photo100k + LSDIR benchmark dataset =="
  python scripts/download_div2k.py \
    --output-dir "${SCRATCH_PROJECT}/datasets/photo/div2k" \
    --manifest "${SCRATCH_PROJECT}/data/manifest_div2k_photo.csv"
  python scripts/download_flickr2k.py \
    --output-dir "${SCRATCH_PROJECT}/datasets/photo/flickr2k" \
    --manifest "${SCRATCH_PROJECT}/data/manifest_flickr2k_photo.csv"
  python scripts/merge_manifests.py \
    --inputs \
      "${SCRATCH_PROJECT}/data/manifest_div2k_photo.csv" \
      "${SCRATCH_PROJECT}/data/manifest_flickr2k_photo.csv" \
    --output "${SCRATCH_PROJECT}/data/manifest_df2k_photo.csv"
  python scripts/download_coco2017.py \
    --output-dir "${SCRATCH_PROJECT}/datasets/photo/coco2017" \
    --manifest "${SCRATCH_PROJECT}/data/manifest_coco2017_photo100k.csv" \
    --target-count 100000 \
    --min-size 320
  python scripts/merge_manifests.py \
    --inputs \
      "${SCRATCH_PROJECT}/data/manifest_df2k_photo.csv" \
      "${SCRATCH_PROJECT}/data/manifest_coco2017_photo100k.csv" \
    --output "${SCRATCH_PROJECT}/data/manifest_photo100k.csv"
  python scripts/download_lsdir_hf.py \
    --output-dir "${SCRATCH_PROJECT}/datasets/photo/lsdir" \
    --manifest "${SCRATCH_PROJECT}/data/manifest_lsdir_photo.csv" \
    --target-count 30000
  python scripts/merge_manifests.py \
    --inputs \
      "${SCRATCH_PROJECT}/data/manifest_photo100k.csv" \
      "${SCRATCH_PROJECT}/data/manifest_lsdir_photo.csv" \
    --output "${SCRATCH_PROJECT}/data/manifest_photo130k_lsdir.csv"
  MANIFEST="${SCRATCH_PROJECT}/data/manifest_photo130k_lsdir.csv"
fi

BENCH_CONFIG="${SCRATCH_PROJECT}/configs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_benchmark.yaml"
echo "== write portable benchmark config =="
BENCH_CONFIG="${BENCH_CONFIG}" \
MANIFEST="${MANIFEST}" \
SCRATCH_PROJECT="${SCRATCH_PROJECT}" \
NUM_WORKERS="${NUM_WORKERS}" \
python - <<'PY'
import os
from pathlib import Path

import yaml

source = Path("configs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue.yaml")
target = Path(os.environ["BENCH_CONFIG"])
manifest = Path(os.environ["MANIFEST"]).resolve()
scratch_project = Path(os.environ["SCRATCH_PROJECT"]).resolve()
num_workers = int(os.environ["NUM_WORKERS"])

with source.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["project"]["output_dir"] = str(scratch_project / "runs" / "stage2_speed_benchmark")
config["autoencoder"]["checkpoint"] = str(
    scratch_project / "runs" / "autoencoder_photo10k_b16_eval_online" / "checkpoints" / "best_eval_recon.pt"
)
config["data"]["manifest"] = str(manifest)
config["data"]["num_workers"] = num_workers
config.setdefault("eval", {})["num_workers"] = num_workers
config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False

target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print(f"wrote {target}")
PY

echo "== run Stage 2 speed benchmark =="
LOG_PATH="${SCRATCH_PROJECT}/runs/benchmarks/stage2_speed_${MODE}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
python tools/bench/benchmark_stage2_speed.py \
  --config "${BENCH_CONFIG}" \
  --python "${VENV}/bin/python" \
  --steps "${STEPS}" \
  --warmup-step "${WARMUP_STEP}" \
  --log-path "${LOG_PATH}"

cat <<EOF

Benchmark complete.

Mode: ${MODE}
Log: ${LOG_PATH}

To run again:
  cd ${WORKDIR}
  source ${VENV}/bin/activate
  python tools/bench/benchmark_stage2_speed.py --config ${BENCH_CONFIG} --python ${VENV}/bin/python

EOF
