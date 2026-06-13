#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LUSIR_DOCKER_IMAGE:-lusir:dev}"
SCRATCH="${LUSIR_SCRATCH:-${HOME}/scratch}"
CONTAINER_HOME="${LUSIR_DOCKER_HOME:-${HOME}/.cache/lusir-docker-home}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/docker_lusir.sh build
  bash scripts/docker_lusir.sh gpu
  bash scripts/docker_lusir.sh shell
  bash scripts/docker_lusir.sh test
  bash scripts/docker_lusir.sh run COMMAND [ARG...]

Environment:
  LUSIR_DOCKER_IMAGE=lusir:dev
  LUSIR_SCRATCH=$HOME/scratch
  LUSIR_DOCKER_HOME=$HOME/.cache/lusir-docker-home
  LUSIR_BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
  PYTORCH_INDEX_URL=...  # optional PyTorch wheel index override during build
  TORCH_VERSION=...      # optional torch version override during build
  TORCHVISION_VERSION=... # optional torchvision version override during build

The runner exposes every visible NVIDIA GPU, bind-mounts the repo, and maps
$LUSIR_SCRATCH to /home/ubuntu/scratch so existing LuSIR configs keep working.
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed; see docs/DOCKER_KO.md" >&2
    exit 1
  fi
}

build_image() {
  local base_image="${LUSIR_BASE_IMAGE:-nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04}"
  local args=(--build-arg "BASE_IMAGE=${base_image}")
  if [[ -n "${PYTORCH_INDEX_URL:-}" ]]; then
    args+=(--build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}")
  fi
  if [[ -n "${TORCH_VERSION:-}" ]]; then
    args+=(--build-arg "TORCH_VERSION=${TORCH_VERSION}")
  fi
  if [[ -n "${TORCHVISION_VERSION:-}" ]]; then
    args+=(--build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}")
  fi
  docker build "${args[@]}" -t "${IMAGE}" "${ROOT}"
}

run_container() {
  mkdir -p "${SCRATCH}" "${CONTAINER_HOME}"
  local tty_args=()
  if [[ -t 0 && -t 1 ]]; then
    tty_args=(-it)
  fi
  docker run --rm "${tty_args[@]}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    --env HOME=/home/lusir \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_DIR=/home/ubuntu/scratch/sr-diffusion/wandb \
    --volume "${ROOT}:/workspace/LuSIR" \
    --volume "${SCRATCH}:/home/ubuntu/scratch" \
    --volume "${CONTAINER_HOME}:/home/lusir" \
    --workdir /workspace/LuSIR \
    "${IMAGE}" "$@"
}

command="${1:-}"
case "${command}" in
  build)
    require_docker
    build_image
    ;;
  gpu)
    require_docker
    run_container python -c \
      'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cudnn", torch.backends.cudnn.version()); print("cuda_available", torch.cuda.is_available()); print("gpus", torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]; assert torch.cuda.is_available(), "NVIDIA GPU is not available inside the container"'
    ;;
  shell)
    require_docker
    run_container bash
    ;;
  test)
    require_docker
    run_container python -m pytest -q
    ;;
  run)
    require_docker
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "run requires a command" >&2
      usage
      exit 1
    fi
    run_container "$@"
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "unknown command: ${command}" >&2
    usage
    exit 1
    ;;
esac
