# LuSIR Docker 사용 가이드

Docker는 LuSIR 코드, Python, PyTorch, CUDA user-space library를 하나의
재현 가능한 image로 묶는다. 새 VM에서는 호스트 NVIDIA driver와 Docker만
준비하고 같은 image를 실행하면 Python 환경을 다시 맞추는 작업을 줄일 수 있다.

Docker는 모델 품질이나 학습 속도를 직접 개선하지 않는다. 데이터, checkpoint,
W&B 로그는 image에 넣지 않고 호스트 scratch를 container에 연결한다.

## 구성

```text
Dockerfile               CUDA/Python/PyTorch/LuSIR image
.dockerignore            dataset/checkpoint/output을 build context에서 제외
scripts/docker_lusir.sh  build, GPU 확인, shell, test, 임의 명령 실행
```

기본 CUDA image는 `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`다. 호스트
CUDA Toolkit 버전은 같을 필요가 없지만, NVIDIA driver는 container CUDA를
지원할 만큼 최신이어야 한다. 필요하면 `LUSIR_BASE_IMAGE`로 바꾼다.
기본 PyTorch/Torchvision은 현재 검증 환경과 같은 `2.12.0+cu130` /
`0.27.0+cu130` wheel로 고정한다.

## 새 Ubuntu GPU VM 준비

호스트에서 먼저 다음이 동작해야 한다.

```bash
nvidia-smi
docker --version
nvidia-container-cli --version
```

Docker Engine과 NVIDIA Container Toolkit은 각 공식 문서대로 설치한다.

- Docker Engine: <https://docs.docker.com/engine/install/ubuntu/>
- NVIDIA Container Toolkit:
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

NVIDIA Container Toolkit 설치 후 Docker runtime을 연결한다.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

현재 사용자를 `docker` group에 추가했다면 재로그인 후 사용한다. group을
사용하지 않으면 아래 `docker` 또는 wrapper 명령 앞에 `sudo`가 필요하다.

## Image 빌드와 확인

```bash
cd ~/LuSIR
bash scripts/docker_lusir.sh build
bash scripts/docker_lusir.sh gpu
bash scripts/docker_lusir.sh test
```

다른 PyTorch wheel index가 필요하면 build 때만 지정한다. 이 경우 Dockerfile의
`TORCH_VERSION`과 `TORCHVISION_VERSION` build arg도 호환되는 값으로 바꿔야 한다.

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
TORCH_VERSION=2.12.0 TORCHVISION_VERSION=0.27.0 \
  bash scripts/docker_lusir.sh build
```

`gpu` 출력에서 `cuda_available True`와 GPU 개수/이름이 보이면 container GPU
연결이 완료된 것이다. GPU가 연결되지 않으면 이 명령은 실패한다.

### 2026-06-13 실제 검증

현재 L40S Ubuntu 24.04 VM에서 다음을 실제로 완료했다.

```text
Docker Engine:            29.5.3
NVIDIA Container Toolkit: 1.19.1
image:                    lusir:dev
container CUDA:           13.0
PyTorch:                  2.12.0+cu130
Torchvision:              0.27.0+cu130
cuDNN:                    92000
GPU:                      NVIDIA L40S
container pytest:         55 passed
```

`scripts/docker_lusir.sh run`으로 detail-need mask GPU 진단도 실행해 checkpoint,
host scratch mount, repo bind mount가 함께 동작하는 것을 확인했다. 새로 Docker
group에 추가된 사용자는 재로그인 전까지 `sg docker -c '명령'`을 임시로 쓸 수
있다.

## Shell과 명령 실행

interactive shell:

```bash
bash scripts/docker_lusir.sh shell
```

checkpoint 다운로드:

```bash
bash scripts/docker_lusir.sh run \
  python scripts/download_hf_checkpoints.py --preset detail_branch_v1d
```

평가 예시:

```bash
bash scripts/docker_lusir.sh run \
  python tools/eval/run_sr_benchmark.py --help
```

학습 예시:

```bash
bash scripts/docker_lusir.sh run \
  python -u tools/train/train_latent_pretrain.py \
  --config configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml
```

DDP는 container 안에서도 기존처럼 `torchrun`을 사용한다. wrapper는 모든 visible
GPU를 container에 전달한다.

```bash
bash scripts/docker_lusir.sh run \
  torchrun --standalone --nproc_per_node=2 \
  tools/train/train_latent_pretrain.py \
  --config configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml
```

## Mount와 인증

wrapper의 기본 mount:

```text
현재 repo             -> /workspace/LuSIR
$HOME/scratch         -> /home/ubuntu/scratch
$HOME/.cache/lusir-docker-home -> /home/lusir
```

기존 config의 `/home/ubuntu/scratch/sr-diffusion/...` 절대경로가 container에서도
그대로 동작한다. 다른 scratch를 쓰려면:

```bash
LUSIR_SCRATCH=/mnt/training-scratch bash scripts/docker_lusir.sh shell
```

HF와 W&B 인증은 호스트 환경변수를 container에 전달한다.

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
bash scripts/docker_lusir.sh shell
```

토큰을 image에 `COPY`하거나 Dockerfile에 기록하지 않는다.

## 주의사항

- Docker image에는 raw dataset, checkpoint, outputs를 포함하지 않는다.
- 현재 repo를 bind mount하므로 container 내부 코드 수정은 호스트에도 반영된다.
- 일반 사용자로 wrapper를 실행하면 같은 UID/GID를 사용해 root 소유 결과 파일을
  만들지 않는다. `sudo bash scripts/docker_lusir.sh ...`로 실행하면 결과도 root
  소유가 될 수 있으므로 가능하면 Docker group 설정 후 일반 사용자로 실행한다.
- `--ipc=host`는 PyTorch DataLoader/DDP shared-memory 병목을 피하기 위한 설정이다.
- Docker build만 성공했다고 GPU가 연결된 것은 아니다. 반드시 `gpu` 명령을 실행한다.
- 첫 build는 CUDA/PyTorch layer와 build cache 때문에 디스크를 크게 사용한다.
  image 검증 후 build cache만 비우려면 `docker builder prune -f`를 사용한다.
- ROCm VM은 이 CUDA Dockerfile 대상이 아니다. 기존 ROCm venv 절차를 사용한다.
