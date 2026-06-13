ARG BASE_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
ARG TORCH_VERSION="2.12.0"
ARG TORCHVISION_VERSION="0.27.0"

ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SRD_SCRATCH=/home/ubuntu/scratch \
    SRD_SCRATCH_PROJECT=/home/ubuntu/scratch/sr-diffusion

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "${PYTORCH_INDEX_URL}" ]; then \
         python -m pip install \
           "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
           --index-url "${PYTORCH_INDEX_URL}"; \
       else \
         python -m pip install \
           "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
       fi

WORKDIR /workspace/LuSIR
COPY . /workspace/LuSIR
RUN python -m pip install pytest \
    && python -m pip install -e .

CMD ["bash"]
