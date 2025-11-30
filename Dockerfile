# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    wget && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt/hunyuan

COPY setup_hunyuan.sh infer.sh handler.py ./

ARG HF_TOKEN=""
ARG HUGGINGFACE_TOKEN=""
ARG HY_SIGLIP_TOKEN=""
ARG RUNPOD_HF_TOKEN=""
ENV HY_TARGET_DIR=/opt/hunyuan/HunyuanVideo-1.5 \
    HY_VENV_DIR=/opt/hunyuan/.venv \
    HY_MODEL_DIR=/opt/hunyuan/HunyuanVideo-1.5/ckpts \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN chmod +x setup_hunyuan.sh infer.sh && \
    SIGLIP_TOKEN="${HY_SIGLIP_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${RUNPOD_HF_TOKEN}}}}" && \
    if [[ -z "${SIGLIP_TOKEN}" ]]; then \
        echo "HY_SIGLIP_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN / RUNPOD_HF_TOKEN must be provided at build time." >&2; \
        exit 1; \
    fi && \
    HY_SIGLIP_TOKEN="${SIGLIP_TOKEN}" ./setup_hunyuan.sh && \
    rm -rf /root/.cache

ENV PATH="/opt/hunyuan/.venv/bin:${PATH}" \
    HY_REPO_DIR=/opt/hunyuan/HunyuanVideo-1.5 \
    HY_MODEL_PATH=/opt/hunyuan/HunyuanVideo-1.5/ckpts \
    PYTHONPATH="/opt/hunyuan/HunyuanVideo-1.5:${PYTHONPATH}"

RUN python -m pip install --upgrade pip && \
    pip install runpod

WORKDIR /opt/hunyuan/HunyuanVideo-1.5

CMD ["python", "/opt/hunyuan/handler.py"]
