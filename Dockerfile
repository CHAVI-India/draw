# DRAW GPU image. Runs the continuous pipeline (or any `draw` command) with CUDA.
#
# Model weights and DICOM are NOT baked in — they are mounted as volumes (see
# docker-compose.yml), so this image stays reusable across sites and weight updates,
# and no PHI ever lands in an image layer.
#
# Base: CUDA 11.8 runtime matches the historical cu118 torch wheels. The host needs an
# NVIDIA driver new enough for CUDA 11.8 (>= 520) and the nvidia-container-toolkit.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    # nnU-Net data dirs (overridable); mounted from the host in compose.
    nnUNet_raw=/app/data/nnUNet_raw \
    nnUNet_preprocessed=/app/data/nnUNet_preprocessed \
    nnUNet_results=/app/data/nnUNet_results

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 1) Install the right CUDA torch wheel FIRST so the `gpu` extra's unpinned torch
#    resolves to the cu118 build rather than a default/CPU wheel.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cu118 \
        torch torchvision torchaudio

# 2) Dependency layer: copy only metadata so this caches across code changes.
COPY pyproject.toml uv.lock README.md ./
# Source is needed because the project is an installable package (hatchling).
COPY src ./src
RUN uv pip install --system ".[conversion,pipeline,gpu]"

# 3) App config that ships with the image (small, versioned). NOT weights, NOT PHI.
COPY config_yaml ./config_yaml

# Console script `draw` is now on PATH. Default to the continuous pipeline; override
# the command to run one-shot `draw predict ...` etc.
ENTRYPOINT ["draw"]
CMD ["start-pipeline"]
