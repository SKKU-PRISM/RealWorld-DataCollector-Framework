# =============================================================================
# ACS Docker Image: AutoDataCollector + RoboBridge 통합 환경
#
# 빌드:   docker build -t acs .
# 실행:   docker run --gpus all -it acs bash
# =============================================================================

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ----- 시스템 패키지 -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libusb-1.0-0 libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ----- Conda (pinocchio, hpp-fcl) -----
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -y -n base python=3.10 pinocchio hpp-fcl meshcat-python \
    && conda clean -afy

# ----- PyTorch (CUDA 12.4) + flash-attn -----
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null \
    || pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 2>/dev/null \
    || echo "WARN: flash-attn install failed, will use eager attention"

# ----- Python 의존성 -----
COPY requirements.docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# ----- 소스 코드 복사 -----
COPY bridge/ /app/bridge/
COPY collector/ /app/collector/
COPY pipeline/ /app/pipeline/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY pyproject.toml requirements.docker.txt /app/

# ----- 패키지 설치 -----
RUN pip install --no-cache-dir -e /app/bridge && \
    pip install --no-cache-dir -e /app/collector && \
    pip install --no-cache-dir -e /app

# ----- 시연용 데이터 -----
# RoboCasa CloseDrawer (NPZ)
COPY demo_data/CloseDrawer/ /app/data/CloseDrawer/
# SO101 pick_redblock (LeRobot v3.0 원본 - 변환 시연용)
COPY demo_data/SO101-single-pick_redblock_place_bluedish/ /app/data/SO101-single-pick_redblock_place_bluedish/

# ----- 시연용 학습된 모델 -----
# SO101 어댑터 (joint6d, augmented, 10hz)
COPY demo_models/groot_so101_joint6d_aug_10hz/ /app/models/groot_so101_joint6d_aug_10hz/
# RoboCasa CloseDrawer 어댑터
COPY demo_models/groot_direction_CloseDrawer/ /app/models/groot_direction_CloseDrawer/
# LIBERO Spatial 어댑터
COPY demo_models/groot_spatial_9d_r128/ /app/models/groot_spatial_9d_r128/

# ----- 학습 스크립트 configs 심볼릭 링크 -----
RUN ln -s /app/bridge/configs /app/bridge/scripts/train/configs

# ----- RoboCasa 정규화 통계 (eval에 필요) -----
RUN mkdir -p /data/craf/multitask_training_package/data && \
    cp /app/bridge/multitask_training_package/data/metadata.json /data/craf/multitask_training_package/data/ && \
    cp /app/bridge/multitask_training_package/data/metadata_extended.json /data/craf/multitask_training_package/data/

CMD ["bash"]
