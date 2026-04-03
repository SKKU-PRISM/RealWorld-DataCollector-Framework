# =============================================================================
# ACS Docker Image: AutoDataCollector + RoboBridge 통합 환경
#
# 빌드:   docker build -t acs .
# 실행:   docker run --gpus all -it -e GOOGLE_API_KEY="your-key" acs
# 셸:     docker run --gpus all -it -e GOOGLE_API_KEY="your-key" acs /bin/bash
# =============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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
RUN conda install -y -n base python=3.10 pinocchio hpp-fcl \
    && conda clean -afy

# ----- PyTorch (CUDA 12.1) -----
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ----- Python 의존성 (collector) -----
COPY collector/requirements.txt /tmp/collector_requirements.txt
RUN pip install --no-cache-dir -r /tmp/collector_requirements.txt \
    && rm /tmp/collector_requirements.txt

# ----- Python 의존성 (bridge) -----
COPY requirements.docker.txt /tmp/bridge_requirements.txt
RUN pip install --no-cache-dir -r /tmp/bridge_requirements.txt \
    && rm /tmp/bridge_requirements.txt

# ----- 소스 코드 복사 -----
COPY collector/ /app/collector/
COPY bridge/ /app/bridge/
COPY pipeline/ /app/pipeline/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY run_agent.sh /app/
COPY pyproject.toml /app/

# ----- 패키지 설치 -----
RUN pip install --no-cache-dir -e /app/bridge 2>/dev/null || true && \
    pip install --no-cache-dir -e /app 2>/dev/null || true

# ----- LeRobot 번들 (collector에 포함) -----
ENV PYTHONPATH="/app/collector/lerobot/src:${PYTHONPATH}"

# ----- RoboCasa 정규화 통계 -----
RUN mkdir -p /app/data && \
    if [ -f /app/bridge/multitask_training_package/data/metadata.json ]; then \
        cp /app/bridge/multitask_training_package/data/metadata.json /app/data/; \
    fi && \
    if [ -f /app/bridge/multitask_training_package/data/metadata_extended.json ]; then \
        cp /app/bridge/multitask_training_package/data/metadata_extended.json /app/data/; \
    fi

# ----- 실행 권한 -----
RUN chmod +x /app/run_agent.sh

# ----- API Key는 빌드 시 넣지 않음 — 실행 시 환경변수로 주입 -----
ENV GOOGLE_API_KEY=""
ENV OPENAI_API_KEY=""
ENV DEEPSEEK_API_KEY=""

# ----- 출력 디렉토리 -----
RUN mkdir -p /app/outputs /app/results

ENTRYPOINT ["./run_agent.sh"]
CMD []
