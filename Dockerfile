# MuseTalk Docker - Standard (CUDA 11.7 / PyTorch 2.0)
# For most NVIDIA GPUs (RTX 20xx, 30xx, 40xx, A100, V100, etc.)
#
# Build: docker build -t musetalk .
# Run:   docker run --gpus all -p 7860:7860 -v ~/.cache/huggingface:/root/.cache/huggingface musetalk

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="cornball-ai"
LABEL description="MuseTalk: Real-Time High Quality Lip Synchronization"
LABEL version="1.5"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.1" \
    && mim install "mmdet>=3.1.0" \
    && mim install "mmpose>=1.1.0"

# Copy application code
COPY . .

# Download model weights
RUN chmod +x download_weights.sh && ./download_weights.sh

EXPOSE 7860

# Default: run Gradio app
CMD ["python", "app.py"]
