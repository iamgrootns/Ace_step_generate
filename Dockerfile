# Use Ubuntu 24.04 which comes with Python 3.12
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04

WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python commands
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.9.1 with CUDA 12.4 support (matching your environment)
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies first
RUN pip install --no-cache-dir \
    runpod \
    requests \
    huggingface_hub \
    soundfile \
    scipy \
    numpy \
    transformers \
    diffusers \
    accelerate \
    einops \
    omegaconf \
    safetensors \
    sentencepiece \
    protobuf \
    librosa \
    datasets \
    peft \
    tensorboard \
    gradio

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
