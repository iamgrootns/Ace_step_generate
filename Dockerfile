# CORRECT TAG - CUDA 12.4.1 with cuDNN 9.1.0.70
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 on Ubuntu 22.04
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3.12-distutils \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create symlinks
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Upgrade pip
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.9.1 with CUDA 12.4
RUN pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu124

# Install all dependencies
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
    gradio

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
