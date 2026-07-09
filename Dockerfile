# ============================================================
# ODFE-OBM-OACL: Optimization-Driven Long-Tailed RS Detection
# ============================================================
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

LABEL maintainer="Rahul Malik <rahul.malik@galgotiasuniversity.edu.in>"
LABEL description="Optimization-Driven Feature Learning for Long-Tailed Object Detection in Optical Remote Sensing Imagery"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies (OpenCV runtime libs etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Default working directories expected by scripts
RUN mkdir -p data checkpoints logs results outputs

EXPOSE 6006
# TensorBoard default port

CMD ["/bin/bash"]
