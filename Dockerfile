FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch
ENV HF_HOME=/workspace/.cache/huggingface

# Create cache directories
RUN mkdir -p /workspace/.cache/torch /workspace/.cache/huggingface

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command
CMD ["/bin/bash"]
