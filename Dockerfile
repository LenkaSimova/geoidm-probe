# Use an official PyTorch image. CUDA 12.1 and PyTorch 2.x are highly optimized for RTX 3090.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install common system utilities you might need
RUN apt-get update && apt-get install -y     git     curl     nano  python3.12-venv pipx  && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pipx install uv

# Set working directory
WORKDIR /workspace
