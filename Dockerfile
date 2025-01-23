# Stage 1: Base CUDA Image with System Dependencies
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt update && \
    apt install -y wget bzip2 libgl1 libglib2.0-0 tzdata git \
		libjpeg-turbo8 libjpeg-turbo8-dev libpng16-16 libpng-tools \
		libpng-dev libsqlite3-dev curl make libssl-dev zlib1g-dev \
		libbz2-dev libreadline-dev libsqlite3-dev wget llvm clang \
		libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
		libffi-dev liblzma-dev build-essential && \
    apt upgrade -y

# Copy UV executable from external image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the PATH variable
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"

# Set the CUDA library paths
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

WORKDIR /app

# Copy meta files from the project
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Install Python using UV
RUN uv python install 3.11.10

# Create a virtual environment
RUN uv venv --python 3.11.10

# Set the PATH variable to the venv
ENV PATH="/app/.venv/bin:$PATH"

# Install project dependencies
RUN uv sync --frozen

# Install PyTorch3D
RUN uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# Copy the project files into the container
COPY ./FlexiSyncMVD /app/FlexiSyncMVD
COPY ./server_demo /app/server_demo
COPY ./server.py /app/server.py

# Expose port 8000
EXPOSE 8000

# Set the entrypoint to start the server.py file
CMD ["uv", "run", "/app/server.py"]

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
