# Use an official CUDA-enabled image as the base image
# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

# Install Miniconda
RUN apt-get update && apt-get install -y wget bzip2 libgl1 libglib2.0-0 tzdata && apt-get upgrade -y && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -u -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash

# Install required packages
RUN apt install -y libjpeg-turbo8 libjpeg-turbo8-dev libpng16-16 libpng-tools libpng-dev build-essential libsqlite3-dev

# Set the PATH variable
ENV PATH="/opt/conda/bin:$PATH"

# Set the CUDA library paths
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/opt/conda/envs/syncmvd/lib:$LD_LIBRARY_PATH"

# Set the working directory
WORKDIR /app

# Create the conda environment
RUN /opt/conda/bin/conda create -n syncmvd python=3.11 --yes

# RUN /opt/conda/bin/conda config --set channel_priority strict

# Install the required packages in the conda environment
# RUN /opt/conda/bin/conda run -n syncmvd conda uninstall -y sqlite libsqlite

# Set the shell to use the new conda environment
SHELL ["/bin/bash", "-c"]

# Copy the project folder into the container
COPY FlexiSyncMVD /app/FlexiSyncMVD

# Install Conda Git
# RUN /opt/conda/bin/conda run -n syncmvd conda install -c conda-forge git

# Install required Python packages with specified versions
RUN /opt/conda/bin/conda run -n syncmvd pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
# RUN /opt/conda/bin/conda run -n syncmvd pip install --pre torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/nightly/cu121
# RUN /opt/conda/bin/conda run -n syncmvd conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

RUN /opt/conda/bin/conda run -n syncmvd pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# Install Conda Git
# RUN /opt/conda/bin/conda run -n syncmvd conda install -c conda-forge git
RUN apt install -y git

RUN /opt/conda/bin/conda run -n syncmvd pip --use-deprecated=legacy-certs install git+https://github.com/openai/CLIP.git
RUN /opt/conda/bin/conda run -n syncmvd pip install -r FlexiSyncMVD/requirements.txt

# Install PyTorch and CUDA toolkit
# RUN /opt/conda/bin/conda run -n syncmvd conda install -c pytorch -c conda-forge pytorch=2.4.0 cudatoolkit=11.8

# Install PyTorch3D
# RUN /opt/conda/bin/conda run -n syncmvd conda install -c iopath iopath
RUN /opt/conda/bin/conda run -n syncmvd conda install -c fvcore -c conda-forge fvcore
RUN /opt/conda/bin/conda run -n syncmvd conda install -c conda-forge pytorch3d

# RUN /opt/conda/bin/conda run -n syncmvd conda install -y cctbx202211::libsqlite==3.40.0

# Set the entrypoint to start with the conda environment activated
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate syncmvd && exec bash"]

# Set environment variables for NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
