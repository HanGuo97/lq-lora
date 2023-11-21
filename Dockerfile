# syntax = docker/dockerfile:1.0-experimental
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# working directory
WORKDIR /workspace

# ---------------------------------------------
# Project-agnostic System Dependencies
# ---------------------------------------------
RUN \
    # Install System Dependencies
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        unzip \
        psmisc \
        vim \
        git \
        ssh \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------
# Build Python depencies and utilize caching
# ---------------------------------------------
COPY ./requirements.txt /workspace/main/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/main/requirements.txt

# upload everything
COPY . /workspace/main/

# Set HOME
ENV HOME="/workspace/main"

# ---------------------------------------------
# Project-agnostic User-dependent Dependencies
# ---------------------------------------------
RUN \
    # Install Awesome vimrc
    git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime && \
    sh ~/.vim_runtime/install_awesome_vimrc.sh

# Reset Entrypoint from Parent Images
# https://stackoverflow.com/questions/40122152/how-to-remove-entrypoint-from-parent-image-on-dockerfile/40122750
ENTRYPOINT []

# load bash
CMD /bin/bash