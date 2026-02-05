# Use the official Microsoft C++ image as a base
FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

# Install media-specific dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    gdb \
    clang-format \
    libopencv-dev \
    libtesseract-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtest-dev \
    libgmock-dev \
    nlohmann-json3-dev \
    # Add Python and pip for AI/ML scripting and tools
    python3 \
    python3-pip \
    python3-dev \
    # Add CUDA toolkit for GPU acceleration (if NVIDIA GPU is available)
    nvidia-cuda-toolkit \
    # Add other productivity tools
    htop \
    tmux \
    git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python ML libraries and build tools via pip
RUN pip3 install --break-system-packages \
    numpy \
    opencv-python \
    tensorflow \
    torch \
    torchvision \
    jupyter \
    matplotlib \
    scikit-learn \
    scikit-build-core
