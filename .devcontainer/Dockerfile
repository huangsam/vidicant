# Use the official Microsoft C++ image as a base
FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

# Install media-specific dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    gdb \
    libopencv-dev \
    libtesseract-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    googletest \
    google-mock \
    clang-format \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
