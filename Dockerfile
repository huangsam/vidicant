FROM ubuntu:noble

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system build dependencies, OpenCV, GTest/GMock, and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    xz-utils \
    ca-certificates \
    g++ \
    libopencv-dev \
    libgtest-dev \
    libgmock-dev \
    nlohmann-json3-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Zig 0.16.0 for the container architecture
ARG ZIG_VERSION=0.16.0
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        TARGET="aarch64-linux"; \
    else \
        TARGET="x86_64-linux"; \
    fi && \
    curl -sSL "https://ziglang.org/download/${ZIG_VERSION}/zig-${TARGET}-${ZIG_VERSION}.tar.xz" -o /tmp/zig.tar.xz && \
    mkdir -p /opt/zig && \
    tar -xf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && \
    ln -s /opt/zig/zig /usr/local/bin/zig && \
    rm /tmp/zig.tar.xz

WORKDIR /workspace

# Default command: build and run full test suite (native C++ unit tests & Python e2e)
CMD ["zig", "build", "test"]
