# Contributing to Vidicant

This document is for developers who want to build, develop, or contribute to Vidicant.

## Development Setup

### Prerequisites

- Zig compiler (v0.16+)
- OpenCV 4.x/5.x development libraries (e.g. `brew install opencv` on macOS or `libopencv-dev` on Linux)
- nlohmann-json development headers (e.g. `brew install nlohmann-json` on macOS or `nlohmann-json3-dev` on Linux)
- C++17 compatible compiler toolchain (driven by Zig)

### Building from Source

```bash
# Clone and navigate to the repository
git clone <repository-url>
cd vidicant

# Build executable CLI and shared library via Zig
zig build

# Run the native CLI binary
./zig-out/bin/vidicant_cli --image examples/sample.jpg --video examples/sample.mp4

# Run end-to-end Python test suite
PYTHONPATH=. python3 e2e.py
```

## Code Formatting

Format all C++ code using clang-format:

```bash
find src test include \( -name '*.cpp' -o -name '*.hpp' \) | xargs clang-format -i
```

## C++ Library Usage

The core functionality is available as a C++ library or via C-ABI:

```cpp
#include "vidicant/image.hpp"
#include "vidicant/video.hpp"

// Image analysis
auto [width, height] = vidicant::getImageDimensions("image.jpg");
double brightness = vidicant::getImageAverageBrightness("image.jpg");
auto colors = vidicant::getImageDominantColors("image.jpg", 5);

// Video analysis
int frames = vidicant::getVideoFrameCount("video.mp4");
double fps = vidicant::getVideoFPS("video.mp4");
double motion = vidicant::getVideoMotionScore("video.mp4");
```

## Architecture

Vidicant uses an interface-based design for extensibility:

- `IImageLoader` / `IVideoLoader`: Abstract interfaces for media loading
- `OpenCVImageLoader` / `OpenCVVideoLoader`: OpenCV-based implementations
- `ImageHandler` / `VideoHandler`: High-level analysis classes
- `vidicant_c_api.cpp`: Clean `extern "C"` C-ABI wrapper layer for multi-language extensions

This design allows swapping backends or adding new analysis methods without changing the API.

## Python Bindings

Vidicant provides a zero-dependency Python package using Python's built-in `ctypes` over the native C-ABI (`libvidicant.dylib` / `.so` / `.dll`). See [USERGUIDE.md](USERGUIDE.md) and [docs/architecture_and_bindings.md](docs/architecture_and_bindings.md) for details.

To test the Python extension during development:

```bash
PYTHONPATH=. python3 e2e.py
```

## Dependency Management

### System Dependencies

```bash
# macOS
brew install opencv nlohmann-json

# Linux (Debian/Ubuntu)
sudo apt install libopencv-dev nlohmann-json3-dev
```

### Build Dependencies

- **Zig**: `zig build` acts as compiler driver, linker, and build orchestrator.
- **Python**: Standard `ctypes` (built into Python 3.11+) loads the native C-ABI shared library.

## Performance Optimization

- All media processing uses OpenCV's optimized algorithms
- Static linking where possible for performance
- Position-independent code (-fPIC) enabled for library usage
- C++17 standard for modern performance features

## Future Enhancements

Potential contributions:

- Machine Learning Integration: Add OpenCV DNN for classification/detection
- GPU Acceleration: Utilize OpenCV CUDA for performance improvements
- Python Bindings Expansion: Add more analysis functions to Python API
- Real-time Processing: Webcam/streaming video analysis
- Additional Formats: Support more exotic video/image formats
