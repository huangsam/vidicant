# Contributing to Vidicant

This document is for developers who want to build, develop, or contribute to Vidicant.

## Development Setup

### Prerequisites

- CMake 3.24+
- C++17 compatible compiler (GCC 11+, Clang 12+)
- OpenCV 4.x development libraries (`libopencv-dev`)
- GTest (`libgtest-dev`)
- pybind11 (automatically fetched via CMake FetchContent)

### Building from Source

```bash
# Clone and navigate to the repository
git clone <repository-url>
cd vidicant

# Configure CMake
cmake -S . -B build

# Build the project
cmake --build build

# Run all tests
ctest --test-dir build

# Run with verbose output
ctest --test-dir build -V
```

## Code Formatting

Format all C++ code using clang-format:

```bash
find src test include \( -name '*.cpp' -o -name '*.hpp' \) | xargs clang-format -i
```

## C++ Library Usage

The core functionality is available as a C++ library:

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
- Convenience functions in the `vidicant` namespace for easy usage

This design allows swapping backends or adding new analysis methods without changing the API.

## Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
ctest --test-dir build

# Run specific test
ctest --test-dir build -R test_image -V

# Run with coverage (if available)
ctest --test-dir build --coverage
```

## Python Bindings

Vidicant is also available as a Python package via pybind11. See [USERGUIDE.md](USERGUIDE.md) and [AGENTS.md](AGENTS.md#python-bindings-implementation) for details.

To rebuild the Python extension during development:

```bash
pip install -e . --break-system-packages
```

## Dependency Management

### System Dependencies (apt)

```bash
sudo apt install libopencv-dev libgtest-dev nlohmann-json3-dev
```

### Build Dependencies

- **pybind11**: Automatically fetched via CMake FetchContent (v3.0.1)
- **scikit-build-core**: Required for Python packaging (`pip install scikit-build-core`)

## DevContainer

This project includes a DevContainer configuration. To use it:

1. Open in VS Code with the Remote Containers extension
2. Click "Reopen in Container"
3. All dependencies are automatically installed

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
