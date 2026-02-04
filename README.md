# Vidicant

Vidicant is a cross-platform C++ library for analyzing photo and video content, providing features similar to Apple's Vision and AVFoundation frameworks but using OpenCV for broader platform compatibility.

## Overview

This project demonstrates the ability to build media analysis tools that extract meaningful features from images and videos. It's designed as a cross-platform alternative to proprietary frameworks like Apple's Vision, offering:

- **Image Analysis**: Dimensions, brightness, color channels, edge detection, dominant colors, blur scoring
- **Video Analysis**: Frame count, FPS, resolution, duration, motion detection, brightness analysis
- **Extensible Architecture**: Interface-based design allowing different backends
- **Performance**: Efficient processing using OpenCV's optimized algorithms

## Getting Started

### Prerequisites
- CMake 3.24+
- OpenCV 4.x
- GTest (for testing)
- C++17 compatible compiler

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd vidicant

# Prepare the build environment
cmake -S . -B build

# Build the project
cmake --build build

# Run tests
ctest --test-dir build

# Format all the code
find src test include \( -name '*.cpp' -o -name '*.hpp' \) | xargs clang-format -i
```

## Usage

### Command Line Interface

The included CLI tool demonstrates all analysis features:

```bash
./build/vidicant_cli image.jpg video.mp4
```

### Library Usage

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
