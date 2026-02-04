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

Example output:
```
--- Processing Image: image.jpg ---
Image width: 200
Image height: 300
Is grayscale: No
Average brightness: 70.5859
Number of channels: 3
Edge count: 10179
Dominant colors (RGB):
  Color 1: (7.52625, 45.8783, 23.4347)
  Color 2: (148.956, 182.436, 153.126)
  Color 3: (69.3954, 117.61, 102.319)
Blur score (variance): 963.161

--- Processing Video: video.mp4 ---
Video frame count: 250
Video FPS: 25
Video resolution: 320x176
Video duration: 10 seconds
First frame extracted: 320x176, channels: 3
Video average brightness: 116.684
Is video grayscale: No
Saved first frame as image: Yes
Video motion score: 5.16946
Video dominant colors (RGB):
  Color 1: (68.8892, 84.6371, 63.5126)
  Color 2: (199.317, 185.903, 197.755)
  Color 3: (99.5565, 116.96, 102.359)
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
