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
```

## Usage

### Command Line Interface

The included CLI tool demonstrates all analysis features:

```bash
./build/vidicant_cli
```

Example output:
```
Image width: 1920
Image height: 1080
Is grayscale: No
Average brightness: 128.45
Number of channels: 3
Edge count: 15420
Dominant colors (RGB):
  Color 1: (255.0, 128.5, 67.2)
  Color 2: (45.3, 200.1, 150.8)
  Color 3: (120.7, 89.4, 210.3)
Blur score (variance): 245.67

--- Video Analysis ---
Video frame count: 1800
Video FPS: 30.0
Video resolution: 1920x1080
Video duration: 60.0 seconds
First frame extracted: 1920x1080, channels: 3
Video average brightness: 132.18
Is video grayscale: No
Saved first frame as image: Yes
Video motion score: 12.45
Video dominant colors (RGB):
  Color 1: (240.2, 135.8, 72.1)
  Color 2: (52.7, 195.3, 145.9)
  Color 3: (115.4, 94.2, 205.8)
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
