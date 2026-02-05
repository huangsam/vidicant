# Vidicant

Vidicant is a cross-platform library for analyzing images and videos, extracting meaningful features like brightness, colors, motion, and edge detection. It provides capabilities similar to Apple's Vision and AVFoundation frameworks but using OpenCV for broader platform compatibility.

## Quick Start

**For Python users:**
```bash
pip install /path/to/vidicant
```

```python
import vidicant

# Analyze an image
result = vidicant.process_image("photo.jpg")
print(f"Brightness: {result['average_brightness']}")
print(f"Colors: {result['dominant_colors']}")

# Analyze a video
result = vidicant.process_video("video.mp4")
print(f"Duration: {result['duration_seconds']}s")
print(f"Motion: {result['motion_score']}")
```

**For C++ users:**
```bash
# Build the library
cmake -S . -B build
cmake --build build

# Run the binary
./build/vidicant_cli image.jpg video.mp4

# You can also use the C++ API for your own projects too
```

## Features

- **Image Analysis**: Dimensions, brightness, color analysis, edge detection, blur scoring
- **Video Analysis**: Frame count, FPS, resolution, duration, motion detection
- **Cross-platform**: Windows, macOS, Linux support
- **Python Integration**: Full Python bindings via pybind11
- **CLI Tool**: Command-line interface for quick analysis

## Documentation

- **[USERGUIDE.md](USERGUIDE.md)** — Python usage, API reference, practical examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Building from source, development setup, C++ API
- **[AGENTS.md](AGENTS.md)** — Project context, development process, Python bindings architecture

## Comparison to Similar Tools

| Tool | Purpose | Strengths |
|------|---------|-----------|
| **Vidicant** | Media analysis & feature extraction | Fast, cross-platform, Python-friendly |
| **PIL/Pillow** | Image manipulation | Mature, many filters and transforms |
| **OpenCV** | Computer vision | Advanced algorithms, low-level control |
| **Apple Vision** | iOS/macOS analysis | Native integration, proprietary |

Vidicant fills the "fast media feature extraction" niche — ideal for ML preprocessing pipelines, batch analysis, and cross-platform distribution.

## Use Cases

- **Machine Learning**: Extract image/video metrics for training datasets
- **Batch Processing**: Analyze thousands of media files quickly
- **Media QA**: Validate image/video quality metrics
- **Data Science**: Prepare media features for analysis
- **Cross-platform Apps**: Consistent media analysis across Windows, Mac, Linux
