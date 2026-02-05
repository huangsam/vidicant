# Vidicant Python Bindings

Vidicant is available as a pip-installable Python package exposing C++ media processing via pybind11.

## Installation

```bash
pip install /path/to/vidicant --break-system-packages
```

**Prerequisites:** Python 3.8+, CMake 3.24+, C++ compiler, OpenCV dev libraries, pybind11 (apt: `pybind11-dev`)

## Quick Usage

```python
import vidicant

# File type checks
vidicant.is_image_file("photo.jpg")   # True
vidicant.is_video_file("video.mp4")   # True

# Image analysis
result = vidicant.process_image("photo.jpg")
# Returns: width, height, is_grayscale, average_brightness, 
#          channels, edge_count, dominant_colors, blur_score

# Video analysis
result = vidicant.process_video("video.mp4")
# Returns: frame_count, fps, width, height, duration_seconds,
#          average_brightness, is_grayscale, motion_score, dominant_colors
```

## API Reference

| Function | Returns | Purpose |
|----------|---------|---------|
| `is_image_file(filename)` | bool | Check if file is supported image format |
| `is_video_file(filename)` | bool | Check if file is supported video format |
| `process_image(filename)` | dict | Analyze image, return metadata & metrics |
| `process_video(filename)` | dict | Analyze video, return metadata & metrics |

## Building from Source

```bash
# Install build dependencies
sudo apt install pybind11-dev python3-pybind11 scikit-build-core

# Build and install
pip install . --break-system-packages
```

## Architecture

- **C++ Layer**: Image/video processing using OpenCV
- **Binding Layer**: `src/vidicant_py.cpp` (pybind11)
- **Python Package**: `vidicant/` exposes C++ functions

CMakeLists.txt uses `pybind11_add_module()` to build the extension module, linking against OpenCV and other dependencies.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: vidicant_py` | Rebuild: `pip install . --break-system-packages` |
| pybind11 not found | Install: `sudo apt install pybind11-dev` |
| OpenCV linking errors | Install: `sudo apt install libopencv-dev` |
| CMake errors | Ensure CMake 3.24+: `cmake --version` |

## Versions

- Python: 3.8+ (tested on 3.12)
- pybind11: 2.11.1 (system apt package)
- OpenCV: 4.13.0+
- CMake: 3.24+

