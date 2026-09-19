# Vidicant

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/huangsam/vidicant/ci.yml)](https://github.com/huangsam/vidicant/actions)
[![License](https://img.shields.io/github/license/huangsam/vidicant)](https://github.com/huangsam/vidicant/blob/main/LICENSE)

Vidicant is a fast, cross-platform library for image and video analysis, feature extraction, and neural assessment. It pairs a high-performance C++17/OpenCV core with a zero-dependency Python standard library (`ctypes`) wrapper.

## Features

- **Image Analysis**: Blur, dominant colors, GLCM texture, and perceptual hashing (dHash).
- **Video Analytics**: Motion scoring, optical flow, scene cuts, and thumbnail selection.
- **Neural Engine**: Classification, object & face detection, embeddings, and quality assessment.
- **Zero Python Dependencies**: Pure Python stdlib runtime (`ctypes`, `json`, `pathlib`).
- **Cross-Platform**: macOS, Linux, and Windows (WSL2).

## Installation

```bash
# Build native engine & CLI (requires OpenCV and Zig 0.16)
zig build

# Install Python package locally
pip install .
```

## Quick Start

### Python

```python
import vidicant

# Image analysis (heuristic & neural classification)
image = vidicant.process_image("photo.jpg", enable_ml=True, task="classify")
print(f"{image['width']}x{image['height']} | Blur: {image['blur_score']:.2f} | Labels: {image['top_labels']}")

# Video analytics (motion & scene changes)
video = vidicant.process_video("clip.mp4")
print(f"{video['duration_seconds']}s @ {video['fps']} fps | Motion: {video['motion_score']:.2f}")
```

### CLI

```bash
# Analyze media files
vidicant_cli photo.jpg clip.mp4 --task detect -o results.json

# Near-duplicate image clustering
vidicant_cli dedupe ./photos/ --threshold 5 -o duplicates.json
```

---
*For full API schemas and CLI flags, see the [User Guide](USERGUIDE.md).*
