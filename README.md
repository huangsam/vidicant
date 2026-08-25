# Vidicant

Vidicant is a fast, cross-platform library for image and video analysis, feature extraction, and neural assessment (C++17/OpenCV core with zero-dependency Python `ctypes` bindings and Zig 0.16 build system).

## Features

- **Image Analysis**: Dimensions, blur, dominant colors, GLCM texture, dHash perceptual hashing.
- **Video Analytics**: FPS, motion scoring, optical flow, scene cuts, best thumbnail selection.
- **Neural Engine**: Classification (Top-K), object & face detection (NMS), embeddings, quality rating.
- **Zero-Dependency Python**: Pure stdlib `ctypes` wrapper around native `libvidicant`.
- **Cross-Platform**: macOS, Linux (x86_64 / arm64), and Windows via WSL2.

## Quick Start

### Python

```python
import vidicant

# Heuristic & Neural Image Analysis
result = vidicant.process_image("photo.jpg", enable_ml=True, task="classify")
print(f"Resolution: {result['width']}x{result['height']}, Labels: {result['top_labels']}")

# In-Memory Image Byte Buffer Processing
with open("photo.jpg", "rb") as f:
    byte_result = vidicant.process_image_bytes(f.read())
print(f"Decoded: {byte_result['width']}x{byte_result['height']}, Blur: {byte_result['blur_score']:.2f}")

# Video Analysis
video = vidicant.process_video("video.mp4")
print(f"Duration: {video['duration_seconds']}s, Motion: {video['motion_score']:.2f}")
```

### C++17 Core Library

```cpp
#include "vidicant/vidicant.hpp"
#include <iostream>

int main() {
    const std::filesystem::path image_path = "photo.jpg";
    vidicant::ImageAnalysisOptions img_opts;
    img_opts.task = "classify";
    img_opts.top_k = 3;

    if (auto metrics = vidicant::getImageMetrics(image_path, img_opts)) {
        std::cout << "Resolution: " << metrics->width << "x" << metrics->height << "\n";
        std::cout << "Blur Score: " << metrics->blur_score << "\n";
    }

    const std::filesystem::path video_path = "video.mp4";
    if (auto video_metrics = vidicant::getVideoMetrics(video_path)) {
        std::cout << "FPS: " << video_metrics->fps << ", Motion: " << video_metrics->motion_score << "\n";
    }
}
```

### Native CLI

```bash
# Batch media processing with positional inputs
vidicant_cli photo.jpg clip.mp4 --task detect -o results.json

# Near-duplicate image clustering
vidicant_cli dedupe ./photos/ --threshold 5 --format json -o duplicates.json
```

## Documentation

- **[USERGUIDE.md](USERGUIDE.md)** — Python API reference, schema specifications, CLI usage, and examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Build instructions, C++ API reference, and development workflow
- **[TODO.md](TODO.md)** — Project roadmap and planned feature tiers
- **[AGENTS.md](AGENTS.md)** — Agent guidelines, constraints, and verification commands
- **[docs/architecture_and_bindings.md](docs/architecture_and_bindings.md)** — Architecture, C-ABI layer, and design decisions
- **[docs/apple_frameworks_comparison.md](docs/apple_frameworks_comparison.md)** — Comparison with Apple Vision & AVFoundation
