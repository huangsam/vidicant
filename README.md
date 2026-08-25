# Vidicant

Vidicant is a fast, cross-platform library for image and video analysis, feature extraction, and neural assessment (C++17/OpenCV core with zero-dependency Python `ctypes` bindings and Zig build system).

## Quick Start

**Python (Zero Dependencies):**
```python
import vidicant

# Heuristic & Neural Image Analysis
result = vidicant.process_image("photo.jpg", enable_ml=True, task="classify")
print(f"Resolution: {result['width']}x{result['height']}, Labels: {result['top_labels']}")

# Video Analysis
video = vidicant.process_video("video.mp4")
print(f"Duration: {video['duration_seconds']}s, Motion: {video['motion_score']:.2f}")
```

**Native CLI:**
```bash
zig build
./zig-out/bin/vidicant_cli photo.jpg clip.mp4 --task detect -o results.json
```

## Features

- **Image Analysis**: Dimensions, brightness, dominant colors, edge counts, blur/sharpness, GLCM texture, white balance, dHash.
- **Video Analysis**: FPS, frame count, duration, motion score, scene cuts, shot stats, flicker, best thumbnail selection.
- **Neural Engine**: Semantic classification (Top-K), object/face detection (NMS), generic tensor embeddings, aesthetic/quality rating.
- **Zero-Dependency Python**: Pure stdlib `ctypes` runtime linking native `libvidicant`.
- **Cross-Platform**: macOS and Linux support via Zig 0.16 build system.

## Documentation

- **[USERGUIDE.md](USERGUIDE.md)** — Python API reference, schema specifications, and examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Build instructions, C++ API, and development workflow
- **[TODO.md](TODO.md)** — Project roadmap and planned feature tiers
- **[AGENTS.md](AGENTS.md)** — Agent guidelines, constraints, and verification commands
- **[docs/architecture_and_bindings.md](docs/architecture_and_bindings.md)** — Architecture & C-ABI design
- **[docs/apple_frameworks_comparison.md](docs/apple_frameworks_comparison.md)** — Comparison with Apple Vision & AVFoundation
