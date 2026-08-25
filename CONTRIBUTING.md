# Contributing to Vidicant

Guidelines for building, developing, and contributing to Vidicant.

---

## Development Setup

### Supported Platforms
- **macOS**: Apple Silicon (`aarch64`) & Intel (`x86_64`)
- **Linux**: `x86_64` & `aarch64` (Ubuntu, Debian, Fedora, Arch, etc.)
- **Windows**: Windows Subsystem for Linux (**WSL2**) is recommended for local building and testing.

### System Prerequisites
```bash
# macOS
brew install opencv nlohmann-json googletest zig

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install -y libopencv-dev nlohmann-json3-dev libgtest-dev libgmock-dev g++
```

### Build & Test Commands
```bash
# Build native shared library (libvidicant) & CLI executable (vidicant_cli)
zig build

# Run full test suite (native C++ GTest unit tests & Python e2e)
zig build test

# Run native C++ unit tests only
zig build test-native

# Run Python e2e test suite directly
zig build test-e2e
PYTHONPATH=. python3 e2e.py

# Lint & format Python
ruff check . && ruff format .

# Format C++ code
find src test include \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) | xargs clang-format -i

# Format Zig build script
zig fmt build.zig
```

---

## C++ Core Usage

The core engine is usable directly as a modern C++17 library or via the C-ABI:

```cpp
#include "vidicant/vidicant.hpp"
#include <iostream>
#include <optional>
#include <filesystem>

int main() {
    // 1. Analyze an image using ImageAnalysisOptions and std::filesystem::path
    const std::filesystem::path img_path = "image.jpg";
    vidicant::ImageAnalysisOptions img_opts;
    img_opts.task = "classify";
    img_opts.top_k = 5;

    std::optional<vidicant::ImageMetrics> img_metrics = vidicant::getImageMetrics(img_path, img_opts);
    if (img_metrics.has_value()) {
        std::cout << "Resolution: " << img_metrics->width << "x" << img_metrics->height << "\n";
        std::cout << "Blur Score: " << img_metrics->blur_score << "\n";
        std::cout << "Top Labels: " << img_metrics->top_labels.size() << "\n";
    }

    // 2. Analyze a video using VideoAnalysisOptions
    const std::filesystem::path vid_path = "video.mp4";
    vidicant::VideoAnalysisOptions vid_opts;
    vid_opts.scene_change_threshold = 30.0;

    std::optional<vidicant::VideoMetrics> vid_metrics = vidicant::getVideoMetrics(vid_path, vid_opts);
    if (vid_metrics.has_value()) {
        std::cout << "FPS: " << vid_metrics->fps << ", Duration: " << vid_metrics->duration << "s\n";
        std::cout << "Motion Score: " << vid_metrics->motion_score << "\n";
    }

    return 0;
}
```

---

## Architecture & Constraints

- **Python Bindings (`vidicant/`)**: Zero third-party runtime pip dependencies (`ctypes`, `json`, `pathlib`, `urllib` standard library only).
- **C-ABI (`src/vidicant_c_api.cpp`, `include/vidicant/c_api.h`)**: `extern "C"` JSON string interface with paired `vidicant_free_string` deallocations.
- **Build System (`build.zig`)**: Single source of truth for native builds (C++17 standard).
