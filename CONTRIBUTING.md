# Contributing to Vidicant

Guidelines for building, developing, and contributing to Vidicant.

---

## Development Setup

### System Prerequisites
```bash
# macOS
brew install opencv nlohmann-json zig

# Linux (Ubuntu/Debian)
sudo apt install libopencv-dev nlohmann-json3-dev
```

### Build & Test Commands
```bash
# Build native shared library & CLI
zig build

# Run end-to-end Python test suite
PYTHONPATH=. python3 e2e.py

# Lint & format Python
ruff check . && ruff format .

# Format C++ code
find src test include \( -name '*.cpp' -o -name '*.hpp' \) | xargs clang-format -i
```

---

## C++ Core Usage

The core engine is usable directly as a C++17 library or via C-ABI:

```cpp
#include "vidicant/image.hpp"
#include "vidicant/video.hpp"

// Heuristic & Neural Image Metrics
ImageMetrics m = vidicant::getImageMetrics("image.jpg", "model.onnx", "classify", 5);
std::cout << "Width: " << m.width << ", Labels: " << m.top_labels.size() << std::endl;

// Video Analysis
VideoMetrics vm = vidicant::getVideoMetrics("video.mp4");
std::cout << "FPS: " << vm.fps << ", Motion: " << vm.motion_score << std::endl;
```

---

## Architecture & Constraints

- **Python Bindings (`vidicant/`)**: Zero third-party runtime pip dependencies (`ctypes`, `json`, `pathlib`, `urllib` only).
- **C-ABI (`src/vidicant_c_api.cpp`)**: `extern "C"` JSON string interface with paired `vidicant_free_string` deallocations.
- **Build System (`build.zig`)**: Single source of truth for native builds (C++17 standard).

For in-depth architectural details, see [docs/architecture_and_bindings.md](docs/architecture_and_bindings.md) and [docs/apple_frameworks_comparison.md](docs/apple_frameworks_comparison.md).
