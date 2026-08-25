# AGENTS.md

Cross-platform media analysis library (C++17/OpenCV) with zero-dependency Python `ctypes` bindings and Zig 0.16 build system.

## Quick Commands

```bash
# Build libvidicant & CLI
zig build
# Run full test suite (native C++ & Python e2e)
zig build test
# Run Python test suite directly
PYTHONPATH=. python3 e2e.py
# Python lint & format
ruff check . && ruff format .
# C++ format
find src test include \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) | xargs clang-format -i
# Zig format
zig fmt build.zig
```

## Rules & Constraints

- **Python package (`vidicant/`)**: Pure stdlib only (`ctypes`, `json`, `pathlib`, `urllib`). No external runtime dependencies.
- **C-ABI (`src/vidicant_c_api.cpp`, `include/vidicant/c_api.h`)**: `extern "C"` JSON string APIs; always pair allocations with `vidicant_free_string`.
- **C++ Core**: Modern C++17 (`const std::filesystem::path &`, `std::optional<T>`, `ImageAnalysisOptions`, `VideoAnalysisOptions`, `#include "vidicant/vidicant.hpp"`).
- **Build (`build.zig`)**: Single source of truth for native builds; keep C++17 compatibility.
- **Platform Scope**: macOS (Apple Silicon & Intel) and Linux (x86_64 & aarch64), with WSL2 recommended for Windows development.
- **Verification**: Always verify changes with `zig build test` and `PYTHONPATH=. python3 e2e.py`.

## References

- [README.md](README.md)
- [USERGUIDE.md](USERGUIDE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [TODO.md](TODO.md)
- [docs/architecture_and_bindings.md](docs/architecture_and_bindings.md)
- [docs/apple_frameworks_comparison.md](docs/apple_frameworks_comparison.md)
