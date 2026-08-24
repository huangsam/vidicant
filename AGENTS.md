# AGENTS.md

Cross-platform media analysis library (C++17/OpenCV) with zero-dependency Python `ctypes` bindings and Zig 0.16 build system.

## Quick Commands

```bash
# Build libvidicant & CLI
zig build
# Run test suite
PYTHONPATH=. python3 e2e.py
# Python lint & format
ruff check . && ruff format .
# C++ format
find src test include \( -name '*.cpp' -o -name '*.hpp' \) | xargs clang-format -i
```

## Rules & Constraints

- **Python package (`vidicant/`)**: Pure stdlib only (`ctypes`, `json`, `pathlib`). No external runtime dependencies.
- **C-ABI (`src/vidicant_c_api.cpp`)**: `extern "C"` JSON string APIs; always pair allocations with `free_json_string`.
- **Build (`build.zig`)**: Single source of truth for native builds; keep C++17 compatibility.
- **Verification**: Always verify changes with `PYTHONPATH=. python3 e2e.py`.

## References

- [USERGUIDE.md](USERGUIDE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/architecture_and_bindings.md](docs/architecture_and_bindings.md)
- [docs/apple_frameworks_comparison.md](docs/apple_frameworks_comparison.md)
