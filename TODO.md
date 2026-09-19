# Vidicant Roadmap & TODO

This document tracks planned improvements and feature roadmaps for Vidicant, prioritizing **batch data curation, media quality inspection, and zero-dependency cloud/local pipelines**.

---

## Scope & Philosophy

- **Target Platforms**: macOS (Apple Silicon & Intel) and Linux (x86_64 & aarch64). Windows users can run seamlessly via WSL2.
- **Core Value**: Fast, deterministic C++17/OpenCV heuristics with pure stdlib Python `ctypes` bindings.
- **Out of Scope (Anti-Goals)**:
    - In-the-loop GPU training data loaders (handled better by PyTorch/DALI).
    - Native MSVC toolchain maintenance (WSL2 recommended for Windows development).
    - Adding external runtime dependencies to the Python package.

---

## Active Roadmap

### Neural Usability & Wheel Distribution

- [ ] **Dynamic Neural Preprocessing & Aspect-Ratio Letterboxing**
    - Support arbitrary model input dimensions (e.g. $336 \times 336$ for CLIP, $640 \times 640$ for YOLO) rather than fixed $224 \times 224$.
    - Add letterbox padding options to preserve image aspect ratios before model inference.
- [ ] **Automated Multi-Arch CI/CD & Prebuilt Wheels**
    - Configure GitHub Actions matrix to build native shared libraries (`.dylib` on macOS arm64/x86_64, `.so` on Linux x86_64/aarch64).
    - Package native binaries inside pure Python wheels for instant `pip install vidicant` without requiring local compilation or system dependencies.
