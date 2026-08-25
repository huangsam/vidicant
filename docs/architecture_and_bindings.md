# Architecture & Python Bindings Design

This document details the architectural layers of Vidicant, the zero-dependency Python bindings design, and key technical decisions.

## System Architecture

Vidicant is structured across three distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Client Layer                     │
│           (vidicant/ - pure ctypes, zero pip deps)          │
└──────────────────────────────┬──────────────────────────────┘
                               │ C-ABI JSON Interface
┌──────────────────────────────▼──────────────────────────────┐
│                      C-ABI Wrapper Layer                    │
│      (include/vidicant/c_api.h, src/vidicant_c_api.cpp)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ Modern C++17 API
┌──────────────────────────────▼──────────────────────────────┐
│                       Core C++ Library                      │
│   (include/vidicant/vidicant.hpp, src/core/, src/dnn/,      │
│     src/io/, src/image.cpp, src/video.cpp, src/pipeline)    │
│                     Linked against OpenCV                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Architectural Layers

### 1. Core C++ Library (`include/`, `src/`)
- **Top-Level Umbrella Header (`include/vidicant/vidicant.hpp`)**: Single include providing access to the entire C++ API.
- **Data Models & Types (`include/vidicant/types.hpp`)**:
  - `ImageMetrics` & `VideoMetrics`: Comprehensive analysis result aggregates.
  - `ImageAnalysisOptions` & `VideoAnalysisOptions`: Options structs for configuring thresholds, ML models, and sampling.
  - `TextureFeatures`, `BoundingBox`, `DetectedObject`, `ClassificationLabel`, `ShotLengthStats`.
- **Pure Algorithms (`src/core/`)**:
  - `image_ops.hpp / .cpp`: Image processing kernels (blur, contrast, GLCM texture, dHash, white balance, symmetry).
  - `video_ops.hpp / .cpp`: Video processing kernels (optical flow, scene detection, motion score, shot statistics).
  - `dedupe.hpp / .cpp`: Perceptual hash deduplication and DisjointSet union-find clustering.
- **Neural Engine Subsystem (`src/dnn/`)**:
  - `dnn_engine.hpp / .cpp`: Encapsulates OpenCV DNN execution (`cv::dnn::readNetFromONNX`), tensor preprocessing, Softmax, NMS, and feature extraction.
- **Media I/O & File Detection (`src/io/`)**:
  - `file_detector.hpp / .cpp`: Fast format identification via file extensions and magic byte signatures.
- **Loader Abstractions (`include/vidicant/image.hpp`, `include/vidicant/video.hpp`)**:
  - `IImageLoader` / `IVideoLoader`: Abstract interfaces for custom loading strategies.
  - `OpenCVImageLoader` / `OpenCVVideoLoader`: Concrete OpenCV-backed implementations.
  - `MemoryImageLoader`: In-memory decoding via `cv::imdecode`.
  - `ImageHandler` / `VideoHandler`: High-level coordinators with instance-level caching.
- **High-Level Functional API**:
  - `vidicant::getImageMetrics(const std::filesystem::path &, const ImageAnalysisOptions &) -> std::optional<ImageMetrics>`
  - `vidicant::getVideoMetrics(const std::filesystem::path &, const VideoAnalysisOptions &) -> std::optional<VideoMetrics>`
  - `vidicant::processImage`, `vidicant::processImageBytes`, `vidicant::processVideo`, `vidicant::dedupeDirectory` (in `pipeline.hpp`).

### 2. C-ABI Wrapper Layer (`include/vidicant/c_api.h`, `src/vidicant_c_api.cpp`)
To ensure long-term stability and eliminate CPython ABI coupling, the native library exports `extern "C"` functions returning JSON strings:
- `vidicant_is_image_file(const char *filename) -> bool`
- `vidicant_is_video_file(const char *filename) -> bool`
- `vidicant_process_image(const char *filename) -> const char *`
- `vidicant_process_image_ml(const char *filename, const char *model_path) -> const char *`
- `vidicant_process_image_dnn(const char *filename, const char *model_path, const char *task, int top_k, float conf_threshold, float nms_threshold) -> const char *`
- `vidicant_process_image_bytes(const uint8_t *buffer, size_t len, const char *model_path, const char *task, int top_k, float conf_threshold, float nms_threshold) -> const char *`
- `vidicant_process_video(const char *filename) -> const char *`
- `vidicant_free_string(const char *str) -> void`: Safely deallocates strings allocated by the native runtime.

### 3. Python Driver (`vidicant/`)
- Built exclusively with Python standard library (`ctypes`, `json`, `pathlib`, `urllib`).
- Zero external runtime dependencies (no `numpy`, `pybind11`, or `cython` needed at runtime).
- Dynamic loader discovers `libvidicant.dylib` (macOS), `libvidicant.so` (Linux), or `vidicant.dll` (Windows) across build and package directories.
- Automatic model caching and on-demand download manager (`vidicant/models.py`) targeting `~/.cache/vidicant/models/`.

### 4. Neural Engine (`opencv_dnn`)
- Direct ONNX inference via OpenCV's built-in `cv::dnn::Net`.
- Supports 4 task modes:
  1. `quality`: Aesthetic rating & technical quality assessment (NIMA-style distribution).
  2. `classify`: Softmax classification with Top-K label extraction.
  3. `detect`: Object and face detection bounding boxes with Non-Maximum Suppression (`cv::dnn::NMSBoxes`).
  4. `embed`: Generic high-dimensional feature vector extraction.
- Graph caching to eliminate repeated model load overhead.

---

## Key Technical Decisions

### Zig 0.16 Build System (`build.zig`)
- Unified build orchestrator and compiler driver for C++17 codebase.
- Replaces CMake for standard build and test flows with clean multi-target commands:
  - `zig build`: Builds shared library and CLI.
  - `zig build test`: Runs native C++ GTest suite and Python e2e suite.
  - `zig build -Dinstall-to-pkg`: Stages library for wheel packaging.

### Platform Support
- **macOS**: Apple Silicon (`aarch64`) & Intel (`x86_64`)
- **Linux**: `x86_64` & `aarch64`
- **Windows**: Windows Subsystem for Linux (**WSL2**) recommended for zero-friction development.

### C-ABI & `ctypes` vs `pybind11`
- **Universal ABI**: `ctypes` avoids compiling separate CPython wheels for every Python minor version (`cp311`, `cp312`, `cp313`, `cp314`).
- **Zero Build Friction**: End users don't need a C++ compiler installed in their Python environment.
- **Isolated Memory Model**: Clear allocation ownership using `vidicant_free_string`.
