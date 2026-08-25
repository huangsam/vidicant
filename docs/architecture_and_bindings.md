# Architecture & Python Bindings Design

This document details the architectural layers of Vidicant, the zero-dependency Python bindings design, and key technical decisions.

## System Architecture

Vidicant is structured across three distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Client Layer                     │
│                  (vidicant/ - pure ctypes)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ C-ABI JSON Interface
┌──────────────────────────────▼──────────────────────────────┐
│                      C-ABI Wrapper Layer                    │
│                  (src/vidicant_c_api.cpp)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Direct C++ API
┌──────────────────────────────▼──────────────────────────────┐
│                       Core C++ Library                      │
│        (src/core/, src/dnn/, src/io/, image.cpp, video.cpp) │
│                     Linked against OpenCV                   │
└─────────────────────────────────────────────────────────────┘
```

### 1. Core C++ Library (`src/`, `include/`)
- **`vidicant::types` (`include/vidicant/types.hpp`)**: Core metric and bounding box structs (`ImageMetrics`, `VideoMetrics`, `TextureFeatures`, `BoundingBox`, etc.).
- **`vidicant::core` (`src/core/`)**: Pure computer vision algorithms (`image_ops.hpp`, `video_ops.hpp`) operating directly on `cv::Mat` frames.
- **`vidicant::dnn` (`src/dnn/`)**: Neural inference subsystem (`dnn_engine.hpp`) encapsulating OpenCV DNN execution and model task decoding.
- **`vidicant::io` (`src/io/`)**: Media format detection (`file_detector.hpp`) via extension and magic byte inspection.
- **`IImageLoader` / `IVideoLoader`**: Abstract interfaces enabling custom media loading strategies.
- **`OpenCVImageLoader` / `OpenCVVideoLoader`**: Concrete OpenCV-backed implementations.
- **`ImageHandler` / `VideoHandler`**: High-level coordinators delegating to `core` and `dnn`.
- **Public API**: Convenience functions under `vidicant::` namespace (`getImageDimensions`, `processImage`, `processVideo`).

### 2. C-ABI Wrapper Layer (`include/vidicant/c_api.h`, `src/vidicant_c_api.cpp`)
To avoid ABI coupling and complex C++ binding tools, the native library exports clean `extern "C"` functions declared in `include/vidicant/c_api.h`:
- `vidicant_process_image(const char* image_path)`: Analyzes image and returns a heap-allocated JSON string.
- `vidicant_process_image_dnn(const char* image_path, const char* model_path, const char* task, int top_k, float conf_threshold, float nms_threshold)`: Runs heuristic + neural pipeline and returns a heap-allocated JSON string.
- `vidicant_process_video(const char* video_path)`: Analyzes video and returns a heap-allocated JSON string.
- `vidicant_free_string(const char* ptr)`: Safely deallocates strings allocated by the C++ runtime.

### 3. Python Driver (`vidicant/`)
- Pure Python using the built-in `ctypes` module.
- Zero third-party dependencies (no numpy/pybind11 required at install time).
- Locates `libvidicant.dylib` / `.so` / `.dll` relative to package location or build output.
- Transparent on-demand model download and cache manager (`vidicant/models.py`) targeting `~/.cache/vidicant/models/`.
- Automatically handles serialization: C++ metrics JSON $\rightarrow$ Python `dict`.

### 4. Neural Engine (`opencv_dnn`)
- Embedded inference using OpenCV's built-in `cv::dnn::readNetFromONNX`.
- Supports 4 neural tasks: Quality scoring (NIMA distribution & technical score), Semantic Classification (Softmax + Top-K), Object & Face Detection (`cv::dnn::NMSBoxes`), and Generic Tensor Embeddings.
- Thread-safe memory caching of loaded network graphs (`cv::dnn::Net`).
- Zero extra external C++ libraries required.

## Key Technical Decisions

### Zig 0.16 Build System (`build.zig`)
- Replaces CMake with a unified build orchestrator and compiler driver.
- Hermetic, fast builds linking OpenCV natively.
- Builds both the native shared library (`libvidicant`) and the CLI tool (`vidicant_cli`).

### C-ABI & `ctypes` vs `pybind11`
- **Stable ABI**: `ctypes` works across all Python 3.11+ minor versions without recompiling wheels for every CPython ABI tag (`cp311`, `cp312`, `cp313`, `cp314`).
- **Zero Build Dependencies**: Users do not need a C++ compiler or wheel building tools in Python environments.
- **JSON Serialization**: Clean boundary between C++ structs and Python dictionaries.

## Future Enhancements

- **GPU Acceleration**: Leverage OpenCV CUDA backends for high-throughput video processing.
- **Streaming Support**: Real-time webcam and RTSP stream analysis.
- **Direct Buffer Passing**: Optional zero-copy frame buffer sharing via numpy arrays.
