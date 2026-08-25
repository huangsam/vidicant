# Comparison to Apple's Media Frameworks

This document details how Vidicant compares to Apple's proprietary media processing frameworks (Vision, AVFoundation, Core Image).

## Framework Overview

| Vidicant Capability | Apple Framework Equivalent | Description |
|---------------------|---------------------------|-------------|
| **Image Analysis** | **Vision Framework** | Dimensions, color analysis, edge detection, blur/sharpness, GLCM texture, grayscale detection |
| **Video Analysis** | **AVFoundation** | Metadata, frame extraction, motion analysis, optical flow, scene cuts, shot stats, brightness curve |
| **Processing Pipeline** | **Core Image** | OpenCV-based processing, extensible filter/loader design, cross-platform execution |

---

## Feature Mapping

### Image & Neural Analysis (Vision Equivalent)
- **Dimensions & Channels**: Resolution and color format extraction.
- **Color Distribution**: Average brightness, saturation, dominant color palette calculation via k-means.
- **Edge & Sharpness**: Laplacian variance for blur detection, mean Sobel gradient magnitude, Canny edge analysis.
- **Grayscale Detection**: Automatic check for monochrome imagery.
- **Classification (`VNClassifyImageRequest`)**: Top-K label prediction via MobileNet / ImageNet ONNX.
- **Face & Object Detection (`VNDetectFaceRectanglesRequest`)**: Bounding box localization with NMS via YOLO / YuNet.
- **Vector Embeddings (`VNGenerateImageFeaturePrintRequest`)**: Generic raw feature tensor extraction.

### Video Analysis (AVFoundation Equivalent)
- **Stream Metadata**: Frame count, duration, FPS, resolution, codec FOURCC, audio track detection.
- **Motion & Temporal Analysis**: Frame differencing motion score and Farneback dense optical flow magnitude.
- **Scene & Shot Analytics**: Scene change frame detection and shot length statistics (mean, stddev, min, max).
- **Frame Extraction**: Seeking, capturing first frame, and automatic best thumbnail selection.

### Processing Architecture (Core Image Equivalent)
- **Extensible Loader Pattern**: Abstract `IImageLoader` / `IVideoLoader` interfaces allow plugging custom media sources.
- **In-Memory Decoding**: `MemoryImageLoader` (`cv::imdecode`) enables zero-disk streaming processing.
- **Optimized C++ Core**: Direct memory operations and OpenCV algorithms without framework overhead.

---

## Key Differences

| Aspect | Apple's Frameworks | Vidicant |
|--------|-------------------|----------|
| **Platforms** | iOS, macOS only (Apple hardware locked) | macOS (Apple Silicon & Intel), Linux (x86_64 & aarch64), Windows (WSL2) |
| **Dependencies** | Proprietary Apple APIs & Metal | Open-source OpenCV (C++17) |
| **Language** | Swift / Objective-C | Modern C++17 with zero-dependency Python `ctypes` bindings |
| **Cost & License** | Proprietary / Closed Source | Free & Open-Source (MIT License) |
| **Customization** | Constrained by Apple APIs & models | Fully customizable algorithms & arbitrary ONNX models |
| **Distribution** | Apple ecosystem / App Store | Anywhere (Python pip wheels, native CLI, C-ABI shared libraries) |
| **Code Density** | Verbose delegate & pipeline setup | Concise, direct C++ & Python APIs |

---

## Development Insights

OpenCV and modern C++17 provide significant advantages for cross-platform workflows:
- **Portability**: Code executes identically across Linux cloud servers (AWS Lambda, Celery workers) and macOS workstations.
- **Direct Algorithm Access**: Direct control over raw OpenCV matrix operations and parameters without OS-specific abstraction layers.
- **Boilerplate Reduction**: Vidicant implements image and video pipelines with significantly fewer lines of setup code than equivalent AVFoundation/Vision Swift pipelines.
