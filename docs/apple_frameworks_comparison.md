# Comparison to Apple's Media Frameworks

This document details how Vidicant compares to Apple's proprietary media processing frameworks (Vision, AVFoundation, Core Image).

## Framework Overview

| Vidicant Capability | Apple Framework Equivalent | Description |
|---------------------|---------------------------|-------------|
| **Image Analysis** | **Vision Framework** | Dimensions, color analysis, edge detection, blur/sharpness, grayscale detection |
| **Video Analysis** | **AVFoundation** | Metadata, frame extraction, motion analysis, brightness, format handling |
| **Processing Pipeline** | **Core Image** | OpenCV-based processing, extensible filter/loader design, cross-platform execution |

## Feature Mapping

### Image & Neural Analysis (Vision Equivalent)
- **Dimensions & Channels**: Resolution and color format extraction.
- **Color Distribution**: Average brightness, dominant color palette calculation via k-means.
- **Edge & Sharpness**: Laplacian variance for blur detection, Canny edge analysis.
- **Grayscale Detection**: Automatic check for monochrome imagery.
- **Classification (`VNClassifyImageRequest`)**: Top-K label prediction via MobileNet / ImageNet ONNX.
- **Face & Object Detection (`VNDetectFaceRectanglesRequest`)**: Bounding box localization with NMS via YOLO / YuNet.
- **Vector Embeddings (`VNGenerateImageFeaturePrintRequest`)**: Generic raw feature tensor extraction.

### Video Analysis (AVFoundation Equivalent)
- **Stream Metadata**: Frame count, duration, FPS, resolution.
- **Motion Analysis**: Frame-by-frame differencing score to quantify scene movement.
- **Frame Extraction**: Seeking and capturing specific frames as images.

### Processing Architecture (Core Image Equivalent)
- **Extensible Loader Pattern**: Abstract `IImageLoader` / `IVideoLoader` interfaces allow plugging custom media sources.
- **Optimized C++ Core**: Direct memory operations and OpenCV algorithms without framework overhead.

## Key Differences

| Aspect | Apple's Frameworks | Vidicant |
|--------|-------------------|----------|
| **Platforms** | iOS, macOS only | Linux, macOS |
| **Dependencies** | Proprietary Apple APIs | Open-source OpenCV |
| **Language** | Swift / Objective-C | C++17 (with Python bindings) |
| **Cost** | Free (Apple ecosystem only) | Free (open-source MIT) |
| **Customization** | Limited by Apple APIs | Fully extensible |
| **Distribution** | Apple ecosystem / App Store | Anywhere (pip, CLI, native binaries) |
| **Code Density** | Verbose API surface | Concise, direct C++ & Python APIs |

## Development Insights

OpenCV provides significant advantages for cross-platform workflows:
- **Portability**: Code runs identically across macOS and Linux.
- **Algorithm Access**: Direct access to raw OpenCV algorithms without platform-specific abstraction layers.
- **Boilerplate Reduction**: C++ and OpenCV implement vision pipelines with significantly fewer lines of setup code than AVFoundation pipelines.
