# Vidicant User Guide

Vidicant is a fast, cross-platform media analysis library combining high-performance C++17/OpenCV with zero-dependency Python stdlib (`ctypes`) bindings.

---

## Installation & Requirements

### Supported Platforms
- **macOS**: Apple Silicon (`aarch64`) & Intel (`x86_64`)
- **Linux**: `x86_64` & `aarch64` (Ubuntu, Debian, Fedora, Arch, etc.)
- **Windows**: Seamless development and execution via **WSL2** (Windows Subsystem for Linux)

### Requirements
- **Python**: 3.11+
- **Python Dependencies**: **Zero runtime pip dependencies** (uses pure standard library `ctypes`, `json`, `pathlib`, `urllib`).
- **Native Runtime**: Pre-built via Zig (`zig build`) or system OpenCV package (`libopencv-dev` / Homebrew `opencv`).

### Build & Install
```bash
# 1. Build native shared library & CLI
zig build

# 2. Run full test suite (native C++ unit tests & Python e2e)
zig build test

# 3. Stage shared library into vidicant/ for local packaging or distribution
zig build -Dinstall-to-pkg

# 4. Use directly or install locally
PYTHONPATH=. python3 -c "import vidicant; print(vidicant.__file__)"
pip install . --break-system-packages
```

---

## Quickstart

### 1. File Type Detection
```python
import vidicant

vidicant.is_image_file("photo.jpg")  # True
vidicant.is_video_file("clip.mp4")  # True
```

### 2. Image Analysis (Heuristic & Neural)
```python
import vidicant

# Standard heuristic analysis
metrics = vidicant.process_image("photo.jpg")
print(f"Resolution: {metrics['width']}x{metrics['height']}")
print(f"Blur: {metrics['blur_score']:.2f}, Noise: {metrics['noise_type']}")

# Neural semantic classification (Top-K)
res_cls = vidicant.process_image("photo.jpg", enable_ml=True, task="classify", top_k=3)
print("Top labels:", res_cls["top_labels"])

# Object & face detection with NMS
res_det = vidicant.process_image("photo.jpg", enable_ml=True, task="detect", conf_threshold=0.5)
print("Detections:", res_det["detected_objects"])

# Raw ONNX tensor embeddings
res_emb = vidicant.process_image("photo.jpg", enable_ml=True, task="embed")
print(f"Embedding ({len(res_emb['embedding'])} dims):", res_emb["embedding"])
```

### 3. In-Memory Byte Buffer Processing
```python
import vidicant

# Decode and analyze directly from raw memory bytes without writing to disk
with open("photo.jpg", "rb") as f:
    raw_data = f.read()

metrics = vidicant.process_image_bytes(raw_data)
print(f"Decoded: {metrics['width']}x{metrics['height']}, Blur: {metrics['blur_score']:.2f}")
```

### 4. Video Analysis
```python
import vidicant

video = vidicant.process_video("clip.mp4")
print(f"Duration: {video['duration_seconds']}s @ {video['fps']} fps")
print(f"Motion: {video['motion_score']:.2f}, Scenes: {video['scene_changes']}")
print(f"Best Thumbnail Frame: #{video['best_thumbnail_frame']}")
```

---

## API Reference

### Image Processing

#### `process_image(filename: str, enable_ml: bool = False, task: str = "quality", model_path: str | None = None, top_k: int = 5, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> dict`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | *required* | Path to image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, etc.). |
| `enable_ml` | `bool` | `False` | Enables ONNX neural inference if `True`. |
| `task` | `str` | `"quality"` | DNN task mode: `"quality"`, `"classify"`, `"detect"`, `"embed"`, or `"auto"`. |
| `model_path` | `str \| None` | `None` | Custom ONNX model path or URL. Defaults to cached task model. |
| `top_k` | `int` | `5` | Number of top classification labels (for `task="classify"`). |
| `conf_threshold` | `float` | `0.5` | Detection confidence threshold (for `task="detect"`). |
| `nms_threshold` | `float` | `0.4` | Non-Maximum Suppression IoU threshold (for `task="detect"`). |

**Returned Image Metrics Schema:**
```json
{
  "filename": "photo.jpg",
  "width": 1920,
  "height": 1080,
  "aspect_ratio": 1.777,
  "channels": 3,
  "is_grayscale": false,
  "average_brightness": 128.5,
  "blur_score": 963.2,
  "sharpness_score": 15.4,
  "contrast_ratio": 42.1,
  "saturation_level": 85.0,
  "entropy": 7.42,
  "edge_count": 45230,
  "noise_estimate": 2.15,
  "noise_type": "gaussian",
  "symmetry_score": 0.88,
  "white_balance_score": 4.2,
  "perceptual_hash": 139123847291837482,
  "dominant_colors": [[255, 200, 150], [100, 120, 140], [40, 50, 60]],
  "histogram": [[...], [...], [...]],
  "hue_histogram": [120, 45, 0, ...],
  "texture_features": {
    "contrast": 12.3,
    "energy": 0.045,
    "homogeneity": 0.62,
    "correlation": 0.89
  },
  "ml_evaluated": true,
  "aesthetic_score": 6.85,
  "technical_quality_score": 0.65,
  "top_labels": [
    {"class_id": 281, "label": "class_281", "confidence": 0.892}
  ],
  "detected_objects": [
    {"box": [120.0, 80.0, 200.0, 250.0], "class_name": "face", "confidence": 0.94}
  ],
  "embedding": [0.124, -0.452, 0.891]
}
```

---

#### `process_image_bytes(data: bytes | bytearray | memoryview, enable_ml: bool = False, task: str = "quality", model_path: str | None = None, top_k: int = 5, conf_threshold: float = 0.5, nms_threshold: float = 0.4) -> dict`

Processes in-memory raw image bytes (`bytes`, `bytearray`, or `memoryview`) via `cv::imdecode` without writing temporary files to disk. Ideal for streaming cloud workers (AWS Lambda, Celery, S3 streaming).

```python
with open("photo.jpg", "rb") as f:
    raw_data = f.read()

metrics = vidicant.process_image_bytes(raw_data)
print(f"Decoded: {metrics['width']}x{metrics['height']}, Blur: {metrics['blur_score']:.2f}")
```

---

### Video Processing

#### `process_video(filename: str, stride: int = 1, sample_fps: float | None = None, export_scenes_dir: str | None = None) -> dict`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | *required* | Path to the video file. |
| `stride` | `int` | `1` | Frame sampling stride (e.g. `2` = inspect every 2nd frame for 2x speedup). |
| `sample_fps` | `float \| None` | `None` | Target sampling rate in frames per second (e.g. `1.0` = sample 1 fps). Overrides `stride` if set. |
| `export_scenes_dir` | `str \| None` | `None` | Optional directory to automatically export the sharpest JPEG thumbnail for each detected scene transition. |

**Returned Video Metrics Schema:**
```json
{
  "filename": "clip.mp4",
  "first_frame_extracted": true,
  "first_frame_info": {
    "width": 1920,
    "height": 1080,
    "channels": 3
  },
  "first_frame_saved": true,
  "first_frame_path": "clip_first_frame.jpg",
  "frame_count": 300,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "duration_seconds": 10.0,
  "average_brightness": 115.4,
  "is_grayscale": false,
  "motion_score": 3.82,
  "dominant_colors": [[120, 140, 160], [40, 50, 60]],
  "scene_changes": [45, 120, 210],
  "scene_thumbnails": [
    {
      "scene_index": 1,
      "frame_index": 45,
      "timestamp_seconds": 1.5,
      "thumbnail_path": "scenes/clip_scene_1_frame_45.jpg",
      "sharpness_score": 425.8
    }
  ],
  "frame_rate_stability": 0.002,
  "color_consistency": 1.15,
  "optical_flow_magnitude": 2.45,
  "has_audio_track": true,
  "shot_length_stats": {
    "mean": 75.0,
    "stddev": 32.1,
    "min": 45.0,
    "max": 120.0,
    "count": 3
  },
  "flicker_score": 0.04,
  "best_thumbnail_frame": 85,
  "temporal_brightness_curve": [114.2, 115.1, 115.8],
  "codec_fourcc": "avc1"
}
```

---

### File Type Detection

```python
import vidicant

# Extension & magic byte inspection
is_img = vidicant.is_image_file("sample.png")  # True
is_vid = vidicant.is_video_file("sample.mp4")  # True
```

---

### Neural Model Management

```python
import vidicant

# Get local cache path for task model (~/.cache/vidicant/models/)
path = vidicant.get_default_model_path(task="classify")

# Ensure model exists locally; downloads to cache if given URL or default task model
local_path = vidicant.ensure_model("https://example.com/model.onnx")
```

---

## Command Line Interface (`vidicant_cli`)

### Batch Media Processing

```bash
# Basic image & video processing (JSON output with positional arguments)
./zig-out/bin/vidicant_cli photo.jpg clip.mp4 -o results.json

# Streaming JSON Lines format (.jsonl) across a directory
./zig-out/bin/vidicant_cli ./dataset/ --format jsonl -o dataset_metrics.jsonl

# Streaming Tabular CSV format (.csv)
./zig-out/bin/vidicant_cli ./dataset/ --format csv -o dataset_metrics.csv

# Accelerated video processing with stride (sample every 5th frame)
./zig-out/bin/vidicant_cli clip.mp4 --stride 5

# Video processing with fixed target sampling rate (1 fps) & automatic scene cut thumbnail export
./zig-out/bin/vidicant_cli clip.mp4 --sample-rate 1.0 --export-scenes ./thumbnails/

# Declarative quality filtering (isolate clean, high-contrast, sharp images)
./zig-out/bin/vidicant_cli ./dataset/ --filter "blur_score > 60 and contrast_ratio > 0.2 and width >= 512" -o clean_images.json

# Classification with Top-3 labels
./zig-out/bin/vidicant_cli photo.jpg --task classify --top-k 3

# Face & Object detection with custom thresholds
./zig-out/bin/vidicant_cli photo.jpg --task detect --conf-threshold 0.6 --nms-threshold 0.3

# Custom ONNX model
./zig-out/bin/vidicant_cli photo.jpg --model custom_model.onnx --task embed
```

### Near-Duplicate Image Clustering (`dedupe`)

Cluster images in a directory using Hamming distance on 64-bit `dHash` perceptual hashes:

```bash
# Human-readable summary output
./zig-out/bin/vidicant_cli dedupe ./photos/ --threshold 5

# JSON output with cluster groupings
./zig-out/bin/vidicant_cli dedupe ./photos/ --threshold 5 --format json -o duplicates.json

# Streaming CSV cluster mappings
./zig-out/bin/vidicant_cli dedupe ./photos/ --threshold 5 --format csv -o duplicates.csv
```

---

## Build System & Testing

```bash
# Build native shared library & CLI
zig build

# Run full test suite (native C++ GTest unit tests & Python e2e)
zig build test

# Run native C++ unit tests only
zig build test-native

# Run Python e2e tests only
zig build test-e2e

# Stage shared library into vidicant/ for wheel packaging
zig build -Dinstall-to-pkg
```

---

## Troubleshooting & Tips

| Issue | Resolution |
|-------|------------|
| Library not found (`libvidicant`) | Run `zig build` to generate `libvidicant` in `zig-out/lib/` or `zig build -Dinstall-to-pkg`. |
| Missing OpenCV headers | macOS: `brew install opencv`; Linux: `sudo apt install libopencv-dev`. |
| Missing GTest/GMock headers | macOS: `brew install googletest`; Linux: `sudo apt install libgtest-dev libgmock-dev`. |
| First-time ONNX download | `ensure_model()` caches models in `~/.cache/vidicant/models/`. Set `VIDICANT_MODEL_PATH` to override. |
| Windows native compilation | Use **WSL2** (Ubuntu recommended) with standard Linux build steps. |
