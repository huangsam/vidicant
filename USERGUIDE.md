# Vidicant User Guide

Vidicant is a fast, cross-platform media analysis library combining high-performance C++17/OpenCV with zero-dependency Python stdlib (`ctypes`) bindings.

---

## Installation & Requirements

### Requirements
- **Python**: 3.11+
- **Python Dependencies**: **Zero runtime pip dependencies** (uses pure standard library `ctypes`, `json`, `pathlib`, `urllib`).
- **Native Runtime**: Pre-built via Zig (`zig build`) or system OpenCV package.

### Build & Install
```bash
# 1. Build native shared library & CLI
zig build

# 2. Use directly or install locally
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

### 3. Video Analysis
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

#### `process_image(filename, enable_ml=False, task="quality", model_path=None, top_k=5, conf_threshold=0.5, nms_threshold=0.4) -> dict`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | *required* | Path to image file (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`, etc.). |
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

#### `process_image_bytes(data, enable_ml=False, task="quality", model_path=None, top_k=5, conf_threshold=0.5, nms_threshold=0.4) -> dict`

Processes in-memory raw image bytes (`bytes`, `bytearray`, or `memoryview`) via `cv::imdecode` without writing temporary files to disk. Ideal for streaming cloud workers (AWS Lambda, Celery, S3 streaming).

```python
with open("photo.jpg", "rb") as f:
    raw_data = f.read()

metrics = vidicant.process_image_bytes(raw_data)
print(f"Decoded: {metrics['width']}x{metrics['height']}, Blur: {metrics['blur_score']:.2f}")
```

---

### Video Processing

#### `process_video(filename: str) -> dict`

**Returned Video Metrics Schema:**
```json
{
  "filename": "clip.mp4",
  "frame_count": 300,
  "fps": 30.0,
  "duration_seconds": 10.0,
  "width": 1920,
  "height": 1080,
  "average_brightness": 115.4,
  "is_grayscale": false,
  "motion_score": 3.82,
  "optical_flow_magnitude": 2.45,
  "dominant_colors": [[120, 140, 160], [40, 50, 60]],
  "scene_changes": [45, 120, 210],
  "shot_length_stats": {"mean": 75.0, "stddev": 32.1, "min": 45, "max": 120, "count": 3},
  "frame_rate_stability": 0.002,
  "color_consistency": 1.15,
  "flicker_score": 0.04,
  "has_audio_track": true,
  "codec_fourcc": "avc1",
  "best_thumbnail_frame": 85,
  "first_frame_extracted": true,
  "first_frame_path": "clip_first_frame.jpg"
}
```

---

### Model Management Utilities

```python
import vidicant

# Get cache path for default task model (~/.cache/vidicant/models/)
path = vidicant.get_default_model_path(task="classify")

# Ensure model exists locally; downloads to cache if given URL or default
local_path = vidicant.ensure_model("https://example.com/model.onnx")
```

---

## Command Line Interface (`vidicant_cli`)

### Batch Media Processing

```bash
# Basic image & video processing (JSON output)
./zig-out/bin/vidicant_cli photo.jpg clip.mp4 -o results.json

# Streaming JSON Lines format (.jsonl)
./zig-out/bin/vidicant_cli ./dataset/ --format jsonl -o dataset_metrics.jsonl

# Streaming Tabular CSV format (.csv)
./zig-out/bin/vidicant_cli ./dataset/ --format csv -o dataset_metrics.csv

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
# Human-readable summary
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

# Stage shared library into vidicant/ for wheel packaging
zig build -Dinstall-to-pkg

# Run native test suite through Zig
zig build test
```

---

## Troubleshooting & Tips

| Issue | Resolution |
|-------|------------|
| Library not found | Run `zig build` to generate `libvidicant` in `zig-out/lib/`. |
| Missing OpenCV headers | macOS: `brew install opencv`; Linux: `apt install libopencv-dev`. |
| First-time ONNX download | `ensure_model()` caches models in `~/.cache/vidicant/models/`. Set `VIDICANT_MODEL_PATH` to override. |
