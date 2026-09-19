# Vidicant User Guide

Vidicant is a fast, cross-platform media analysis library combining high-performance C++17/OpenCV with zero-dependency Python standard library (`ctypes`) bindings.

---

## Installation & Requirements

### Requirements
- **Python**: 3.11+ (zero external pip runtime dependencies)
- **Native Runtime**: Pre-built via Zig (`zig build`) or system OpenCV package (`libopencv-dev` / Homebrew `opencv`).
- **Supported Platforms**: macOS (Apple Silicon & Intel), Linux (`x86_64` & `aarch64`), and Windows (WSL2).

### Build & Install
```bash
# 1. Build native shared library & CLI
zig build

# 2. Install Python package locally
pip install .
```

---

## Python API Reference

### Image Analysis

#### `process_image`
Analyzes an image file from disk, extracting heuristic quality metrics (resolution, blur, brightness, contrast, perceptual hash) and optional neural assessments (classification, object detection, embeddings).

```python
import vidicant

# Heuristic quality analysis
metrics = vidicant.process_image("photo.jpg")
print(f"Resolution: {metrics['width']}x{metrics['height']}")
print(f"Blur score: {metrics['blur_score']:.2f}, Noise: {metrics['noise_type']}")
print(f"Perceptual hash: {metrics['perceptual_hash']}")

# Neural classification (Top-K)
classified = vidicant.process_image("photo.jpg", enable_ml=True, task="classify", top_k=3)
print("Top labels:", classified["top_labels"])

# Object & face detection with NMS
detected = vidicant.process_image("photo.jpg", enable_ml=True, task="detect", conf_threshold=0.5)
print("Objects:", detected["detected_objects"])

# Raw tensor embeddings
embedded = vidicant.process_image("photo.jpg", enable_ml=True, task="embed")
print(f"Embedding ({len(embedded['embedding'])} dims):", embedded["embedding"])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | *required* | Path to image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`). |
| `enable_ml` | `bool` | `False` | Enables ONNX neural inference if `True`. |
| `task` | `str` | `"quality"` | DNN task: `"quality"`, `"classify"`, `"detect"`, `"embed"`, or `"auto"`. |
| `model_path` | `str \| None` | `None` | Custom ONNX model path or URL. Defaults to cached task model. |
| `top_k` | `int` | `5` | Number of top classification labels (for `task="classify"`). |
| `conf_threshold` | `float` | `0.5` | Detection confidence threshold (for `task="detect"`). |
| `nms_threshold` | `float` | `0.4` | Non-Maximum Suppression IoU threshold (for `task="detect"`). |

#### `process_image_bytes`
Analyzes raw in-memory image bytes (`bytes`, `bytearray`, or `memoryview`) via `cv::imdecode` without writing temporary files to disk. Ideal for streaming cloud workers (AWS Lambda, Celery, S3 uploads).

```python
with open("photo.jpg", "rb") as f:
    metrics = vidicant.process_image_bytes(f.read())

print(f"Decoded: {metrics['width']}x{metrics['height']} | Blur: {metrics['blur_score']:.2f}")
```

*Accepts the same neural parameters (`enable_ml`, `task`, `model_path`, `top_k`, `conf_threshold`, `nms_threshold`) as `process_image`.*

---

### Video Analysis

#### `process_video`
Extracts motion score, scene cuts, shot statistics, optical flow, and identifies the best thumbnail frame.

```python
import vidicant

video = vidicant.process_video("clip.mp4")
print(f"Duration: {video['duration_seconds']}s @ {video['fps']} fps ({video['frame_count']} frames)")
print(f"Motion score: {video['motion_score']:.2f}")
print(f"Scene changes detected at frames: {video['scene_changes']}")
print(f"Best thumbnail frame: #{video['best_thumbnail_frame']}")

# Accelerated processing with stride (sample every 2nd frame) and thumbnail export
fast_video = vidicant.process_video("clip.mp4", stride=2, export_scenes_dir="./scenes/")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | *required* | Path to the video file. |
| `stride` | `int` | `1` | Frame sampling stride (e.g. `2` = inspect every 2nd frame for 2x speedup). |
| `sample_fps` | `float \| None` | `None` | Target sampling rate in frames per second (e.g. `1.0` = sample 1 fps). Overrides `stride`. |
| `export_scenes_dir` | `str \| None` | `None` | Directory to export thumbnail JPEGs for detected scene transitions. |

---

### Utility Functions

#### File Type Detection
Inspects file extensions and magic byte headers:

```python
import vidicant

vidicant.is_image_file("sample.png")  # True
vidicant.is_video_file("sample.mp4")  # True
```

#### Neural Model Management
Manages ONNX model downloads and local caching (`~/.cache/vidicant/models/`):

```python
import vidicant

# Path to cached default model for a task
path = vidicant.get_default_model_path(task="classify")

# Ensure remote model exists locally; downloads if necessary
local_path = vidicant.ensure_model("https://example.com/model.onnx")
```

---

## Command Line Interface (`vidicant_cli`)

### Basic Analysis & Output Formats

```bash
# Analyze media files and output JSON
vidicant_cli photo.jpg clip.mp4 -o results.json

# Stream JSON Lines across an entire directory
vidicant_cli ./dataset/ --format jsonl -o dataset_metrics.jsonl

# Output tabular CSV format
vidicant_cli ./dataset/ --format csv -o dataset_metrics.csv
```

### Video Sampling & Quality Filtering

```bash
# Accelerated video processing (sample every 5th frame)
vidicant_cli clip.mp4 --stride 5

# Sample at 1 fps and export thumbnail images for each scene change
vidicant_cli clip.mp4 --sample-rate 1.0 --export-scenes ./thumbnails/

# Quality filtering: keep clean, high-contrast, sharp images
vidicant_cli ./dataset/ --filter "blur_score > 60 and contrast_ratio > 0.2 and width >= 512" -o clean.json
```

### Neural Inference

```bash
# Classification with Top-3 labels
vidicant_cli photo.jpg --task classify --top-k 3

# Face & Object detection with custom thresholds
vidicant_cli photo.jpg --task detect --conf-threshold 0.6 --nms-threshold 0.3

# Custom ONNX model embeddings
vidicant_cli photo.jpg --model custom_model.onnx --task embed
```

### Near-Duplicate Detection (`dedupe`)

Clusters images by perceptual hash Hamming distance (`dHash`):

```bash
# Summary output to console
vidicant_cli dedupe ./photos/ --threshold 5

# Export duplicate clusters to JSON or CSV
vidicant_cli dedupe ./photos/ --threshold 5 --format json -o duplicates.json
vidicant_cli dedupe ./photos/ --threshold 5 --format csv -o duplicates.csv
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
