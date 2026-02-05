# Vidicant User Guide

A comprehensive guide for using Vidicant to analyze images and videos.

## Installation

### Python Package

```bash
pip install /path/to/vidicant --break-system-packages
```

Or from PyPI (when published):
```bash
pip install vidicant
```

### Requirements
- Python 3.8+
- OpenCV (automatically installed as dependency)
- NumPy (automatically installed as dependency)

## Basic Usage

### Import and Setup

```python
import vidicant

# Check what you can analyze
vidicant.is_image_file("photo.jpg")    # True
vidicant.is_image_file("video.mp4")    # False
vidicant.is_video_file("video.mp4")    # True
```

### Analyzing Images

```python
import vidicant
import json

result = vidicant.process_image("photo.jpg")

print(json.dumps(result, indent=2))
# Output:
# {
#   "width": 1920,
#   "height": 1080,
#   "is_grayscale": false,
#   "average_brightness": 128.5,
#   "channels": 3,
#   "edge_count": 45230,
#   "dominant_colors": [[255, 200, 150], [100, 120, 140], ...],
#   "blur_score": 0.45
# }
```

### Analyzing Videos

```python
import vidicant

result = vidicant.process_video("video.mp4")

print(f"Duration: {result['duration_seconds']} seconds")
print(f"Resolution: {result['width']}x{result['height']}")
print(f"Frame rate: {result['fps']} fps")
print(f"Total frames: {result['frame_count']}")
print(f"Motion intensity: {result['motion_score']}")
```

## API Reference

### Functions

#### `is_image_file(filename: str) -> bool`
Check if a file is a supported image format.

**Supported formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.gif`, `.webp`

```python
if vidicant.is_image_file("file.jpg"):
    result = vidicant.process_image("file.jpg")
```

#### `is_video_file(filename: str) -> bool`
Check if a file is a supported video format.

**Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`

```python
if vidicant.is_video_file("file.mp4"):
    result = vidicant.process_video("file.mp4")
```

#### `process_image(filename: str) -> dict`
Analyze an image file and return metrics.

**Returns:**
```python
{
    "width": int,                    # Image width in pixels
    "height": int,                   # Image height in pixels
    "is_grayscale": bool,            # True if image is grayscale
    "average_brightness": float,     # Mean brightness (0-255)
    "channels": int,                 # Number of color channels (1 or 3)
    "edge_count": int,               # Estimated number of edges detected
    "dominant_colors": list[list],   # Top 3 colors [[R,G,B], ...]
    "blur_score": float              # Sharpness metric (0=blurry, 1=sharp)
}
```

#### `process_video(filename: str) -> dict`
Analyze a video file and return metrics.

**Returns:**
```python
{
    "frame_count": int,              # Total number of frames
    "fps": float,                    # Frames per second
    "width": int,                    # Video width in pixels
    "height": int,                   # Video height in pixels
    "duration_seconds": float,       # Video duration in seconds
    "average_brightness": float,     # Mean brightness across frames
    "is_grayscale": bool,            # True if video is grayscale
    "motion_score": float,           # Motion intensity (0-1)
    "dominant_colors": list[list]    # Top dominant colors
}
```

## Practical Examples

### Batch Image Analysis

```python
import vidicant
import pandas as pd
from pathlib import Path

results = []
for path in Path("images/").glob("*"):
    if vidicant.is_image_file(str(path)):
        result = vidicant.process_image(str(path))
        result["filename"] = path.name
        results.append(result)

df = pd.DataFrame(results)
print(df[["filename", "average_brightness", "blur_score"]])
```

### Quality Control

```python
import vidicant

def check_quality(filename):
    r = vidicant.process_image(filename)
    issues = []
    if r["blur_score"] < 0.5: issues.append("blurry")
    if r["average_brightness"] > 240: issues.append("overexposed")
    if r["average_brightness"] < 20: issues.append("underexposed")
    return "PASS" if not issues else f"FAIL: {', '.join(issues)}"

for img in ["photo1.jpg", "photo2.jpg"]:
    print(f"{img}: {check_quality(img)}")
```

### ML Feature Extraction

```python
import vidicant
from sklearn.preprocessing import StandardScaler

features, labels = [], []
for category in ["cats", "dogs"]:
    for path in Path(f"data/{category}").glob("*.jpg"):
        r = vidicant.process_image(str(path))
        features.append([r["average_brightness"], r["edge_count"], r["blur_score"]])
        labels.append(category)

X = StandardScaler().fit_transform(features)
# Now use X, labels with sklearn, PyTorch, TensorFlow
```

### Video Motion Detection

```python
import vidicant

r = vidicant.process_video("video.mp4")
activity = "HIGH" if r["motion_score"] > 0.7 else "LOW"
print(f"Duration: {r['duration_seconds']:.1f}s, Activity: {activity}")
```

## Performance Tips

- **Batch processing**: Process multiple files in a loop rather than with list comprehensions
- **Large videos**: Motion detection scales with video length
- **Memory**: Results are lightweight Python dicts

## Common Issues

| Issue | Solution |
|-------|----------|
| Module not found | Reinstall: `pip install /path/to/vidicant --break-system-packages` |
| File not found | Verify file exists: `os.path.exists("file.jpg")` |
| Unsupported format | Use `is_image_file()` or `is_video_file()` to check first |
| macOS import errors | Install OpenCV: `brew install opencv` |

## What's Next?

- See [AGENTS.md](AGENTS.md#python-bindings-implementation) for technical details about Python bindings
- See [CONTRIBUTING.md](CONTRIBUTING.md) for building from source
- See [README.md](README.md) for project overview
