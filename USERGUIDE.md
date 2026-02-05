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

### Example 1: Batch Image Analysis

```python
import vidicant
import os
import pandas as pd

# Analyze all images in a directory
results = []
for filename in os.listdir("images/"):
    if vidicant.is_image_file(filename):
        result = vidicant.process_image(f"images/{filename}")
        result["filename"] = filename
        results.append(result)

# Convert to DataFrame for analysis
df = pd.DataFrame(results)
print(df[["filename", "width", "height", "average_brightness"]])

# Find brightest images
brightest = df.nlargest(5, "average_brightness")
```

### Example 2: Quality Control

```python
import vidicant

def check_image_quality(filename):
    result = vidicant.process_image(filename)
    
    # Check various quality metrics
    issues = []
    
    if result["blur_score"] < 0.5:
        issues.append("Image is blurry")
    
    if result["average_brightness"] > 240:
        issues.append("Image is overexposed")
    
    if result["average_brightness"] < 20:
        issues.append("Image is underexposed")
    
    return "PASS" if not issues else f"FAIL: {', '.join(issues)}"

# Test multiple images
for image in ["photo1.jpg", "photo2.jpg", "photo3.jpg"]:
    print(f"{image}: {check_image_quality(image)}")
```

### Example 3: ML Feature Extraction

```python
import vidicant
import numpy as np
from sklearn.preprocessing import StandardScaler

# Extract features for machine learning
features = []
labels = []

for category in ["cats", "dogs"]:
    for filename in os.listdir(f"data/{category}"):
        if vidicant.is_image_file(f"data/{category}/{filename}"):
            result = vidicant.process_image(f"data/{category}/{filename}")
            
            # Extract numeric features
            feature_vec = [
                result["average_brightness"],
                result["edge_count"],
                len(result["dominant_colors"]),
                result["blur_score"]
            ]
            features.append(feature_vec)
            labels.append(category)

# Normalize features
X = StandardScaler().fit_transform(features)
y = labels

# Now X, y ready for sklearn, PyTorch, TensorFlow, etc.
```

### Example 4: Video Motion Detection

```python
import vidicant

def analyze_video_activity(video_file):
    result = vidicant.process_video(video_file)
    
    motion = result["motion_score"]
    duration = result["duration_seconds"]
    
    if motion > 0.7:
        activity_level = "HIGH"
    elif motion > 0.3:
        activity_level = "MEDIUM"
    else:
        activity_level = "LOW"
    
    print(f"Video: {video_file}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Motion: {motion:.2f}")
    print(f"  Activity Level: {activity_level}")

analyze_video_activity("security_footage.mp4")
```

## Performance Tips

- **Batch Processing**: Use a loop for multiple files rather than calling once per file
  ```python
  # Good
  for file in files:
      result = vidicant.process_image(file)
  
  # Bad (slower)
  results = [vidicant.process_image(file) for file in files]
  ```

- **Large Videos**: Motion detection may take time for long videos
  - Consider sampling frames if working with hour-long videos

- **Memory**: Results are returned as Python dicts, which are memory-efficient

## Common Issues

### "ModuleNotFoundError: No module named 'vidicant'"

**Solution:** Reinstall the package:
```bash
pip install /path/to/vidicant --break-system-packages
```

### "Cannot open video file"

**Solution:** Ensure the file exists and is in a supported format:
```python
import os
if os.path.exists("video.mp4"):
    if vidicant.is_video_file("video.mp4"):
        result = vidicant.process_video("video.mp4")
    else:
        print("File format not supported")
```

### Import errors on non-Linux systems

**Solution:** Ensure you have the necessary development libraries installed:
- **macOS:** `brew install opencv`
- **Windows:** Use pre-built wheels (when available on PyPI)

## What's Next?

- See [AGENTS.md](AGENTS.md#python-bindings-implementation) for technical details about Python bindings
- See [CONTRIBUTING.md](CONTRIBUTING.md) for building from source
- See [README.md](README.md) for project overview
