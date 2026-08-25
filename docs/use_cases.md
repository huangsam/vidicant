# Real-World Use Cases & Architectural Patterns

This document details practical, production-ready use cases and architectural patterns enabled by Vidicant. It illustrates how Vidicant's combination of high-performance C++17 algorithms, OpenCV processing, and zero-dependency Python `ctypes` bindings solves real-world media engineering challenges.

---

## Overview: Where Vidicant Fits

Vidicant sits between low-level media decoders and high-level artificial intelligence workflows:

```
┌─────────────────────────────────────────────────────────────┐
│                 High-Level AI / Applications                │
│    (LLM Vision Agents, Vector DBs, CMS Platforms, Apps)     │
└──────────────────────────────▲──────────────────────────────┘
                               │ Structured Metadata / Embeddings
┌──────────────────────────────┴──────────────────────────────┐
│                    Vidicant Engine (C++/Python)             │
│   • Pre-flight QA (Blur, Noise, White Balance, Symmetry)    │
│   • Video Diagnostics (Motion, Optical Flow, Scene Cuts)    │
│   • Perceptual Hashing & Deduplication (dHash)              │
│   • In-Memory Stream Processing (zero disk I/O)             │
│   • Embedded ONNX Inference (NMS Detection, Top-K Tagging)  │
└──────────────────────────────▲──────────────────────────────┘
                               │ Raw Media Bytes / Files
┌──────────────────────────────┴──────────────────────────────┐
│                   Storage & Ingestion Sources               │
│        (AWS S3, GCP Cloud Storage, Edge Cameras, HTTP)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Digital Asset Management (DAM) & Cloud Ingestion Gates

### Problem
Large media platforms (content marketplaces, photo storage, social networks) ingest millions of user uploads daily. Uploading low-quality, blurry, under-exposed, corrupted, or duplicated assets pollutes catalogs and wastes storage and CDN bandwidth.

### Solution
Deploy Vidicant as a fast, lightweight ingestion filter before committing assets to permanent storage or kicking off expensive GPU processing.

```python
import vidicant


def evaluate_upload_quality(file_bytes: bytes) -> dict:
    """Run in-memory pre-flight quality check on incoming upload."""
    metrics = vidicant.process_image_bytes(file_bytes)

    reasons = []
    if metrics["blur_score"] < 50.0:
        reasons.append("Image is too blurry")
    if metrics["average_brightness"] < 25.0:
        reasons.append("Image is severely underexposed")
    elif metrics["average_brightness"] > 235.0:
        reasons.append("Image is severely overexposed")
    if metrics["noise_estimate"] > 15.0:
        reasons.append("Excessive noise detected")

    return {
        "accepted": len(reasons) == 0,
        "metrics": metrics,
        "rejection_reasons": reasons,
    }
```

---

## 2. Automated Video Chaptering & Smart Thumbnail Selection

### Problem
Video-sharing platforms and learning management systems (LMS) require automated chapter markers and engaging thumbnail previews. Defaulting to frame 0 often yields blank transition frames, while naive interval sampling misses key scene changes.

### Solution
Use Vidicant's `scene_changes`, `shot_length_stats`, and `best_thumbnail_frame` to automatically segment videos into scenes and extract the sharpest, most representative preview frame.

```python
import vidicant


def analyze_video_structure(video_path: str):
    """Extract scene boundaries and optimal thumbnail."""
    metrics = vidicant.process_video(video_path)

    print(f"Total Duration: {metrics['duration_seconds']:.2f}s ({metrics['frame_count']} frames)")
    print(f"Optimal Thumbnail Frame: #{metrics['best_thumbnail_frame']}")
    print(f"Detected {len(metrics['scene_changes'])} scene transitions:")

    for idx, change in enumerate(metrics["scene_changes"], 1):
        timestamp = change["timestamp_seconds"]
        print(f"  Chapter {idx} @ {timestamp:.2f}s (Frame {change['frame_index']})")

    stats = metrics["shot_length_stats"]
    print(f"Pacing: Mean shot length {stats['mean']:.1f} frames (stddev: {stats['stddev']:.1f})")
```

---

## 3. E-Commerce Product Listing QA & Visual Search

### Problem
Marketplace sellers (e.g., Shopify, Amazon, Airbnb) upload unstandardized product photos. Inconsistent backgrounds, incorrect white balances, and off-brand color schemes degrade buyer trust and conversion rates.

### Solution
Automate photo guideline compliance and extract dominant color palettes for catalog search and filtering:

```python
import vidicant


def validate_product_photo(image_path: str):
    metrics = vidicant.process_image(image_path)

    # 1. White balance validation
    if metrics["white_balance_score"] > 0.15:
        print("Warning: Strong color cast detected; consider white balance correction.")

    # 2. Extract top 3 dominant colors for search indexing
    dominant_rgb = metrics["dominant_colors"]
    print(f"Indexed Dominant Colors (RGB): {dominant_rgb}")

    # 3. Subject-to-background contrast and texture
    textures = metrics["texture_features"]
    print(f"Texture Homogeneity: {textures['homogeneity']:.2f}, Contrast: {textures['contrast']:.2f}")
```

---

## 4. Multimodal RAG & Vision LLM Cost Optimization

### Problem
Large Vision-Language Models (e.g., GPT-4V, Gemini 1.5 Pro, Claude 3.5 Sonnet) charge per image/frame processed. Sending 30 FPS video or thousands of redundant image frames into LLMs is prohibitively expensive and introduces high latency.

### Solution
Use Vidicant as an upstream semantic and visual filter:
1. Deduplicate consecutive frames with `perceptual_hash` and `optical_flow_magnitude`.
2. Select only keyframes with high sharpness (`sharpness_score`) and significant motion.
3. Pre-tag objects/scenes using embedded ONNX classifiers before invoking cloud LLMs.

```python
import vidicant


def extract_rag_keyframes(video_path: str, min_motion_threshold: float = 2.0) -> list[int]:
    """Select high-value keyframes for downstream multimodal LLM ingestion."""
    video = vidicant.process_video(video_path)

    # Only ingest keyframes at scene cuts and high-information points
    keyframe_indices = [cut["frame_index"] for cut in video["scene_changes"]]
    if video["best_thumbnail_frame"] not in keyframe_indices:
        keyframe_indices.append(video["best_thumbnail_frame"])

    return sorted(keyframe_indices)
```

---

## 5. Zero-Dependency Microservices (AWS Lambda / Cloud Run / FastAPI)

### Problem
Packaging heavy machine learning frameworks (e.g., PyTorch, TensorFlow, large OpenCV Python wrappers) into serverless functions leads to multi-gigabyte container sizes, slow cold starts, and complex dependency conflicts.

### Solution
Because Vidicant’s Python client uses pure standard library (`ctypes`), serverless bundles remain ultra-lightweight (<50MB) and cold starts take milliseconds.

```python
from fastapi import FastAPI, UploadFile, HTTPException
import vidicant

app = FastAPI(title="Media Analysis Service")


@app.post("/analyze/image")
async def analyze_image(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid media type")

    contents = await file.read()
    metrics = vidicant.process_image_bytes(contents)
    return {
        "status": "success",
        "resolution": f"{metrics['width']}x{metrics['height']}",
        "sharpness": metrics["sharpness_score"],
        "is_grayscale": metrics["is_grayscale"],
        "dominant_colors": metrics["dominant_colors"],
    }
```

---

## 6. Large-Scale Catalog Deduplication (`vidicant dedupe`)

### Problem
Storage bloat and catalog spam caused by mirrored, resized, watermarked, or slightly modified re-uploads.

### Solution
Use Vidicant's CLI to scan and cluster entire asset directories in streaming JSONL or text format:

```bash
# Scan a directory and find near-duplicate clusters (Hamming distance <= 5)
vidicant dedupe /path/to/media/ --threshold 5

# Stream real-time image analytics into analytics pipelines
vidicant stream /path/to/images/ --format jsonl | jq '.path, .sharpness_score'
```

---

## 7. Edge & IoT Security Camera Analytics

### Problem
CCTV and edge camera streams can experience tampering (lens covering, spraying), occlusion, or sudden lighting failures.

### Solution
Continuously monitor camera feeds for anomalous metric shifts:
* **Camera Spraying / Blurring**: Sharp drop in `edge_count` and `sharpness_score`.
* **Occlusion / Blocking**: Sudden collapse in `average_brightness` and `contrast_ratio`.
* **Infrared / Night-Mode Transitions**: `is_grayscale` flips to `True` with increased `noise_estimate`.

---

## Summary Matrix

| Domain | Key Vidicant Features | Primary Benefit |
| :--- | :--- | :--- |
| **Cloud Ingestion** | In-memory bytes, blur score, noise estimation | Blocks corrupt/subpar uploads at zero I/O cost |
| **Video Platforms** | Best thumbnail, scene cuts, shot statistics | Automated preview generation and video chaptering |
| **E-Commerce** | Dominant colors, white balance, contrast | Catalog standardization and visual search |
| **Multimodal RAG** | Keyframe filtering, perceptual hashing, ONNX | 10x-50x reduction in LLM vision token costs |
| **Serverless/APIs** | Pure `ctypes` bindings, sub-millisecond execution | Instant cold starts in lightweight Docker images |
| **Edge & Security** | Optical flow, temporal brightness, edge density | Real-time tampering and anomaly detection |
