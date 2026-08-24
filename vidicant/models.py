"""Model caching and download utilities for Vidicant ONNX neural assessment.

Zero-dependency implementation using Python standard library (urllib, pathlib).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

# Default lightweight ONNX models
DEFAULT_MODEL_NAME = "aesthetic_mobilenetv2.onnx"
DEFAULT_MODEL_URL = "https://github.com/idealo/image-quality-assessment/raw/master/tests/test_models/mobilenet_v2_0.75_224.onnx"

CLASSIFICATION_MODEL_NAME = "mobilenetv2_imagenet.onnx"
CLASSIFICATION_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"

DETECTION_MODEL_NAME = "yunet_face_detection.onnx"
DETECTION_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def get_cache_dir() -> Path:
    """Return the base cache directory for Vidicant models (~/.cache/vidicant/models)."""
    cache_dir = Path.home() / ".cache" / "vidicant" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_default_model_path(task: str = "quality") -> Path:
    """Return the standard path to the cached default ONNX model for a given task."""
    env_path = os.getenv("VIDICANT_MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()

    if task == "classify":
        return get_cache_dir() / CLASSIFICATION_MODEL_NAME
    if task == "detect":
        return get_cache_dir() / DETECTION_MODEL_NAME
    return get_cache_dir() / DEFAULT_MODEL_NAME


def _get_default_model_url(task: str = "quality") -> str:
    """Return the remote URL for a default task model."""
    if task == "classify":
        return CLASSIFICATION_MODEL_URL
    if task == "detect":
        return DETECTION_MODEL_URL
    return DEFAULT_MODEL_URL


def ensure_model(
    model_path_or_url: str | None = None,
    task: str = "quality",
) -> str | None:
    """Ensure that an ONNX model file exists and is accessible.

    If model_path_or_url is a path to an existing local file, returns its absolute path.
    If model_path_or_url is None, checks local cache and downloads the default model for the task if needed.

    Returns:
        Absolute filepath as string if available, or None if unavailable/offline.
    """
    if model_path_or_url:
        p = Path(model_path_or_url).expanduser().resolve()
        if p.is_file():
            return str(p)
        # If it's a URL, download to cache
        if model_path_or_url.startswith(("http://", "https://")):
            dest = get_cache_dir() / Path(model_path_or_url).name
            if dest.is_file() and dest.stat().st_size > 0:
                return str(dest)
            try:
                urllib.request.urlretrieve(model_path_or_url, str(dest))
                return str(dest)
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                print(f"[Vidicant] Warning: Failed to download model from {model_path_or_url}: {e}")
                return None
        return None

    # Check default model for task
    default_path = get_default_model_path(task=task)
    if default_path.is_file() and default_path.stat().st_size > 0:
        return str(default_path)

    # Attempt to download default model
    model_url = _get_default_model_url(task=task)
    try:
        urllib.request.urlretrieve(model_url, str(default_path))
        if default_path.is_file() and default_path.stat().st_size > 0:
            return str(default_path)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"[Vidicant] Warning: Could not download default {task} model ({e}). Continuing with heuristic metrics.")
        return None

    return None
