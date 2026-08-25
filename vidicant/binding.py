from __future__ import annotations

import ctypes
import json
import platform
from pathlib import Path

from .models import ensure_model, get_default_model_path


def _find_library():
    """Find and load the libvidicant shared library."""
    base_dir = Path(__file__).resolve().parent.parent
    system = platform.system()

    if system == "Darwin":
        lib_name = "libvidicant.dylib"
    elif system == "Windows":
        lib_name = "vidicant.dll"
    else:
        lib_name = "libvidicant.so"

    search_paths = [
        base_dir / "zig-out" / "lib" / lib_name,
        base_dir / "vidicant" / lib_name,
        base_dir / lib_name,
    ]

    for p in search_paths:
        if p.is_file():
            return ctypes.CDLL(str(p))

    raise FileNotFoundError(f"Could not locate {lib_name}. Please run 'zig build' first.")


_lib = _find_library()

# Setup C function signatures
_lib.vidicant_is_image_file.argtypes = [ctypes.c_char_p]
_lib.vidicant_is_image_file.restype = ctypes.c_bool

_lib.vidicant_is_video_file.argtypes = [ctypes.c_char_p]
_lib.vidicant_is_video_file.restype = ctypes.c_bool

_lib.vidicant_process_image.argtypes = [ctypes.c_char_p]
_lib.vidicant_process_image.restype = ctypes.c_void_p

_lib.vidicant_process_image_ml.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.vidicant_process_image_ml.restype = ctypes.c_void_p

_lib.vidicant_process_image_dnn.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
]
_lib.vidicant_process_image_dnn.restype = ctypes.c_void_p

_lib.vidicant_process_image_bytes.argtypes = [
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
]
_lib.vidicant_process_image_bytes.restype = ctypes.c_void_p

_lib.vidicant_process_video.argtypes = [ctypes.c_char_p]
_lib.vidicant_process_video.restype = ctypes.c_void_p

_lib.vidicant_dedupe_directory.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_bool,
]
_lib.vidicant_dedupe_directory.restype = ctypes.c_void_p

_lib.vidicant_free_string.argtypes = [ctypes.c_void_p]
_lib.vidicant_free_string.restype = None


def is_image_file(filename: str) -> bool:
    """Check if a file path is a supported image format."""
    return bool(_lib.vidicant_is_image_file(filename.encode("utf-8")))


def is_video_file(filename: str) -> bool:
    """Check if a file path is a supported video format."""
    return bool(_lib.vidicant_is_video_file(filename.encode("utf-8")))


def process_image(
    filename: str,
    enable_ml: bool = False,
    task: str = "quality",
    model_path: str | None = None,
    top_k: int = 5,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> dict:
    """Process an image file and return analysis metrics as a dictionary.

    Args:
        filename: Path to the image file.
        enable_ml: If True, evaluates neural network model via ONNX/DNN.
        task: Neural task type ("quality", "classify", "detect", "embed", "auto").
        model_path: Optional custom path to an ONNX model file or URL. If None and
                    enable_ml is True, uses the cached default model for the task.
        top_k: Number of top classification labels to retrieve (for task="classify").
        conf_threshold: Minimum confidence score for detection (for task="detect").
        nms_threshold: Non-Maximum Suppression IoU threshold (for task="detect").
    """
    if enable_ml or model_path:
        resolved = ensure_model(model_path, task=task)
        if resolved:
            raw_ptr = _lib.vidicant_process_image_dnn(
                filename.encode("utf-8"),
                resolved.encode("utf-8"),
                task.encode("utf-8"),
                top_k,
                conf_threshold,
                nms_threshold,
            )
        else:
            raw_ptr = _lib.vidicant_process_image(filename.encode("utf-8"))
    else:
        raw_ptr = _lib.vidicant_process_image(filename.encode("utf-8"))

    if not raw_ptr:
        raise ValueError(f"Failed to process image: {filename}")
    try:
        s = ctypes.string_at(raw_ptr).decode("utf-8")
        return json.loads(s)
    finally:
        _lib.vidicant_free_string(raw_ptr)


def process_image_bytes(
    data: bytes | bytearray | memoryview,
    enable_ml: bool = False,
    task: str = "quality",
    model_path: str | None = None,
    top_k: int = 5,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> dict:
    """Process an in-memory image byte buffer and return analysis metrics as a dictionary.

    Args:
        data: Raw image bytes (JPEG, PNG, WebP, BMP, etc.).
        enable_ml: If True, evaluates neural network model via ONNX/DNN.
        task: Neural task type ("quality", "classify", "detect", "embed", "auto").
        model_path: Optional custom path to an ONNX model file or URL. If None and
                    enable_ml is True, uses the cached default model for the task.
        top_k: Number of top classification labels to retrieve (for task="classify").
        conf_threshold: Minimum confidence score for detection (for task="detect").
        nms_threshold: Non-Maximum Suppression IoU threshold (for task="detect").
    """
    buf = bytes(data) if not isinstance(data, bytes) else data
    if not buf:
        raise ValueError("Image data buffer is empty")

    resolved = None
    if enable_ml or model_path:
        resolved = ensure_model(model_path, task=task)

    resolved_bytes = resolved.encode("utf-8") if resolved else None
    task_bytes = task.encode("utf-8")

    raw_ptr = _lib.vidicant_process_image_bytes(
        buf,
        len(buf),
        resolved_bytes,
        task_bytes,
        top_k,
        conf_threshold,
        nms_threshold,
    )

    if not raw_ptr:
        raise ValueError("Failed to process in-memory image bytes")
    try:
        s = ctypes.string_at(raw_ptr).decode("utf-8")
        res = json.loads(s)
        if "error" in res:
            raise ValueError(f"Failed to process in-memory image bytes: {res['error']}")
        return res
    finally:
        _lib.vidicant_free_string(raw_ptr)


def process_video(filename: str) -> dict:
    """Process a video file and return analysis metrics as a dictionary."""
    raw_ptr = _lib.vidicant_process_video(filename.encode("utf-8"))
    if not raw_ptr:
        raise ValueError(f"Failed to process video: {filename}")
    try:
        s = ctypes.string_at(raw_ptr).decode("utf-8")
        return json.loads(s)
    finally:
        _lib.vidicant_free_string(raw_ptr)


def find_duplicates(
    directory: str,
    threshold: int = 5,
    recursive: bool = True,
) -> dict:
    """Find duplicate and near-duplicate images in a directory using perceptual hashing.

    Args:
        directory: Path to the directory to scan.
        threshold: Hamming distance threshold (default: 5).
        recursive: Whether to scan subdirectories recursively (default: True).

    Returns:
        Dictionary containing deduplication clusters and statistics.
    """
    raw_ptr = _lib.vidicant_dedupe_directory(directory.encode("utf-8"), threshold, recursive)
    if not raw_ptr:
        raise ValueError(f"Failed to deduplicate directory: {directory}")
    try:
        s = ctypes.string_at(raw_ptr).decode("utf-8")
        return json.loads(s)
    finally:
        _lib.vidicant_free_string(raw_ptr)


__all__ = [
    "ensure_model",
    "find_duplicates",
    "get_default_model_path",
    "is_image_file",
    "is_video_file",
    "process_image",
    "process_image_bytes",
    "process_video",
]
