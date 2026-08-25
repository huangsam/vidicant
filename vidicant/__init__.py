"""Vidicant - Cross-platform media analysis library.

A Python package for analyzing images and videos using C++ and OpenCV,
providing fast, cross-platform media processing capabilities.
"""

import sys

if sys.version_info < (3, 11):  # noqa: UP036
    raise RuntimeError(f"Vidicant requires Python 3.11 or newer (running on Python {sys.version_info.major}.{sys.version_info.minor}).")

from .binding import (
    ensure_model,
    find_duplicates,
    get_default_model_path,
    is_image_file,
    is_video_file,
    process_image,
    process_image_bytes,
    process_video,
)

__version__ = "0.1.0"
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
