"""Vidicant - Cross-platform media analysis library.

A Python package for analyzing images and videos using C++ and OpenCV,
providing fast, cross-platform media processing capabilities.
"""

try:
    from .binding import (
        ensure_model,
        get_default_model_path,
        is_image_file,
        is_video_file,
        process_image,
        process_video,
    )
except (ImportError, FileNotFoundError):
    try:
        from .vidicant_py import (
            is_image_file,
            is_video_file,
            process_image,
            process_video,
        )
    except ImportError:
        import vidicant_py

        is_image_file = vidicant_py.is_image_file
        is_video_file = vidicant_py.is_video_file
        process_image = vidicant_py.process_image
        process_video = vidicant_py.process_video

__version__ = "0.1.0"
__all__ = [
    "ensure_model",
    "get_default_model_path",
    "is_image_file",
    "is_video_file",
    "process_image",
    "process_video",
]
