"""
Vidicant - Cross-platform media analysis library

A Python package for analyzing images and videos using C++ and OpenCV,
providing fast, cross-platform media processing capabilities.
"""

# Import the compiled extension module
try:
    from . import vidicant_py
except ImportError:
    import vidicant_py

process_image = vidicant_py.process_image
process_video = vidicant_py.process_video
is_image_file = vidicant_py.is_image_file
is_video_file = vidicant_py.is_video_file

__version__ = "0.1.0"
__all__ = [
    "process_image",
    "process_video",
    "is_image_file",
    "is_video_file",
]
