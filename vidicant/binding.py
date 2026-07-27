import ctypes
import json
import platform
from pathlib import Path


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

    raise FileNotFoundError(
        f"Could not locate {lib_name}. Please run 'zig build' first."
    )


_lib = _find_library()

# Setup C function signatures
_lib.vidicant_is_image_file.argtypes = [ctypes.c_char_p]
_lib.vidicant_is_image_file.restype = ctypes.c_bool

_lib.vidicant_is_video_file.argtypes = [ctypes.c_char_p]
_lib.vidicant_is_video_file.restype = ctypes.c_bool

_lib.vidicant_process_image.argtypes = [ctypes.c_char_p]
_lib.vidicant_process_image.restype = ctypes.c_void_p

_lib.vidicant_process_video.argtypes = [ctypes.c_char_p]
_lib.vidicant_process_video.restype = ctypes.c_void_p

_lib.vidicant_free_string.argtypes = [ctypes.c_void_p]
_lib.vidicant_free_string.restype = None


def is_image_file(filename: str) -> bool:
    """Check if a file path is a supported image format."""
    return bool(_lib.vidicant_is_image_file(filename.encode("utf-8")))


def is_video_file(filename: str) -> bool:
    """Check if a file path is a supported video format."""
    return bool(_lib.vidicant_is_video_file(filename.encode("utf-8")))


def process_image(filename: str) -> dict:
    """Process an image file and return analysis metrics as a dictionary."""
    raw_ptr = _lib.vidicant_process_image(filename.encode("utf-8"))
    if not raw_ptr:
        raise ValueError(f"Failed to process image: {filename}")
    try:
        s = ctypes.string_at(raw_ptr).decode("utf-8")
        return json.loads(s)
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
