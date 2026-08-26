#!/usr/bin/env python3
"""In-Memory Pre-Flight Image Upload Gate.

Simulates a zero-disk-I/O pre-flight quality check for cloud microservices,
AWS Lambda, or HTTP upload endpoints using Vidicant's in-memory byte buffer API.

Usage:
    PYTHONPATH=. python3 examples/python/ingestion_gate.py [path/to/image.jpg]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import vidicant


def evaluate_upload_quality(
    image_bytes: bytes,
    min_blur_score: float = 40.0,
    min_brightness: float = 20.0,
    max_brightness: float = 240.0,
    max_noise: float = 20.0,
    min_dimension: int = 64,
) -> dict:
    """Run in-memory pre-flight quality check on raw image bytes.

    Args:
        image_bytes: Encoded image bytes (JPEG, PNG, WebP, etc.).
        min_blur_score: Minimum acceptable blur score.
        min_brightness: Minimum acceptable average brightness (0-255).
        max_brightness: Maximum acceptable average brightness (0-255).
        max_noise: Maximum acceptable noise estimation.
        min_dimension: Minimum acceptable width and height in pixels.

    Returns:
        Dictionary containing validation decision, rejection reasons, and metrics.
    """
    metrics = vidicant.process_image_bytes(image_bytes)

    rejections: list[str] = []
    warnings: list[str] = []

    # 1. Resolution checks
    width, height = metrics.get("width", 0), metrics.get("height", 0)
    if width < min_dimension or height < min_dimension:
        rejections.append(f"Resolution too low: {width}x{height} (minimum {min_dimension}px)")

    # 2. Blur / Sharpness check
    blur_score = metrics.get("blur_score", 0.0)
    if blur_score < min_blur_score:
        rejections.append(f"Image is too blurry: score {blur_score:.1f} < {min_blur_score}")

    # 3. Exposure checks
    avg_brightness = metrics.get("average_brightness", 128.0)
    if avg_brightness < min_brightness:
        rejections.append(f"Image is severely underexposed: brightness {avg_brightness:.1f} < {min_brightness}")
    elif avg_brightness > max_brightness:
        rejections.append(f"Image is severely overexposed: brightness {avg_brightness:.1f} > {max_brightness}")

    # 4. Noise check
    noise = metrics.get("noise_estimate", 0.0)
    if noise > max_noise:
        warnings.append(f"Elevated image noise detected: {noise:.1f} > {max_noise}")

    return {
        "accepted": len(rejections) == 0,
        "status": "ACCEPTED" if len(rejections) == 0 else "REJECTED",
        "rejection_reasons": rejections,
        "warnings": warnings,
        "metrics": {
            "dimensions": f"{width}x{height}",
            "blur_score": blur_score,
            "brightness": avg_brightness,
            "contrast_ratio": metrics.get("contrast_ratio", 0.0),
            "noise_estimate": noise,
            "is_grayscale": metrics.get("is_grayscale", False),
            "dominant_colors": metrics.get("dominant_colors", []),
        },
    }


def main() -> int:
    default_path = Path(__file__).parent.parent / "sample.jpg"
    if not default_path.exists():
        default_path = Path(__file__).parent / "sample.jpg"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not target_path.exists():
        print(f"Error: Target file '{target_path}' does not exist.", file=sys.stderr)
        return 1

    print(f"--> Reading image into memory buffer: {target_path}")
    with open(target_path, "rb") as f:
        file_bytes = f.read()

    print(f"--> Executing zero-I/O in-memory preflight QA ({len(file_bytes)} bytes)...")
    result = evaluate_upload_quality(file_bytes)

    print("\n--- Ingestion Gate Decision ---")
    print(f"Status: {result['status']}")
    if result["accepted"]:
        print("✓ Asset passed all pre-flight quality criteria.")
    else:
        print("✗ Asset failed pre-flight quality criteria:")
        for r in result["rejection_reasons"]:
            print(f"    - {r}")

    if result["warnings"]:
        print("! Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")

    print("\n--- Extracted Metrics ---")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
