#!/usr/bin/env python3
"""E-Commerce Product Listing QA & Visual Search Indexer.

Validates e-commerce product photos against marketplace quality guidelines:
1. White balance deviation (color cast detection).
2. Texture homogeneity and subject contrast.
3. Dominant color extraction & hex palette tagging for visual search indexing.

Usage:
    PYTHONPATH=. python3 examples/python/product_qa.py [path/to/product_photo.jpg]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import vidicant


def rgb_to_hex(rgb: list[int | float]) -> str:
    """Convert RGB values (0-255) to hex string."""
    r, g, b = int(round(rgb[0])), int(round(rgb[1])), int(round(rgb[2]))
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}".upper()


def approximate_color_name(rgb: list[int | float]) -> str:
    """Classify RGB into standard color category for visual search indexing."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    max_c, min_c = max(r, g, b), min(r, g, b)

    if max_c < 40:
        return "Black"
    if min_c > 220 and (max_c - min_c) < 20:
        return "White"
    if (max_c - min_c) < 25:
        return "Gray"

    if r > g and r > b:
        if g > 150 and b < 100:
            return "Orange"
        if g < 80 and b < 80:
            return "Red"
        if b > 150:
            return "Magenta/Purple"
        return "Warm Brown" if max_c < 140 else "Red"
    elif g > r and g > b:
        return "Green" if b < 180 else "Teal/Cyan"
    else:
        return "Blue" if r < 120 else "Indigo"


def audit_product_photo(
    image_path: str | Path,
    max_white_balance_cast: float = 0.15,
    min_contrast_ratio: float = 0.20,
    min_sharpness: float = 45.0,
) -> dict:
    """Audit product photo compliance and extract catalog search tags."""
    metrics = vidicant.process_image(str(image_path))
    if "error" in metrics:
        raise ValueError(f"Failed to process image: {metrics['error']}")

    warnings: list[str] = []
    wb_score = metrics.get("white_balance_score", 0.0)
    contrast = metrics.get("contrast_ratio", 0.0)
    blur_score = metrics.get("blur_score", 0.0)
    textures = metrics.get("texture_features", {})

    # White balance color cast check
    if wb_score > max_white_balance_cast:
        warnings.append(f"Strong color cast detected (cast score: {wb_score:.3f} > {max_white_balance_cast:.2f})")

    # Low contrast check (washed out product)
    if contrast < min_contrast_ratio:
        warnings.append(f"Low image contrast (contrast: {contrast:.2f} < {min_contrast_ratio:.2f})")

    # Sharpness check
    if blur_score < min_sharpness:
        warnings.append(f"Sub-optimal product sharpness (blur score: {blur_score:.1f} < {min_sharpness:.1f})")

    # Color palette extraction
    raw_colors = metrics.get("dominant_colors", [])
    color_palette = []
    search_color_tags = []
    for c in raw_colors:
        hex_code = rgb_to_hex(c)
        name = approximate_color_name(c)
        color_palette.append({"hex": hex_code, "rgb": [int(x) for x in c], "family": name})
        if name not in search_color_tags:
            search_color_tags.append(name)

    return {
        "image_file": str(image_path),
        "compliant": len(warnings) == 0,
        "compliance_status": "COMPLIANT" if len(warnings) == 0 else "NEEDS_REVIEW",
        "guideline_warnings": warnings,
        "visual_search_tags": {
            "dominant_color_families": search_color_tags,
            "palette": color_palette,
            "texture_homogeneity": round(textures.get("homogeneity", 0.0), 3),
            "texture_contrast": round(textures.get("contrast", 0.0), 3),
            "symmetry_score": round(metrics.get("symmetry_score", 0.0), 3),
        },
        "quality_metrics": {
            "resolution": f"{metrics.get('width', 0)}x{metrics.get('height', 0)}",
            "blur_score": round(blur_score, 2),
            "white_balance_cast": round(wb_score, 3),
            "contrast_ratio": round(contrast, 3),
            "average_brightness": round(metrics.get("average_brightness", 0.0), 1),
            "noise_estimate": round(metrics.get("noise_estimate", 0.0), 2),
        },
    }


def main() -> int:
    default_path = Path(__file__).parent.parent / "sample.jpg"
    if not default_path.exists():
        default_path = Path(__file__).parent / "sample.jpg"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not target_path.exists():
        print(f"Error: Target image file '{target_path}' does not exist.", file=sys.stderr)
        return 1

    print(f"--> Auditing product photo compliance for: {target_path}")
    report = audit_product_photo(target_path)

    print("\n" + "=" * 60)
    print("E-COMMERCE PRODUCT LISTING QA REPORT")
    print("=" * 60)
    print(f"Status:     {report['compliance_status']}")
    print(f"File:       {report['image_file']}")
    print(f"Resolution: {report['quality_metrics']['resolution']}")
    print(f"Sharpness:  {report['quality_metrics']['blur_score']}")
    print(f"WB Cast:    {report['quality_metrics']['white_balance_cast']}")

    if report["guideline_warnings"]:
        print("\n! Guideline Warnings:")
        for w in report["guideline_warnings"]:
            print(f"    - {w}")
    else:
        print("\n✓ Image fully complies with e-commerce listing standards.")

    print("\n--- Visual Search & Palette Indexing ---")
    print(f"Color Tags: {', '.join(report['visual_search_tags']['dominant_color_families'])}")
    print("Palette:")
    for p in report["visual_search_tags"]["palette"]:
        print(f"  • {p['hex']} ({p['family']}) - RGB{p['rgb']}")

    print("\n--- Full JSON Audit Output ---")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
