#!/usr/bin/env python3
"""Media Catalog Near-Duplicate Clustering.

Scans an image directory and clusters duplicates and near-duplicates using
perceptual hashes (dHash) and Hamming distance thresholds via Vidicant.

Usage:
    PYTHONPATH=. python3 examples/python/dedupe_catalog.py [path/to/image_dir] [--threshold 5]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import vidicant


def cluster_catalog_duplicates(
    directory: str | Path,
    threshold: int = 5,
    recursive: bool = True,
) -> dict:
    """Find near-duplicate image clusters in a directory."""
    return vidicant.find_duplicates(str(directory), threshold=threshold, recursive=recursive)


def main() -> int:
    parser = argparse.ArgumentParser(description="Near-duplicate image clustering with Vidicant.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Target directory to scan for duplicates. Defaults to creating a demo cluster with sample assets.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=5,
        help="Maximum Hamming distance between 64-bit dHash perceptual hashes (default: 5).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive directory traversal.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format.",
    )
    args = parser.parse_args()

    temp_dir_to_clean: Path | None = None
    target_dir: Path

    if args.directory:
        target_dir = Path(args.directory)
        if not target_dir.is_dir():
            print(f"Error: Directory '{target_dir}' does not exist.", file=sys.stderr)
            return 1
    else:
        # Create a transient demo directory with sample duplicates
        sample_img = Path(__file__).parent.parent / "sample.jpg"
        if not sample_img.exists():
            sample_img = Path(__file__).parent / "sample.jpg"
        if not sample_img.exists():
            print(f"Error: Sample asset '{sample_img}' not found.", file=sys.stderr)
            return 1

        temp_dir = Path(tempfile.mkdtemp(prefix="vidicant_dedupe_demo_"))
        temp_dir_to_clean = temp_dir
        shutil.copy(sample_img, temp_dir / "photo_original.jpg")
        shutil.copy(sample_img, temp_dir / "photo_copy_backup.jpg")
        shutil.copy(sample_img, temp_dir / "photo_reupload.jpg")
        target_dir = temp_dir
        print(f"--> No directory provided. Created demo dataset in {temp_dir} with 3 duplicate copies.")

    try:
        print(f"--> Scanning directory '{target_dir}' (Hamming distance threshold: {args.threshold})...")
        results = cluster_catalog_duplicates(target_dir, threshold=args.threshold, recursive=not args.no_recursive)

        if args.json:
            print(json.dumps(results, indent=2))
            return 0

        total = results.get("total_images", 0)
        clusters = results.get("duplicate_clusters", [])
        cluster_count = results.get("clusters_count", len(clusters))

        print("\n" + "=" * 60)
        print("PERCEPTUAL HASH DEDUPLICATION SUMMARY")
        print("=" * 60)
        print(f"Total Images Scanned: {total}")
        print(f"Duplicate Clusters:   {cluster_count}")

        if not clusters:
            print("\n✓ No near-duplicate clusters detected within the given threshold.")
        else:
            for idx, cluster in enumerate(clusters, start=1):
                files = cluster.get("files", cluster.get("members", []))
                count = cluster.get("count", len(files))
                print(f"\nCluster #{idx} ({count} files):")
                for f in files:
                    print(f"  • {f}")

        print("\n--- JSON Summary ---")
        print(json.dumps(results, indent=2))
    finally:
        if temp_dir_to_clean and temp_dir_to_clean.exists():
            shutil.rmtree(temp_dir_to_clean, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
