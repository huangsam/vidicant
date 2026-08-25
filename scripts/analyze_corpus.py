"""Corpus-scale analysis and benchmark script for Vidicant."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vidicant


def _analyze_single_image(img_path_str: str) -> dict | None:
    """Worker function to process a single image via Vidicant."""
    try:
        res = vidicant.process_image(img_path_str)
        return {
            "path": img_path_str,
            "success": True,
            "width": res.get("width", 0),
            "height": res.get("height", 0),
            "channels": res.get("channels", 0),
            "is_grayscale": res.get("is_grayscale", False),
            "average_brightness": res.get("average_brightness", 0.0),
            "sharpness_score": res.get("sharpness_score", 0.0),
            "edge_count": res.get("edge_count", 0),
            "perceptual_hash": res.get("perceptual_hash", 0),
        }
    except Exception as exc:
        return {
            "path": img_path_str,
            "success": False,
            "error": str(exc),
        }


def hamming_distance(h1: int, h2: int) -> int:
    """Calculate Hamming distance between two 64-bit unsigned integers."""
    return bin(h1 ^ h2).count("1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run large-scale media analysis and benchmarking using Vidicant.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("test_data"),
        help="Directory containing test images",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="Glob pattern for image discovery (default: *.jpg)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_data/corpus_report.json"),
        help="Path to save summary report JSON",
    )
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=5,
        help="Hamming distance threshold for near-duplicate clustering (default: 5)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' not found.", file=sys.stderr)
        return 1

    img_files = [str(p) for p in sorted(data_dir.rglob(args.pattern))]
    total_files = len(img_files)
    if total_files == 0:
        print(
            f"No files matching '{args.pattern}' found in '{data_dir}'.",
            file=sys.stderr,
        )
        return 1

    print(f"Discovered {total_files} files matching '{args.pattern}'.")
    print(f"Starting parallel analysis with {args.workers} workers...")

    start_time = time.perf_counter()
    results: list[dict] = []
    failed: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_analyze_single_image, p): p for p in img_files}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res and res.get("success"):
                results.append(res)
            else:
                failed.append(res)

            if i % 200 == 0 or i == total_files:
                elapsed = time.perf_counter() - start_time
                fps = i / elapsed if elapsed > 0 else 0
                print(f"Progress: [{i}/{total_files}] ({i * 100 // total_files}%) | {fps:.1f} imgs/sec")

    total_time = time.perf_counter() - start_time
    throughput = len(results) / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("CORPUS ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total Files Processed: {total_files}")
    print(f"Successful:             {len(results)}")
    print(f"Failed:                 {len(failed)}")
    print(f"Total Wall Clock Time:  {total_time:.2f}s")
    print(f"Throughput:             {throughput:.1f} images/sec")

    if not results:
        print("No successful results to summarize.", file=sys.stderr)
        return 1

    # Aggregate Statistics
    brightnesses = [r["average_brightness"] for r in results]
    sharpnesses = [r["sharpness_score"] for r in results]
    edge_counts = [r["edge_count"] for r in results]
    total_pixels = sum(r["width"] * r["height"] for r in results)
    grayscale_count = sum(1 for r in results if r["is_grayscale"])

    # Near-duplicate clustering
    print(f"\nScanning for near-duplicate image clusters (Hamming dist <= {args.hash_threshold})...")
    clusters: list[list[str]] = []
    visited: set[int] = set()

    for idx1, r1 in enumerate(results):
        if idx1 in visited:
            continue
        h1 = r1["perceptual_hash"]
        cluster = [r1["path"]]
        for idx2 in range(idx1 + 1, len(results)):
            if idx2 in visited:
                continue
            h2 = results[idx2]["perceptual_hash"]
            if hamming_distance(h1, h2) <= args.hash_threshold:
                cluster.append(results[idx2]["path"])
                visited.add(idx2)
        if len(cluster) > 1:
            clusters.append(cluster)

    print(f"Found {len(clusters)} near-duplicate cluster(s).")
    for i, cl in enumerate(clusters[:5], 1):
        print(f"  Cluster {i} ({len(cl)} images):")
        for path in cl[:3]:
            print(f"    - {Path(path).name}")
        if len(cl) > 3:
            print(f"    ... and {len(cl) - 3} more.")

    summary = {
        "total_files": total_files,
        "successful_count": len(results),
        "failed_count": len(failed),
        "total_time_seconds": total_time,
        "throughput_images_per_sec": throughput,
        "total_megapixels_processed": round(total_pixels / 1_000_000, 2),
        "grayscale_images": grayscale_count,
        "color_images": len(results) - grayscale_count,
        "brightness": {
            "mean": round(statistics.mean(brightnesses), 2),
            "stdev": round(statistics.stdev(brightnesses), 2) if len(brightnesses) > 1 else 0,
            "min": round(min(brightnesses), 2),
            "max": round(max(brightnesses), 2),
        },
        "sharpness": {
            "mean": round(statistics.mean(sharpnesses), 2),
            "median": round(statistics.median(sharpnesses), 2),
            "min": round(min(sharpnesses), 2),
            "max": round(max(sharpnesses), 2),
        },
        "edge_count": {
            "mean": round(statistics.mean(edge_counts), 1),
            "median": round(statistics.median(edge_counts), 1),
            "min": min(edge_counts),
            "max": max(edge_counts),
        },
        "near_duplicate_clusters_count": len(clusters),
        "near_duplicate_clusters": [[Path(p).name for p in cl] for cl in clusters],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detailed summary report to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
