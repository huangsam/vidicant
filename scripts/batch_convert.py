"""Batch convert Y4M test sequences to JPG and MP4 test fixtures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def convert_file(y4m_path: Path) -> tuple[Path, bool, str]:
    """Convert a single Y4M file into JPG and MP4 fixtures."""
    jpg_path = y4m_path.with_suffix(".jpg")
    mp4_path = y4m_path.with_suffix(".mp4")

    # 1. Convert to JPG
    cmd_jpg = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(y4m_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(jpg_path),
    ]
    res_jpg = subprocess.run(cmd_jpg, capture_output=True, text=True)
    if res_jpg.returncode != 0:
        return y4m_path, False, f"JPG error: {res_jpg.stderr.strip()}"

    # 2. Convert to MP4 (H.264, yuv420p)
    cmd_mp4 = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(y4m_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    res_mp4 = subprocess.run(cmd_mp4, capture_output=True, text=True)
    if res_mp4.returncode != 0:
        return y4m_path, False, f"MP4 error: {res_mp4.stderr.strip()}"

    return y4m_path, True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Y4M test media into JPG and MP4 fixtures.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("test_data"),
        help="Directory containing Y4M files or subsets",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker threads (default: CPU count)",
    )
    parser.add_argument(
        "--cleanup-archives",
        action="store_true",
        help="Remove .tar.gz archives from data-dir after conversion",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' not found.", file=sys.stderr)
        return 1

    y4m_files = sorted(data_dir.rglob("*.y4m"))
    total_files = len(y4m_files)
    print(f"Found {total_files} Y4M files in '{data_dir}'.")
    print(f"Starting conversion to JPG and MP4 with {args.workers} workers...")

    start_time = time.perf_counter()
    success_count = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_file, path): path for path in y4m_files}
        for i, future in enumerate(as_completed(futures), 1):
            path, ok, err = future.result()
            if ok:
                success_count += 1
            else:
                errors.append(f"{path.name}: {err}")

            if i % 100 == 0 or i == total_files:
                elapsed = time.perf_counter() - start_time
                print(f"Progress: [{i}/{total_files}] ({i * 100 // total_files}%) in {elapsed:.1f}s")

    total_time = time.perf_counter() - start_time
    print(f"\nConversion complete: {success_count}/{total_files} successful in {total_time:.2f}s.")

    if errors:
        print(f"\nEncountered {len(errors)} error(s):", file=sys.stderr)
        for err in errors[:10]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more.", file=sys.stderr)

    if args.cleanup_archives:
        archives = list(data_dir.glob("*.tar.gz"))
        print(f"\nCleaning up {len(archives)} archive file(s)...")
        for archive in archives:
            try:
                archive.unlink()
                print(f"  Removed {archive.name}")
            except OSError as e:
                print(f"  Failed to remove {archive.name}: {e}", file=sys.stderr)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
