#!/usr/bin/env python3
"""Video Chaptering and Timeline Segmentation Indexer.

Analyzes video structure to automatically detect scene boundaries, extract pacing
statistics, and identify the optimal representative thumbnail frame.

Usage:
    PYTHONPATH=. python3 examples/python/video_chapters.py [path/to/video.mp4]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import vidicant


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS format."""
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{int(hours):02d}:{int(mins):02d}:{secs:05.2f}"
    return f"{int(mins):02d}:{secs:05.2f}"


def analyze_video_chapters(video_path: str | Path) -> dict:
    """Analyze video and extract chapter markers and pacing metrics."""
    metrics = vidicant.process_video(str(video_path))
    if "error" in metrics:
        raise ValueError(f"Failed to process video: {metrics['error']}")

    fps = metrics.get("fps", 30.0)
    if fps <= 0:
        fps = 30.0

    duration = metrics.get("duration_seconds", 0.0)
    raw_scene_cuts = metrics.get("scene_changes", [])

    # Ensure chapter 0 (start of video) is included
    cut_frames = sorted(set([0] + [c if isinstance(c, int) else c.get("frame_index", 0) for c in raw_scene_cuts]))

    chapters = []
    for i, frame_idx in enumerate(cut_frames, start=1):
        timestamp_sec = frame_idx / fps
        end_sec = cut_frames[i] / fps if i < len(cut_frames) else duration
        chapters.append(
            {
                "chapter_number": i,
                "title": f"Scene {i}",
                "start_frame": frame_idx,
                "start_time_seconds": round(timestamp_sec, 2),
                "start_timestamp": format_timestamp(timestamp_sec),
                "duration_seconds": round(end_sec - timestamp_sec, 2),
            }
        )

    best_thumbnail_frame = metrics.get("best_thumbnail_frame", 0)
    best_thumbnail_sec = best_thumbnail_frame / fps if fps > 0 else 0.0

    stats = metrics.get("shot_length_stats", {})

    return {
        "filename": metrics.get("filename", str(video_path)),
        "duration_seconds": duration,
        "duration_formatted": format_timestamp(duration),
        "resolution": f"{metrics.get('width', 0)}x{metrics.get('height', 0)}",
        "fps": round(fps, 2),
        "total_frames": metrics.get("frame_count", 0),
        "motion_score": round(metrics.get("motion_score", 0.0), 3),
        "optical_flow_magnitude": round(metrics.get("optical_flow_magnitude", 0.0), 3),
        "optimal_thumbnail": {
            "frame_index": best_thumbnail_frame,
            "timestamp_seconds": round(best_thumbnail_sec, 2),
            "timestamp_formatted": format_timestamp(best_thumbnail_sec),
        },
        "pacing_statistics": {
            "mean_shot_length_frames": round(stats.get("mean", 0.0), 1),
            "stddev_shot_length_frames": round(stats.get("stddev", 0.0), 1),
            "total_scene_cuts": len(raw_scene_cuts),
        },
        "chapters": chapters,
    }


def main() -> int:
    default_path = Path(__file__).parent.parent / "sample.mp4"
    if not default_path.exists():
        default_path = Path(__file__).parent / "sample.mp4"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not target_path.exists():
        print(f"Error: Target video file '{target_path}' does not exist.", file=sys.stderr)
        return 1

    print(f"--> Processing video analytics for: {target_path}")
    manifest = analyze_video_chapters(target_path)

    print("\n" + "=" * 60)
    print("VIDEO CHAPTER MANIFEST & PACING ANALYSIS")
    print("=" * 60)
    print(f"Video:      {manifest['filename']}")
    print(f"Duration:   {manifest['duration_formatted']} ({manifest['total_frames']} frames @ {manifest['fps']} fps)")
    print(f"Resolution: {manifest['resolution']}")
    print(f"Motion:     {manifest['motion_score']} (Flow: {manifest['optical_flow_magnitude']})")
    print(f"Thumbnail:  Frame #{manifest['optimal_thumbnail']['frame_index']} (@ {manifest['optimal_thumbnail']['timestamp_formatted']})")
    print(
        f"Pacing:     Mean shot {manifest['pacing_statistics']['mean_shot_length_frames']} frames "
        f"(stddev: {manifest['pacing_statistics']['stddev_shot_length_frames']})"
    )

    print("\n--- Detected Chapters ---")
    for ch in manifest["chapters"]:
        print(f"  [{ch['start_timestamp']}] Chapter {ch['chapter_number']} (Frame {ch['start_frame']}, length {ch['duration_seconds']:.1f}s)")

    print("\n--- JSON Manifest Output ---")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
