#!/usr/bin/env python3
"""Multimodal RAG & Vision LLM Keyframe Pruner.

Demonstrates upstream video pre-filtering to minimize token usage and latency
when sending video inputs to Multimodal LLMs (e.g. Gemini 1.5 Pro, GPT-4o, Claude 3.5).

Prunes 90%+ of redundant consecutive frames by selecting only high-information keyframes
(scene transitions, representative thumbnails, and high-motion frames).

Usage:
    PYTHONPATH=. python3 examples/python/rag_keyframe_filter.py [path/to/video.mp4]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import vidicant

# Typical Vision LLM token consumption estimate per image frame (e.g. Gemini/OpenAI standard tile)
ESTIMATED_TOKENS_PER_FRAME = 512
COST_PER_MILLION_VISION_TOKENS_USD = 2.50


def extract_rag_keyframes(video_path: str | Path) -> dict:
    """Extract optimal sparse keyframes and calculate token savings."""
    metrics = vidicant.process_video(str(video_path))
    if "error" in metrics:
        raise ValueError(f"Failed to process video: {metrics['error']}")

    total_frames = metrics.get("frame_count", 0)
    fps = metrics.get("fps", 30.0)
    duration = metrics.get("duration_seconds", 0.0)

    # 1. Collect candidate keyframes
    keyframe_indices: set[int] = set()

    # Always include frame 0
    keyframe_indices.add(0)

    # Include optimal thumbnail frame
    best_thumb = metrics.get("best_thumbnail_frame")
    if best_thumb is not None and 0 <= best_thumb < total_frames:
        keyframe_indices.add(best_thumb)

    # Include all detected scene transition boundaries
    raw_cuts = metrics.get("scene_changes", [])
    for cut in raw_cuts:
        idx = cut if isinstance(cut, int) else cut.get("frame_index", 0)
        if 0 <= idx < total_frames:
            keyframe_indices.add(idx)

    # If video is long and has few cuts, add periodic interval anchors
    sorted_frames = sorted(keyframe_indices)
    keyframes = []
    for f_idx in sorted_frames:
        ts = f_idx / fps if fps > 0 else 0.0
        keyframes.append(
            {
                "frame_index": f_idx,
                "timestamp_seconds": round(ts, 2),
                "is_scene_boundary": f_idx in [c if isinstance(c, int) else c.get("frame_index", 0) for c in raw_cuts],
                "is_best_thumbnail": f_idx == best_thumb,
            }
        )

    # Token and cost economics
    raw_tokens = total_frames * ESTIMATED_TOKENS_PER_FRAME
    pruned_tokens = len(keyframes) * ESTIMATED_TOKENS_PER_FRAME
    reduction_pct = ((raw_tokens - pruned_tokens) / raw_tokens * 100.0) if raw_tokens > 0 else 0.0

    raw_cost_usd = (raw_tokens / 1_000_000.0) * COST_PER_MILLION_VISION_TOKENS_USD
    pruned_cost_usd = (pruned_tokens / 1_000_000.0) * COST_PER_MILLION_VISION_TOKENS_USD

    return {
        "video_file": str(video_path),
        "video_metadata": {
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "motion_score": round(metrics.get("motion_score", 0.0), 3),
        },
        "keyframe_selection": {
            "selected_count": len(keyframes),
            "keyframes": keyframes,
        },
        "token_economics": {
            "raw_frames_total": total_frames,
            "pruned_keyframes_total": len(keyframes),
            "token_reduction_percent": round(reduction_pct, 1),
            "estimated_raw_tokens": raw_tokens,
            "estimated_pruned_tokens": pruned_tokens,
            "estimated_raw_cost_usd": round(raw_cost_usd, 4),
            "estimated_pruned_cost_usd": round(pruned_cost_usd, 4),
        },
    }


def main() -> int:
    default_path = Path(__file__).parent.parent / "sample.mp4"
    if not default_path.exists():
        default_path = Path(__file__).parent / "sample.mp4"
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not target_path.exists():
        print(f"Error: Target video file '{target_path}' does not exist.", file=sys.stderr)
        return 1

    print(f"--> Computing RAG keyframe pruning for: {target_path}")
    plan = extract_rag_keyframes(target_path)

    print("\n" + "=" * 60)
    print("MULTIMODAL VISION LLM KEYFRAME PRUNING PLAN")
    print("=" * 60)
    meta = plan["video_metadata"]
    econ = plan["token_economics"]
    print(f"Video:           {plan['video_file']}")
    print(f"Duration:        {meta['duration_seconds']}s ({meta['total_frames']} frames @ {meta['fps']} fps)")
    print(f"Keyframes Chosen: {econ['pruned_keyframes_total']} of {econ['raw_frames_total']}")
    print(f"Token Reduction: {econ['token_reduction_percent']}% reduction")
    print(f"Est. Tokens:     {econ['estimated_raw_tokens']:,} (raw) -> {econ['estimated_pruned_tokens']:,} (pruned)")
    print(f"Est. LLM Cost:   ${econ['estimated_raw_cost_usd']:.4f} -> ${econ['estimated_pruned_cost_usd']:.4f}")

    print("\n--- Selected Keyframe Sequence ---")
    for kf in plan["keyframe_selection"]["keyframes"]:
        tags = []
        if kf["is_best_thumbnail"]:
            tags.append("BEST_THUMB")
        if kf["is_scene_boundary"]:
            tags.append("SCENE_CUT")
        if not tags:
            tags.append("ANCHOR")
        print(f"  Frame #{kf['frame_index']:04d} @ {kf['timestamp_seconds']:6.2f}s  [{', '.join(tags)}]")

    print("\n--- Full JSON Optimization Plan ---")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
