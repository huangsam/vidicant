#!/usr/bin/env python3
"""
End-to-end tests for Vidicant Python integration.

This script tests the Python bindings against the example assets
to verify that the full media analysis pipeline works correctly.
"""

import vidicant
import json
import sys


def test_import_and_setup():
    """Test file type detection functions."""
    print("=" * 60)
    print("TEST: Import and Setup")
    print("=" * 60)

    assert vidicant.is_image_file("examples/sample.jpg") == True
    assert vidicant.is_image_file("examples/sample.mp4") == False
    assert vidicant.is_video_file("examples/sample.mp4") == True
    assert vidicant.is_video_file("examples/sample.jpg") == False

    print("✓ File type detection works correctly")
    print()


def test_analyzing_images():
    """Test image analysis functionality."""
    print("=" * 60)
    print("TEST: Analyzing Images")
    print("=" * 60)

    result = vidicant.process_image("examples/sample.jpg")

    # Verify all expected fields are present
    expected_fields = [
        "width",
        "height",
        "is_grayscale",
        "average_brightness",
        "channels",
        "edge_count",
        "dominant_colors",
        "blur_score",
    ]

    for field in expected_fields:
        assert field in result, f"Missing field: {field}"

    # Verify field types and reasonable values
    assert isinstance(result["width"], int) and result["width"] > 0
    assert isinstance(result["height"], int) and result["height"] > 0
    assert isinstance(result["is_grayscale"], bool)
    assert 0 <= result["average_brightness"] <= 255
    assert result["channels"] in [1, 3]
    assert isinstance(result["edge_count"], int) and result["edge_count"] >= 0
    assert isinstance(result["dominant_colors"], list)
    assert isinstance(result["blur_score"], (int, float))

    print("Image analysis result:")
    print(json.dumps(result, indent=2))
    print("✓ Image analysis works correctly")
    print()


def test_analyzing_videos():
    """Test video analysis functionality."""
    print("=" * 60)
    print("TEST: Analyzing Videos")
    print("=" * 60)

    result = vidicant.process_video("examples/sample.mp4")

    # Verify all expected fields are present
    expected_fields = [
        "frame_count",
        "fps",
        "width",
        "height",
        "duration_seconds",
        "average_brightness",
        "is_grayscale",
        "motion_score",
        "dominant_colors",
    ]

    for field in expected_fields:
        assert field in result, f"Missing field: {field}"

    # Verify field types and reasonable values
    assert isinstance(result["frame_count"], int) and result["frame_count"] > 0
    assert isinstance(result["fps"], (int, float)) and result["fps"] > 0
    assert isinstance(result["width"], int) and result["width"] > 0
    assert isinstance(result["height"], int) and result["height"] > 0
    assert isinstance(result["duration_seconds"], (int, float)) and result["duration_seconds"] > 0
    assert 0 <= result["average_brightness"] <= 255
    assert isinstance(result["is_grayscale"], bool)
    assert isinstance(result["motion_score"], (int, float))
    assert isinstance(result["dominant_colors"], list)

    print(f"Duration: {result['duration_seconds']} seconds")
    print(f"Resolution: {result['width']}x{result['height']}")
    print(f"Frame rate: {result['fps']} fps")
    print(f"Total frames: {result['frame_count']}")
    print(f"Motion intensity: {result['motion_score']}")
    print("✓ Video analysis works correctly")
    print()


def test_video_motion_detection():
    """Test video motion detection logic."""
    print("=" * 60)
    print("TEST: Video Motion Detection")
    print("=" * 60)

    r = vidicant.process_video("examples/sample.mp4")
    activity = "HIGH" if r["motion_score"] > 0.7 else "LOW"

    print(f"Duration: {r['duration_seconds']:.1f}s, Activity: {activity}")
    print("✓ Video motion detection logic works correctly")
    print()


def main():
    """Run all end-to-end tests."""
    print("\n")
    print("=" * 60)
    print("VIDICANT END-TO-END TESTS")
    print("=" * 60)
    print()

    try:
        test_import_and_setup()
        test_analyzing_images()
        test_analyzing_videos()
        test_video_motion_detection()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
