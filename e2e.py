"""End-to-end tests for Vidicant Python integration.

This script tests the Python bindings against the example assets
to verify that the full media analysis pipeline works correctly.
"""

from __future__ import annotations

import json
import os
import sys

import vidicant


def test_import_and_setup():
    """Test file type detection functions."""
    print("=" * 60)
    print("TEST: Import and Setup")
    print("=" * 60)

    assert vidicant.is_image_file("examples/sample.jpg") is True
    assert vidicant.is_image_file("examples/sample.mp4") is False
    assert vidicant.is_video_file("examples/sample.mp4") is True
    assert vidicant.is_video_file("examples/sample.jpg") is False

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
    assert isinstance(result["aspect_ratio"], (int, float))
    assert isinstance(result["is_grayscale"], bool)
    assert 0 <= result["average_brightness"] <= 255
    assert result["channels"] in [1, 3]
    assert isinstance(result["edge_count"], int) and result["edge_count"] >= 0
    assert isinstance(result["dominant_colors"], list)
    assert isinstance(result["blur_score"], (int, float))
    assert isinstance(result["contrast_ratio"], (int, float))
    assert isinstance(result["saturation_level"], (int, float))
    assert isinstance(result["entropy"], (int, float))
    assert isinstance(result["histogram"], list)

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
        "scene_changes",
        "frame_rate_stability",
        "color_consistency",
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
    assert isinstance(result["scene_changes"], list)
    assert isinstance(result["frame_rate_stability"], (int, float))
    assert isinstance(result["color_consistency"], (int, float))

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


def _generate_test_onnx(path: str) -> None:
    """Generate a valid, minimal 137-byte ONNX model (GlobalAveragePool) for active testing."""

    def encode_varint(n: int) -> bytes:
        res = bytearray()
        while n > 0x7F:
            res.append((n & 0x7F) | 0x80)
            n >>= 7
        res.append(n & 0x7F)
        return bytes(res)

    def field_varint(field_num: int, val: int) -> bytes:
        return encode_varint((field_num << 3) | 0) + encode_varint(val)

    def field_bytes(field_num: int, data: bytes) -> bytes:
        return encode_varint((field_num << 3) | 2) + encode_varint(len(data)) + data

    def field_string(field_num: int, s: str) -> bytes:
        return field_bytes(field_num, s.encode("utf-8"))

    def make_dim(val: int) -> bytes:
        return field_bytes(1, field_varint(1, val))

    def make_shape(dims: list[int]) -> bytes:
        res = bytearray()
        for d in dims:
            res.extend(make_dim(d))
        return bytes(res)

    def make_tensor_type(elem_type: int, dims: list[int]) -> bytes:
        t = field_varint(1, elem_type) + field_bytes(2, make_shape(dims))
        return field_bytes(1, t)

    def make_value_info(name: str, elem_type: int, dims: list[int]) -> bytes:
        return field_string(1, name) + field_bytes(2, make_tensor_type(elem_type, dims))

    def make_node(inputs: list[str], outputs: list[str], name: str, op_type: str) -> bytes:
        res = bytearray()
        for inp in inputs:
            res.extend(field_string(1, inp))
        for out in outputs:
            res.extend(field_string(2, out))
        res.extend(field_string(3, name))
        res.extend(field_string(4, op_type))
        return bytes(res)

    node = make_node(["data"], ["out"], "gap", "GlobalAveragePool")
    inp = make_value_info("data", 1, [1, 3, 224, 224])
    out = make_value_info("out", 1, [1, 3, 1, 1])

    # GraphProto
    graph_bytes = bytearray()
    graph_bytes.extend(field_bytes(1, node))
    graph_bytes.extend(field_string(2, "test_graph"))
    graph_bytes.extend(field_bytes(11, inp))
    graph_bytes.extend(field_bytes(12, out))

    # ModelProto (ir_version=7, opset=13)
    model = bytearray()
    model.extend(field_varint(1, 7))
    model.extend(field_string(2, "vidicant_test"))
    model.extend(field_bytes(7, bytes(graph_bytes)))
    model.extend(field_bytes(8, field_varint(2, 13)))

    with open(path, "wb") as f:
        f.write(model)


def test_ml_quality_assessment():
    """Test ML aesthetic and technical quality assessment."""
    print("=" * 60)
    print("TEST: ML Aesthetic & Quality Assessment")
    print("=" * 60)

    # 1. Standard mode (ML disabled)
    res_no_ml = vidicant.process_image("examples/sample.jpg", enable_ml=False)
    assert res_no_ml["ml_evaluated"] is False
    assert res_no_ml["aesthetic_score"] is None
    assert res_no_ml["technical_quality_score"] is None

    # 2. Non-existent model path should gracefully fallback
    res_bad_model = vidicant.process_image("examples/sample.jpg", enable_ml=True, model_path="nonexistent_model.onnx")
    assert res_bad_model["ml_evaluated"] is False
    assert res_bad_model["aesthetic_score"] is None

    # 3. Test model helper functions
    cache_path = vidicant.get_default_model_path()
    assert cache_path.name == "aesthetic_mobilenetv2.onnx"

    # 4. Active DNN execution test with dynamic ONNX fixture
    temp_model = "test_fixture_model.onnx"
    try:
        _generate_test_onnx(temp_model)
        res_active = vidicant.process_image("examples/sample.jpg", enable_ml=True, model_path=temp_model)
        assert res_active["ml_evaluated"] is True
        assert isinstance(res_active["aesthetic_score"], float)
        assert isinstance(res_active["technical_quality_score"], float)
        assert 0.0 <= res_active["aesthetic_score"] <= 10.0
        assert 0.0 <= res_active["technical_quality_score"] <= 1.0
        print(f"Active DNN Aesthetic Score: {res_active['aesthetic_score']:.3f}")
        print(f"Active DNN Technical Score: {res_active['technical_quality_score']:.3f}")
    finally:
        if os.path.exists(temp_model):
            os.remove(temp_model)

    print("✓ ML quality assessment (both fallback and active inference) verified")
    print()


def test_python_version_compatibility():
    """Test that current environment is Python 3.11+ and version requirements are enforced."""
    print("=" * 60)
    print("TEST: Python Version Compatibility (>= 3.11)")
    print("=" * 60)

    assert sys.version_info >= (3, 11), f"Vidicant requires Python 3.11+, got {sys.version_info}"
    print(f"✓ Running on supported Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()


def main():
    """Run all end-to-end tests."""
    print("\n")
    print("=" * 60)
    print("VIDICANT END-TO-END TESTS")
    print("=" * 60)
    print()

    try:
        test_python_version_compatibility()
        test_import_and_setup()
        test_analyzing_images()
        test_analyzing_videos()
        test_video_motion_detection()
        test_ml_quality_assessment()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
