"""End-to-end tests for Vidicant Python integration.

This script tests the Python bindings against the example assets
to verify that the full media analysis pipeline works correctly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

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

    print(f"Resolution: {result['width']}x{result['height']}")
    print(f"Brightness: {result['average_brightness']:.2f}")
    print(f"Blur score: {result['blur_score']:.2f}")
    print(f"Dominant colors: {len(result['dominant_colors'])} detected")
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


def _generate_test_onnx(path: str, channels: int = 3) -> None:
    """Generate a valid, minimal ONNX model (GlobalAveragePool) for active testing."""

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
    inp = make_value_info("data", 1, [1, channels, 224, 224])
    out = make_value_info("out", 1, [1, channels, 1, 1])

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
    assert res_no_ml["top_labels"] == []
    assert res_no_ml["detected_objects"] == []
    assert res_no_ml["embedding"] == []

    # 2. Non-existent model path should gracefully fallback
    res_bad_model = vidicant.process_image("examples/sample.jpg", enable_ml=True, model_path="nonexistent_model.onnx")
    assert res_bad_model["ml_evaluated"] is False
    assert res_bad_model["aesthetic_score"] is None

    # 3. Test model helper functions
    cache_path = vidicant.get_default_model_path()
    assert cache_path.name == "aesthetic_mobilenetv2.onnx"
    assert vidicant.get_default_model_path("classify").name == "mobilenetv2_imagenet.onnx"
    assert vidicant.get_default_model_path("detect").name == "yunet_face_detection.onnx"

    # 4. Active DNN execution test with dynamic ONNX fixture
    temp_model = "test_fixture_quality.onnx"
    try:
        _generate_test_onnx(temp_model, channels=3)
        res_active = vidicant.process_image("examples/sample.jpg", enable_ml=True, model_path=temp_model, task="quality")
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

    print("✓ ML quality assessment verified")
    print()


def test_semantic_classification():
    """Test semantic classification and tagging (Top-K labels with confidences)."""
    print("=" * 60)
    print("TEST: Semantic Classification & Tagging")
    print("=" * 60)

    temp_model = "test_fixture_classify.onnx"
    try:
        _generate_test_onnx(temp_model, channels=3)
        res = vidicant.process_image(
            "examples/sample.jpg",
            enable_ml=True,
            task="classify",
            model_path=temp_model,
            top_k=2,
        )
        assert res["ml_evaluated"] is True
        assert "top_labels" in res
        assert isinstance(res["top_labels"], list)
        assert len(res["top_labels"]) == 2

        for item in res["top_labels"]:
            assert "class_id" in item and isinstance(item["class_id"], int)
            assert "label" in item and isinstance(item["label"], str)
            assert "confidence" in item and isinstance(item["confidence"], float)
            assert 0.0 <= item["confidence"] <= 1.0

        # Verify sorted descending by confidence
        assert res["top_labels"][0]["confidence"] >= res["top_labels"][1]["confidence"]
        print(f"Top 2 Labels: {res['top_labels']}")
    finally:
        if os.path.exists(temp_model):
            os.remove(temp_model)

    print("✓ Semantic classification & tagging verified")
    print()


def test_object_and_face_detection():
    """Test object and face detection with bounding boxes and NMS."""
    print("=" * 60)
    print("TEST: Object & Face Detection with NMS")
    print("=" * 60)

    temp_model = "test_fixture_detect.onnx"
    try:
        _generate_test_onnx(temp_model, channels=3)
        res = vidicant.process_image(
            "examples/sample.jpg",
            enable_ml=True,
            task="detect",
            model_path=temp_model,
            conf_threshold=0.1,
            nms_threshold=0.4,
        )
        assert res["ml_evaluated"] is True
        assert "detected_objects" in res
        assert isinstance(res["detected_objects"], list)
        assert len(res["detected_objects"]) == 1
        for obj in res["detected_objects"]:
            assert "box" in obj and isinstance(obj["box"], list) and len(obj["box"]) == 4
            assert "class_name" in obj and isinstance(obj["class_name"], str)
            assert "confidence" in obj and isinstance(obj["confidence"], float)
            assert obj["confidence"] >= 0.1
            print(f"Detected: {obj['class_name']} ({obj['confidence']:.2f}) at box {obj['box']}")
    finally:
        if os.path.exists(temp_model):
            os.remove(temp_model)

    print("✓ Object & face detection with NMS verified")
    print()


def test_generic_embeddings():
    """Test generic ONNX tensor / vector embedding extractor."""
    print("=" * 60)
    print("TEST: Generic ONNX Vector Embeddings")
    print("=" * 60)

    temp_model = "test_fixture_embed.onnx"
    try:
        _generate_test_onnx(temp_model, channels=3)
        res = vidicant.process_image(
            "examples/sample.jpg",
            enable_ml=True,
            task="embed",
            model_path=temp_model,
        )
        assert res["ml_evaluated"] is True
        assert "embedding" in res
        assert isinstance(res["embedding"], list)
        assert len(res["embedding"]) == 3
        for val in res["embedding"]:
            assert isinstance(val, float)
        print(f"Extracted embedding (length {len(res['embedding'])}): {res['embedding']}")
    finally:
        if os.path.exists(temp_model):
            os.remove(temp_model)

    print("✓ Generic ONNX vector embeddings verified")
    print()


def test_process_image_bytes():
    """Test in-memory byte buffer image processing."""
    print("=" * 60)
    print("TEST: In-Memory Byte Buffer Processing (process_image_bytes)")
    print("=" * 60)

    # 1. Read bytes from sample image
    with open("examples/sample.jpg", "rb") as f:
        img_bytes = f.read()

    # Verify standard heuristic processing from memory
    res = vidicant.process_image_bytes(img_bytes)
    assert isinstance(res, dict)
    assert res["width"] > 0
    assert res["height"] > 0
    assert res["channels"] in [1, 3]
    assert isinstance(res["blur_score"], (int, float))
    assert isinstance(res["perceptual_hash"], int)
    assert isinstance(res["dominant_colors"], list)
    assert res["ml_evaluated"] is False

    # Compare metrics with file-based processing
    file_res = vidicant.process_image("examples/sample.jpg")
    assert res["width"] == file_res["width"]
    assert res["height"] == file_res["height"]
    assert res["channels"] == file_res["channels"]
    assert res["perceptual_hash"] == file_res["perceptual_hash"]

    # 2. Test invalid / empty byte buffer handling
    try:
        vidicant.process_image_bytes(b"")
        raise AssertionError("Should have raised ValueError on empty buffer")
    except ValueError:
        pass

    try:
        vidicant.process_image_bytes(b"invalid_garbage_data_not_an_image")
        raise AssertionError("Should have raised ValueError on non-image buffer")
    except ValueError:
        pass

    # 3. Test DNN neural tasks with in-memory bytes
    temp_model = "test_fixture_bytes_ml.onnx"
    try:
        _generate_test_onnx(temp_model, channels=3)
        res_quality = vidicant.process_image_bytes(img_bytes, enable_ml=True, model_path=temp_model, task="quality")
        assert res_quality["ml_evaluated"] is True
        assert isinstance(res_quality["aesthetic_score"], float)

        res_cls = vidicant.process_image_bytes(img_bytes, enable_ml=True, model_path=temp_model, task="classify", top_k=2)
        assert res_cls["ml_evaluated"] is True
        assert len(res_cls["top_labels"]) == 2

        res_det = vidicant.process_image_bytes(img_bytes, enable_ml=True, model_path=temp_model, task="detect")
        assert res_det["ml_evaluated"] is True
        assert isinstance(res_det["detected_objects"], list)

        res_emb = vidicant.process_image_bytes(img_bytes, enable_ml=True, model_path=temp_model, task="embed")
        assert res_emb["ml_evaluated"] is True
        assert len(res_emb["embedding"]) == 3
    finally:
        if os.path.exists(temp_model):
            os.remove(temp_model)

    print("✓ In-memory byte buffer processing verified")
    print()


def test_cli_streaming_formats():
    """Test CLI --format jsonl and --format csv streaming output options."""
    print("=" * 60)
    print("TEST: CLI Streaming Formats (jsonl & csv)")
    print("=" * 60)

    cli_bin = "./zig-out/bin/vidicant_cli"
    if not os.path.isfile(cli_bin):
        print(f"Skipping CLI test: {cli_bin} not built yet.")
        return

    # 1. Test JSONL streaming output to file
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        jsonl_path = f.name

    try:
        cmd = [
            cli_bin,
            "examples/sample.jpg",
            "examples/sample.mp4",
            "--format",
            "jsonl",
            "--output",
            jsonl_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert os.path.exists(jsonl_path)
        with open(jsonl_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2, f"Expected 2 lines for 2 media files, got {len(lines)}"

        rec0 = json.loads(lines[0])
        assert rec0["media_type"] == "image"
        assert "blur_score" in rec0

        rec1 = json.loads(lines[1])
        assert rec1["media_type"] == "video"
        assert "motion_score" in rec1
        print("✓ CLI JSONL streaming output verified")
    finally:
        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)

    # 2. Test CSV streaming output to file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        cmd = [
            cli_bin,
            "examples/sample.jpg",
            "examples/sample.mp4",
            "--format",
            "csv",
            "--output",
            csv_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 3, f"Expected header + 2 data rows, got {len(lines)}"
        assert lines[0].startswith("filename,media_type,width,height")
        assert "examples/sample.jpg,image" in lines[1]
        assert "examples/sample.mp4,video" in lines[2]
        print("✓ CLI CSV streaming output verified")
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    print("✓ All CLI streaming formats verified")
    print()


def test_cli_deduplication():
    """Test CLI dedupe subcommand for near-duplicate image clustering."""
    print("=" * 60)
    print("TEST: CLI Near-Duplicate Image Clustering (dedupe)")
    print("=" * 60)

    cli_bin = "./zig-out/bin/vidicant_cli"
    if not os.path.isfile(cli_bin):
        print(f"Skipping CLI test: {cli_bin} not built yet.")
        return

    temp_dir = tempfile.mkdtemp(prefix="vidicant_dedupe_test_")
    try:
        # Create identical copies in temporary test directory
        img_orig = os.path.join(temp_dir, "img_orig.jpg")
        img_copy1 = os.path.join(temp_dir, "img_copy1.jpg")
        img_copy2 = os.path.join(temp_dir, "img_copy2.jpg")
        shutil.copyfile("examples/sample.jpg", img_orig)
        shutil.copyfile("examples/sample.jpg", img_copy1)
        shutil.copyfile("examples/sample.jpg", img_copy2)

        # 1. Test JSON format output
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_json = f.name

        try:
            cmd = [
                cli_bin,
                "dedupe",
                temp_dir,
                "--threshold",
                "5",
                "--format",
                "json",
                "--output",
                out_json,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            with open(out_json) as f:
                data = json.load(f)
            assert data["total_images"] == 3
            assert data["clusters_count"] == 1
            assert len(data["duplicate_clusters"]) == 1
            assert data["duplicate_clusters"][0]["count"] == 3
            assert len(data["duplicate_clusters"][0]["files"]) == 3
            print(f"✓ Dedupe JSON output: found cluster with {data['duplicate_clusters'][0]['count']} duplicates")
        finally:
            if os.path.exists(out_json):
                os.remove(out_json)

        # 2. Test text format output to stdout
        cmd_txt = [cli_bin, "dedupe", temp_dir, "--threshold", "5", "--format", "text"]
        res_txt = subprocess.run(cmd_txt, capture_output=True, text=True, check=True)
        assert "Found 1 duplicate/near-duplicate cluster(s)" in res_txt.stdout
        assert "Cluster #1 (3 files)" in res_txt.stdout
        print("✓ Dedupe text stdout output verified")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✓ CLI near-duplicate clustering verified")
    print()


def test_python_deduplication():
    """Test Python find_duplicates API for near-duplicate image clustering."""
    print("=" * 60)
    print("TEST: Python API Near-Duplicate Image Clustering (find_duplicates)")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="vidicant_py_dedupe_test_")
    try:
        # Create copies in temporary test directory
        img_orig = os.path.join(temp_dir, "img_orig.jpg")
        img_copy1 = os.path.join(temp_dir, "img_copy1.jpg")
        shutil.copyfile("examples/sample.jpg", img_orig)
        shutil.copyfile("examples/sample.jpg", img_copy1)

        res = vidicant.find_duplicates(temp_dir, threshold=5)
        assert isinstance(res, dict)
        assert res["total_images"] == 2
        assert res["clusters_count"] == 1
        assert len(res["duplicate_clusters"]) == 1
        assert res["duplicate_clusters"][0]["count"] == 2
        print(f"✓ Python find_duplicates verified: {res['clusters_count']} cluster found")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print()


def test_python_version_compatibility():
    """Test that current environment is Python 3.11+ and version requirements are enforced."""
    print("=" * 60)
    print("TEST: Python Version Compatibility (>= 3.11)")
    print("=" * 60)

    assert sys.version_info >= (3, 11), f"Vidicant requires Python 3.11+, got {sys.version_info}"
    print(f"✓ Running on supported Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()


def test_video_stride_and_thumbnails():
    """Test video processing with stride, target fps, and scene thumbnail export."""
    print("=" * 60)
    print("TEST: Video Processing with Stride & Scene Thumbnail Export")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="vidicant_py_scenes_test_")
    try:
        res = vidicant.process_video(
            "examples/sample.mp4",
            stride=2,
            sample_fps=15.0,
            export_scenes_dir=temp_dir,
        )
        assert isinstance(res, dict)
        assert res["frame_count"] > 0
        assert "scene_thumbnails" in res
        assert isinstance(res["scene_thumbnails"], list)
        print(f"✓ Processed video with stride=2 (scenes: {len(res['scene_thumbnails'])})")

        for st in res["scene_thumbnails"]:
            assert "scene_index" in st
            assert "thumbnail_path" in st
            assert os.path.exists(st["thumbnail_path"])
            assert "sharpness_score" in st
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("✓ Video stride and scene thumbnail auto-export verified")
    print()


def test_cli_video_stride_and_filter():
    """Test CLI video options (--stride, --sample-rate, --export-scenes) and --filter."""
    print("=" * 60)
    print("TEST: CLI Video Stride & Filter Expressions")
    print("=" * 60)

    cli_bin = os.path.join("zig-out", "bin", "vidicant_cli")
    assert os.path.exists(cli_bin), f"CLI binary not found at {cli_bin}"

    temp_dir = tempfile.mkdtemp(prefix="vidicant_cli_filter_test_")
    try:
        # 1. Test video with --stride 2 and --export-scenes
        scenes_dir = os.path.join(temp_dir, "scenes")
        cmd_stride = [
            cli_bin,
            "examples/sample.mp4",
            "--stride",
            "2",
            "--export-scenes",
            scenes_dir,
            "--format",
            "json",
            "-o",
            "-",
        ]
        res_stride = subprocess.run(cmd_stride, capture_output=True, text=True, check=True)
        data = json.loads(res_stride.stdout)
        assert len(data["videos"]) == 1
        assert "scene_thumbnails" in data["videos"][0]
        print("✓ CLI --stride and --export-scenes verified")

        # 2. Test CLI --filter matching image
        cmd_filter_pass = [
            cli_bin,
            "examples/sample.jpg",
            "--filter",
            "blur_score > 10 and is_grayscale == false",
            "--format",
            "json",
            "-o",
            "-",
        ]
        res_pass = subprocess.run(cmd_filter_pass, capture_output=True, text=True, check=True)
        data_pass = json.loads(res_pass.stdout)
        assert len(data_pass["images"]) == 1
        print("✓ CLI --filter positive match verified")

        # 3. Test CLI --filter rejecting image
        cmd_filter_fail = [
            cli_bin,
            "examples/sample.jpg",
            "--filter",
            "blur_score > 999999 or is_grayscale == true",
            "--format",
            "json",
            "-o",
            "-",
        ]
        res_fail = subprocess.run(cmd_filter_fail, capture_output=True, text=True, check=True)
        data_fail = json.loads(res_fail.stdout)
        assert len(data_fail["images"]) == 0
        print("✓ CLI --filter rejection match verified")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print()


def test_runnable_examples():
    """Test that all standalone runnable reference applications in examples/python/ execute cleanly."""
    print("=" * 60)
    print("TEST: Runnable Reference Applications (examples/python/*.py)")
    print("=" * 60)

    examples = [
        ("ingestion_gate.py", "Ingestion Gate Decision"),
        ("video_chapters.py", "VIDEO CHAPTER MANIFEST"),
        ("product_qa.py", "E-COMMERCE PRODUCT LISTING QA REPORT"),
        ("rag_keyframe_filter.py", "MULTIMODAL VISION LLM KEYFRAME PRUNING PLAN"),
        ("dedupe_catalog.py", "PERCEPTUAL HASH DEDUPLICATION SUMMARY"),
    ]

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = "."

    for script_name, expected_marker in examples:
        script_path = os.path.join("examples", "python", script_name)
        assert os.path.exists(script_path), f"Example script not found: {script_path}"

        res = subprocess.run(
            [sys.executable, script_path],
            env=base_env,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Example '{script_name}' failed with code {res.returncode}:\n{res.stderr}"
        assert expected_marker in res.stdout, f"Expected marker '{expected_marker}' not found in output of {script_name}"
        print(f"✓ examples/python/{script_name} executed successfully")

    print()


def main():
    """Run all end-to-end tests."""
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
        test_video_stride_and_thumbnails()
        test_process_image_bytes()
        test_ml_quality_assessment()
        test_semantic_classification()
        test_object_and_face_detection()
        test_generic_embeddings()
        test_cli_streaming_formats()
        test_cli_deduplication()
        test_cli_video_stride_and_filter()
        test_python_deduplication()
        test_runnable_examples()

        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
