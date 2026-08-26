# Vidicant Roadmap & TODO

This document tracks planned improvements and feature roadmaps for Vidicant, prioritizing **batch data curation, media quality inspection, and zero-dependency cloud/local pipelines**.

---

## Scope & Philosophy

- **Target Platforms**: macOS (Apple Silicon & Intel) and Linux (x86_64 & aarch64). Windows users can run seamlessly via WSL2.
- **Core Value**: Fast, deterministic C++17/OpenCV heuristics with pure stdlib Python `ctypes` bindings.
- **Out of Scope (Anti-Goals)**:
    - In-the-loop GPU training data loaders (handled better by PyTorch/DALI).
    - Native MSVC toolchain maintenance (WSL2 recommended for Windows development).
    - Adding external runtime dependencies to the Python package.

---

## Tier 1: High-Impact Batch & Ingestion Essentials

Immediate wins to enhance throughput and unlock cloud microservices with zero disk I/O overhead.

- [x] **In-Memory Byte Buffer C-ABI (`process_image_bytes`)**
    - Expose `vidicant_process_image_bytes(const uint8_t *buffer, size_t len)` in C-ABI using `cv::imdecode`.
    - Add `vidicant.process_image_bytes(data: bytes)` in Python for streaming cloud workers (AWS Lambda, Celery, S3 streaming) without writing temporary files to `/tmp`.
- [x] **Streaming JSON Lines (`.jsonl`) & CSV CLI Output**
    - Add `--format jsonl` and `--format csv` to `vidicant_cli` to stream per-file records line-by-line.
    - Enable direct, memory-efficient ingestion into DuckDB, Polars, Pandas, and data lakes on 100k+ file datasets.
- [x] **Near-Duplicate Clustering CLI (`dedupe`)**
    - Add a dedicated deduplication command (`vidicant_cli dedupe <dir> --threshold <int>`).
    - Cluster near-duplicate images using Hamming distance on 64-bit `dHash` perceptual hashes.

---

## Tier 2: Video Performance & Pipeline Automation

Optimizations and automation tools for processing large video collections efficiently.

- [ ] **Video Frame Sampling Stride (`--stride` / `--sample-rate`)**
    - Allow skipping frames during video analysis (e.g. sample 1 frame per second or every $N$ frames) rather than decoding all consecutive frames.
    - Achieve a 30x–50x speedup when computing motion scores and scene cuts on long high-resolution videos.
- [ ] **Scene Cut Thumbnail Auto-Export (`--export-scenes`)**
    - Automatically extract and save the sharpest, highest-quality thumbnail for each detected scene change to an output directory.
- [ ] **Quality Filter Presets in CLI (`--filter`)**
    - Provide declarative filter expressions to quickly isolate clean assets:
    ```bash
    vidicant_cli ./raw_dataset/ --filter "blur_score > 60 and contrast_ratio > 0.2 and width >= 512" -o clean_manifest.txt
    ```

---

## Tier 3: Neural Usability & Distribution

Enhance ONNX model integration flexibility and simplify distribution across platforms.

- [ ] **Dynamic Neural Preprocessing & Aspect-Ratio Letterboxing**
    - Upgrade `runDNNInference` to support arbitrary input dimensions (e.g. $336 \times 336$ for CLIP, $640 \times 640$ for YOLO) rather than fixed $224 \times 224$.
    - Add letterbox padding options to preserve image aspect ratios before model forward pass.
- [ ] **Automated Multi-Arch CI/CD & Prebuilt Wheels**
    - Configure GitHub Actions matrix to build native shared libraries (`.dylib` on macOS arm64/x86_64, `.so` on Linux x86_64/aarch64).
    - Package native binaries inside pure Python wheels for instant `pip install vidicant` without requiring local compilation or system dependencies.

---

## Tier 4: Runnable Reference Applications (Dogfooding & Real-World Integration)

Bridge `docs/use_cases.md` with executable, self-contained reference scripts in `examples/python/` and `examples/go/` that run against bundled sample assets (`examples/sample.jpg`, `examples/sample.mp4`).

- [x] **In-Memory Cloud Ingestion Gate (`examples/python/ingestion_gate.py`)**
    - Implement a pre-flight upload validator checking blur, exposure, and noise thresholds using `process_image_bytes` with zero disk I/O.
- [x] **Video Chapter & Thumbnail Indexer (`examples/python/video_chapters.py`)**
    - Implement a chapter generator that segments scene cuts, computes pacing stats, and identifies optimal thumbnail frames.
- [x] **E-Commerce Product QA & Color Indexer (`examples/python/product_qa.py`)**
    - Validate product image guidelines (white balance deviation, GLCM texture contrast, and dominant color palette extraction).
- [x] **Multimodal RAG Keyframe Pruner (`examples/python/rag_keyframe_filter.py`)**
    - Prune redundant video frames using scene changes and motion magnitude to minimize Vision LLM token costs.
- [x] **Catalog Deduplication Script (`examples/python/dedupe_catalog.py`)**
    - Cluster near-duplicate images using 64-bit dHash perceptual hashes and Hamming distance thresholds.
- [x] **Go Cgo Concurrent Worker & Upload Server (`examples/go/main.go`)**
    - Implement a concurrent Go service combining `net/http` in-memory upload validation with a goroutine-backed media worker pool.
