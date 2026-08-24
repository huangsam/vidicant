# Vidicant Roadmap & TODO

This document tracks planned improvements and feature roadmaps for Vidicant, prioritizing **batch data curation, media quality inspection, and zero-dependency cloud/local pipelines**.

---

## Scope & Philosophy

* **Target Platforms**: macOS (Apple Silicon & Intel) and Linux (x86_64 & aarch64). Windows users are supported via WSL2.
* **Core Value**: Fast, deterministic C++17/OpenCV heuristics with pure stdlib Python `ctypes` bindings.
* **Out of Scope (Anti-Goals)**:
  * In-the-loop GPU training data loaders (handled better by PyTorch/DALI).
  * Native Windows MSVC DLL toolchains (handled seamlessly via WSL2).
  * Adding external runtime dependencies to the Python package.

---

## Tier 1: High-Impact Batch & Ingestion Essentials

Immediate wins to enhance throughput and unlock cloud microservices with zero disk I/O overhead.

- [ ] **In-Memory Byte Buffer C-ABI (`process_image_bytes`)**
  - Expose `vidicant_process_image_bytes(const uint8_t *buffer, size_t len)` in C-ABI using `cv::imdecode`.
  - Add `vidicant.process_image_bytes(data: bytes)` in Python for streaming cloud workers (AWS Lambda, Celery, S3 streaming) without writing temporary files to `/tmp`.
- [ ] **Streaming JSON Lines (`.jsonl`) & CSV CLI Output**
  - Add `--format jsonl` and `--format csv` to `vidicant_cli` to stream per-file records line-by-line.
  - Enable direct, memory-efficient ingestion into DuckDB, Polars, Pandas, and data lakes on 100k+ file datasets.
- [ ] **Near-Duplicate Clustering CLI (`dedupe`)**
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
