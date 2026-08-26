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

## Active Roadmap (Planned)

### Neural Usability & Wheel Distribution (Tier 3)

- [ ] **Dynamic Neural Preprocessing & Aspect-Ratio Letterboxing**
    - Support arbitrary model input dimensions (e.g. $336 \times 336$ for CLIP, $640 \times 640$ for YOLO) rather than fixed $224 \times 224$.
    - Add letterbox padding options to preserve image aspect ratios before model inference.
- [ ] **Automated Multi-Arch CI/CD & Prebuilt Wheels**
    - Configure GitHub Actions matrix to build native shared libraries (`.dylib` on macOS arm64/x86_64, `.so` on Linux x86_64/aarch64).
    - Package native binaries inside pure Python wheels for instant `pip install vidicant` without requiring local compilation or system dependencies.

---

## Completed Milestones

### Batch & Ingestion Essentials (Tier 1)
- [x] **In-Memory Byte Buffer Processing (`process_image_bytes`)**: In-memory C-ABI & Python bindings for zero-disk I/O microservices.
- [x] **Streaming JSON Lines (`.jsonl`) & CSV CLI Output**: Stream per-file records line-by-line for direct DuckDB, Polars, and Pandas ingestion.
- [x] **Near-Duplicate Clustering CLI (`dedupe`)**: Perceptual 64-bit `dHash` clustering with Hamming distance thresholds.

### Video Performance & Pipeline Automation (Tier 2)
- [x] **Video Frame Sampling Stride (`--stride` / `--sample-rate`)**: 30x–60x speedup by skipping frame decoding during video inspection.
- [x] **Scene Cut Thumbnail Auto-Export (`--export-scenes`)**: In-flight extraction and saving of the sharpest scene transition frames.
- [x] **Declarative Quality Filtering (`--filter`)**: Streamingly evaluate boolean/comparison filter expressions against media metrics.

### Runnable Reference Applications (Tier 4)
- [x] **In-Memory Ingestion Gate (`examples/python/ingestion_gate.py`)**: Upload preflight QA checking blur, exposure, and noise.
- [x] **Video Chapter Indexer (`examples/python/video_chapters.py`)**: Scene cuts, pacing metrics, and optimal thumbnail extraction.
- [x] **Product Listing QA (`examples/python/product_qa.py`)**: E-commerce white balance, texture, and dominant palette extraction.
- [x] **Multimodal RAG Keyframe Pruner (`examples/python/rag_keyframe_filter.py`)**: Vision LLM keyframe pruning and token optimizer.
- [x] **Catalog Deduplication (`examples/python/dedupe_catalog.py`)**: Perceptual dHash near-duplicate image clustering.
- [x] **Go / Cgo High-Concurrency Server (`examples/go/main.go`)**: Concurrent Go HTTP server and 4-worker goroutine pool.
