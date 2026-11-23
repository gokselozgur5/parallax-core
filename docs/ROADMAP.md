# Parallax Core — Roadmap & Architecture Alignment

## Overview & Vision Alignment
Parallax Core has evolved from an initial first-principles autograd prototype into a multi-backend, bare-metal accelerated inference and training engine. The engine provides zero-external-framework neural network primitives, modular model frontends, high-performance attention kernels, native compute backends (C, Rust, Metal JIT), and real-time visualization tooling.

This roadmap consolidates the repository roadmap with `VISION.md`, aligning planned architectural milestones with verified repository deliverables.

---

## Phase 1: Real-World Ingestion & Architecture Frontends (Completed)
- **Model Formats & Loaders:**
  - Standard `.safetensors` and binary `.gguf` weight parsers implemented with zero external framework dependencies (`parallax/safetensors.py`, `parallax/gguf.py`, `parallax/gguf_loader.py`, `parallax/hf_loader.py`).
  - Architecture loaders for contemporary decoder models: Llama 3 (`parallax/models/llama.py`), Qwen 2 and 2.5 (`parallax/models/qwen2.py`), and Qwen 3 (`parallax/models/qwen3.py`).
- **Tokenization:**
  - Native BPE tokenization and HuggingFace-compatible tokenizers (`parallax/tokenizer.py`, `parallax/hf_tokenizer.py`, `parallax/gguf_tokenizer.py`, `crates/parallax-rs/src/bpe.rs`).
- **Autograd & Foundational Modules:**
  - Reverse-mode automatic differentiation DAG engine and tensor abstractions (`parallax/engine.py`, `parallax/tensor.py`, `parallax/gradcheck.py`).
  - Autoregressive generation routines, sampling strategies, and evaluation drivers (`parallax/generator.py`, `parallax/min_p.py`, `parallax/top_a.py`, `parallax/eval.py`).

---

## Phase 2: Bare-Metal Speed & Native Acceleration (Active Focus)
- **Automatic Multi-Backend Dispatch:**
  - Dynamic routing across available execution backends with zero-overhead fallback (`parallax/backends/dispatch.py`, `parallax/backends/registry.py`).
- **C-Engine SIMD Kernels:**
  - Optimized CPU kernels targeting NEON and AVX-512 extensions for GEMM, RMSNorm, SwiGLU, and online FlashAttention (`parallax/csrc/gemm.c`, `parallax/csrc/rmsnorm.c`, `parallax/csrc/flashattn_neon.c`, `parallax/csrc/flashattn_avx512.c`, `parallax/backends/c_backend.py`).
- **Rust Native Acceleration (`crates/parallax-rs`):**
  - High-performance Rust C-ABI engine exposing FlashAttention, quantized matrix multiplication, and memory-mapped tensor storage (`crates/parallax-rs/Cargo.toml`, `crates/parallax-rs/src/lib.rs`, `crates/parallax-rs/src/flash_attn.rs`, `parallax/backends/rust_backend.py`).
- **Apple Silicon Metal JIT:**
  - Zero-copy unified memory buffers and Metal Shading Language compute pipelines (`parallax/backends/metal.py`, `parallax/backends/metal_backend.py`, `parallax/backends/metal_bridge.py`).
- **Serving & Integration:**
  - Lightweight OpenAI-compatible HTTP serving endpoint (`parallax/openai_server.py`, `scripts/serve_openai.py`).

---

## Phase 3: Parallax Studio & Transparent LLM Inspection (Next Milestone)
- **Real-Time Attention & Latent Diagnostics:**
  - Interactive web inspection dashboard and telemetry server (`demo/studio_server.py`, `demo/playground.py`, `demo/visualizer.html`, `parallax/visualizer_server.py`).
- **MoE Routing & Speculative Decoding Visualizer:**
  - Real-time visualization for Expert Choice Routing (ECR) and speculative draft-vs-target acceptance verification (`parallax/moe.py`, `parallax/ecr.py`, `parallax/speculative.py`, `parallax/tree_drafting_v2.py`).
