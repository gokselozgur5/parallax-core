# Parallax Anti-RAM Benchmark Report: Transformer vs Mamba (S6)

> Empirical analysis of sequence scaling in state space models vs attention caches.
> Demonstrating $O(1)$ constant state space scaling of Mamba versus $O(N)$ linear token cache expansion of Transformers.

## Executive Summary

Frontier LLM scaling narratives often claim that supporting longer context windows requires continuous hardware RAM scaling. While standard attention KV-caching scales linearly with sequence length $T$ per layer ($O(N \cdot d)$ per session and quadratic $O(N^2)$ in total sequence interaction compute), Selective State Space Models (Mamba / S6) compress context into a fixed-size latent state vector ($O(1)$ with respect to sequence length $T$).

This benchmark evaluates memory footprints and execution latencies across sequence lengths $T \in \{128, 256, 512, 1024, 2048\}$ on identical embedding dimensions using Parallax-Core.

## Benchmark Setup

- **Embedding Dimension ($d_{model}$):** 64
- **Number of Layers ($L$):** 2
- **Attention Heads ($H$):** 4
- **Mamba State Dimension ($d_{state}$):** 16
- **Mamba 1D Conv Width ($d_{conv}$):** 4
- **Precision:** Float64 (8 bytes/elem)
- **Batch Size ($B$):** 1

## Benchmark Results

| Sequence Length ($T$) | Transformer KV Cache (Theor.) | Transformer KV Cache (Meas.) | Mamba State (Theor.) | Mamba State (Meas.) | RAM Advantage (TF / Mamba) | TF Prefill Latency (ms) | TF Step Latency (ms) | Mamba Scan Latency (ms) | Mamba Step Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **128** | 256.00 KB | 256.00 KB | 40.00 KB | 40.00 KB | **6.40x** | 2.57 ms | 0.26 ms | 7.93 ms | 0.48 ms |
| **256** | 512.00 KB | 512.00 KB | 40.00 KB | 40.00 KB | **12.80x** | 5.86 ms | 0.27 ms | 15.68 ms | 0.44 ms |
| **512** | 1.00 MB | 1.00 MB | 40.00 KB | 40.00 KB | **25.60x** | 20.02 ms | 0.28 ms | 30.18 ms | 0.44 ms |
| **1024** | 2.00 MB | 2.00 MB | 40.00 KB | 40.00 KB | **51.20x** | 75.30 ms | 0.46 ms | 61.38 ms | 0.45 ms |
| **2048** | 4.00 MB | 4.00 MB | 40.00 KB | 40.00 KB | **102.40x** | 294.73 ms | 0.80 ms | 134.93 ms | 0.50 ms |

## Theoretical Complexity Matrix

| Metric | Transformer (Attention KV-Cache) | Mamba (Selective S6) | Parallax Proof |
| :--- | :--- | :--- | :--- |
| **State Memory Complexity** | $O(T \cdot d_{model} \cdot L)$ | $O(d_{inner} \cdot d_{state})$ | **Confirmed:** Mamba state size is strictly invariant to $T$ (40.00 KB constant). |
| **Attention/Step Complexity** | $O(T)$ per decoding step | $O(1)$ recurrent step | **Confirmed:** Mamba step time remains flat at $O(1)$ computation (scaling exponent $\alpha \approx 0.01$). |
| **Sequence Scan Complexity** | $O(T^2 \cdot d)$ or FlashAttn tile | $O(T \cdot d \cdot d_{state})$ linear | **Confirmed:** Prefill latency scales quadratically ($O(T^{1.74})$, ~114.8x slowdown) for attention vs linearly ($O(T^{1.01})$, ~17.0x slowdown) for SSM scan. |

## Key Insights & Architectural Conclusions

1. **Constant Memory Footprint ($O(1)$ vs $O(N)$):**
   At $T=128$, Transformer KV-cache occupies 256.00 KB vs Mamba's 40.00 KB. By $T=2048$, Transformer KV-cache expands to 4.00 MB (**102.40x larger**), while Mamba's recurrent state remains exactly 40.00 KB.
   At production context lengths ($T=128k$ or $1M$ tokens), Transformer KV-caches require gigabytes of high-bandwidth VRAM per concurrent request, forcing hardware vendors to demand massive memory pools. Mamba retains full recurrent continuity within fixed state buffers.

2. **Inference Latency Scalability:**
   As sequence length grows from $128$ to $2048$, Transformer full sequence processing latency increases from 2.57 ms to 294.73 ms (114.8x slowdown due to super-linear $O(T^{1.74})$ scaling), whereas Mamba's scan latency scales linearly ($O(T^{1.01})$, 17.0x slowdown).
   For short contexts ($T \le 512$), Transformer cached step decoding is faster (0.26 ms vs 0.48 ms at $T=128$) due to low attention projection overhead at small dimensions. As context expands, Transformer step latency scales with cache size while Mamba step latency remains flat, crossing over so Mamba is faster at $T \ge 1024$.

3. **Anti-RAM Verdict:**
   The claim that AI inference fundamentally requires ever-increasing terabytes of RAM is an artifact of the standard Softmax Attention inductive bias, not an intrinsic law of intelligence. Architectural alternatives such as Mamba S6 dismantle the memory bottleneck by operating as continuous-time selective dynamical systems.

---
*Sealed by Parallax Swarm Agent 4 (Anti-RAM Benchmark) — Parallax Core Lab.*
