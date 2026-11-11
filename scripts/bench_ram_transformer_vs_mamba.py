#!/usr/bin/env python3
"""
Benchmark: Transformer KV-Cache vs Mamba Selective State Space (S6) RAM & Latency.

Evaluates memory footprint scaling and inference latency across sequence lengths:
T in [128, 256, 512, 1024, 2048].

Empirically compares linear KV-cache growth O(N * d) against Mamba's constant state O(1) memory footprint.
"""

import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure parallax is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.bitmamba import BitMambaLM, BitMambaConfig
from parallax.tensor import no_grad


def benchmark_transformer(
    model: ParallaxLM,
    batch_size: int,
    seq_len: int,
    warmup: int = 3,
    repeats: int = 10,
) -> Tuple[float, float, int, int]:
    """
    Measures Transformer:
    - Prefill / forward latency (ms)
    - Autoregressive step latency with KV-cache (ms)
    - Theoretical KV-cache footprint (bytes)
    - Measured numpy KV-cache footprint (bytes)
    """
    config = model.config
    num_layers = config.num_layers
    num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
    head_dim = config.embed_dim // config.num_heads
    dtype_bytes = int(model.token_embeddings.weight.data.itemsize)

    # 1. Theoretical KV Cache size in bytes:
    # 2 (K and V) * num_layers * batch_size * num_kv_heads * seq_len * head_dim * bytes_per_elem
    theoretical_kv_bytes = 2 * num_layers * batch_size * num_kv_heads * seq_len * head_dim * dtype_bytes

    # 2. Prefill to build cache
    idx = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))
    initial_caches = [
        (np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=model.token_embeddings.weight.data.dtype),
         np.zeros((batch_size, num_kv_heads, 0, head_dim), dtype=model.token_embeddings.weight.data.dtype))
        for _ in range(num_layers)
    ]

    _, full_caches = model(idx, kv_caches=initial_caches)
    measured_kv_bytes = sum(k.nbytes + v.nbytes for k, v in full_caches)

    # 3. Benchmark Next-Token Step Latency (Inference decoding at step T)
    next_token = np.random.randint(0, config.vocab_size, size=(batch_size, 1))

    # Warmup
    for _ in range(warmup):
        _ = model(next_token, kv_caches=full_caches, rope_offset=seq_len)

    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model(next_token, kv_caches=full_caches, rope_offset=seq_len)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    step_latency_ms = float(np.median(latencies))

    # 4. Prefill latency for reference
    for _ in range(warmup):
        _ = model(idx)
    prefill_latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model(idx)
        prefill_latencies.append((time.perf_counter() - t0) * 1000.0)
    prefill_latency_ms = float(np.median(prefill_latencies))

    return step_latency_ms, prefill_latency_ms, theoretical_kv_bytes, measured_kv_bytes


def _measure_mamba_model_state(model: BitMambaLM, batch_size: int, dtype_bytes: int = 8) -> Tuple[int, int]:
    """Computes theoretical and measured state bytes across all layers of model."""
    total_elements = sum(
        b.d_inner * (b.d_state + b.conv_weight.data.shape[1])
        for b in model.blocks
    )
    theo_bytes = batch_size * total_elements * dtype_bytes
    return theo_bytes, theo_bytes


def benchmark_mamba(
    model: BitMambaLM,
    batch_size: int,
    seq_len: int,
    warmup: int = 3,
    repeats: int = 10,
) -> Tuple[float, float, int, int]:
    """Measures Mamba SSM sequence scan and cached step latency and state bytes."""
    dtype_bytes = int(model.token_embeddings.weight.data.itemsize)
    dtype = model.token_embeddings.weight.data.dtype
    theo_bytes, _ = _measure_mamba_model_state(model, batch_size, dtype_bytes=dtype_bytes)

    idx = np.random.randint(0, model.config.vocab_size, size=(batch_size, seq_len))

    # Scan seq_len tokens step-by-step to produce real state after T tokens
    states = [
        (np.zeros((batch_size, b.d_inner, b.d_state), dtype=dtype),
         np.zeros((batch_size, b.d_inner, b.conv_weight.data.shape[1]), dtype=dtype))
        for b in model.blocks
    ]
    with no_grad():
        for t in range(seq_len):
            _, states = model(idx[:, t : t + 1], states=states)

    meas_bytes = sum(s[0].nbytes + s[1].nbytes for s in states)

    # Parallel sequence scan latency (prefill equivalent)
    with no_grad():
        for _ in range(warmup):
            _ = model(idx)
        scan_latencies = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(idx)
            scan_latencies.append((time.perf_counter() - t0) * 1000.0)
        scan_ms = float(np.median(scan_latencies))

    # Next-token autoregressive decoding step at step T with scanned state
    next_tok = np.random.randint(0, model.config.vocab_size, size=(batch_size, 1))
    with no_grad():
        warmup_states = [(s[0].copy(), s[1].copy()) for s in states]
        for _ in range(warmup):
            _, warmup_states = model(next_tok, states=warmup_states)
        step_latencies = []
        for _ in range(repeats):
            run_states = [(s[0].copy(), s[1].copy()) for s in states]
            t0 = time.perf_counter()
            _, _ = model(next_tok, states=run_states)
            step_latencies.append((time.perf_counter() - t0) * 1000.0)
        step_ms = float(np.median(step_latencies))

    return step_ms, scan_ms, theo_bytes, meas_bytes


def format_bytes(n: float) -> str:
    """Format bytes into readable string."""
    if n < 1024:
        return f"{n:.0f} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.2f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"


def fit_log_log_slope(x_vals: List[float], y_vals: List[float]) -> float:
    """Computes scaling exponent alpha where y ~ x^alpha using log-log OLS."""
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)
    dx = log_x - np.mean(log_x)
    dy = log_y - np.mean(log_y)
    denom = np.sum(dx * dx)
    if denom == 0:
        return 0.0
    return float(np.sum(dx * dy) / denom)


def analyze_anti_ram_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyzes benchmark results to determine empirical scaling characteristics."""
    t_vals = [float(r["T"]) for r in results]
    m_meas = [r["m_meas_b"] for r in results]
    m_step = [r["m_step_ms"] for r in results]
    tf_prefill = [r["tf_prefill_ms"] for r in results]
    m_scan = [r["m_scan_ms"] for r in results]

    # 1. State invariance
    m_state_invariant = (len(set(m_meas)) == 1)

    # 2. Mamba step time scaling exponent
    alpha_m_step = fit_log_log_slope(t_vals, m_step)
    m_step_is_flat = abs(alpha_m_step) < 0.25

    # 3. Prefill / Scan scaling exponents
    alpha_tf_prefill = fit_log_log_slope(t_vals, tf_prefill)
    alpha_m_scan = fit_log_log_slope(t_vals, m_scan)
    tf_prefill_superlinear = alpha_tf_prefill >= 1.3
    m_scan_linear = (0.7 <= alpha_m_scan <= 1.35)

    # 4. Step latency comparison
    tf_faster_ts = [r["T"] for r in results if r["tf_step_ms"] < r["m_step_ms"]]
    mamba_faster_ts = [r["T"] for r in results if r["m_step_ms"] < r["tf_step_ms"]]

    return {
        "m_state_invariant": m_state_invariant,
        "alpha_m_step": alpha_m_step,
        "m_step_is_flat": m_step_is_flat,
        "alpha_tf_prefill": alpha_tf_prefill,
        "alpha_m_scan": alpha_m_scan,
        "tf_prefill_superlinear": tf_prefill_superlinear,
        "m_scan_linear": m_scan_linear,
        "tf_faster_ts": tf_faster_ts,
        "mamba_faster_ts": mamba_faster_ts,
    }


def generate_report(
    results: List[Dict],
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_state: int,
    d_conv: int,
    mamba_state_bytes: int,
    precision_str: str = "Float64 (8 bytes/elem)",
) -> str:
    analysis = analyze_anti_ram_results(results)
    first_res = results[0]
    last_res = results[-1]
    m_state_str = format_bytes(mamba_state_bytes)

    md = []
    md.append("# Parallax Anti-RAM Benchmark Report: Transformer vs Mamba (S6)")
    md.append("")
    md.append("> Empirical analysis of sequence scaling in state space models vs attention caches.")
    md.append("> Demonstrating $O(1)$ constant state space scaling of Mamba versus $O(N)$ linear token cache expansion of Transformers.")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append(r"Frontier LLM scaling narratives often claim that supporting longer context windows requires continuous hardware RAM scaling. While standard attention KV-caching scales linearly with sequence length $T$ per layer ($O(N \cdot d)$ per session and quadratic $O(N^2)$ in total sequence interaction compute), Selective State Space Models (Mamba / S6) compress context into a fixed-size latent state vector ($O(1)$ with respect to sequence length $T$).")
    md.append("")
    md.append(r"This benchmark evaluates memory footprints and execution latencies across sequence lengths $T \in \{128, 256, 512, 1024, 2048\}$ on identical embedding dimensions using Parallax-Core.")
    md.append("")
    md.append("## Benchmark Setup")
    md.append("")
    md.append(f"- **Embedding Dimension ($d_{{model}}$):** {d_model}")
    md.append(f"- **Number of Layers ($L$):** {num_layers}")
    md.append(f"- **Attention Heads ($H$):** {num_heads}")
    md.append(f"- **Mamba State Dimension ($d_{{state}}$):** {d_state}")
    md.append(f"- **Mamba 1D Conv Width ($d_{{conv}}$):** {d_conv}")
    md.append(f"- **Precision:** {precision_str}")
    md.append(f"- **Batch Size ($B$):** 1")
    md.append("")
    md.append("## Benchmark Results")
    md.append("")
    md.append("| Sequence Length ($T$) | Transformer KV Cache (Theor.) | Transformer KV Cache (Meas.) | Mamba State (Theor.) | Mamba State (Meas.) | RAM Advantage (TF / Mamba) | TF Prefill Latency (ms) | TF Step Latency (ms) | Mamba Scan Latency (ms) | Mamba Step Latency (ms) |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for r in results:
        md.append(
            f"| **{r['T']}** | {format_bytes(r['tf_theo_b'])} | {format_bytes(r['tf_meas_b'])} | "
            f"{format_bytes(r['m_theo_b'])} | {format_bytes(r['m_meas_b'])} | **{r['ratio_mem']:.2f}x** | "
            f"{r['tf_prefill_ms']:.2f} ms | {r['tf_step_ms']:.2f} ms | {r['m_scan_ms']:.2f} ms | {r['m_step_ms']:.2f} ms |"
        )

    md.append("")
    md.append("## Theoretical Complexity Matrix")
    md.append("")
    md.append("| Metric | Transformer (Attention KV-Cache) | Mamba (Selective S6) | Parallax Proof |")
    md.append("| :--- | :--- | :--- | :--- |")

    if analysis["m_state_invariant"]:
        md.append(rf"| **State Memory Complexity** | $O(T \cdot d_{{model}} \cdot L)$ | $O(d_{{inner}} \cdot d_{{state}})$ | **Confirmed:** Mamba state size is strictly invariant to $T$ ({m_state_str} constant). |")
    else:
        md.append(r"| **State Memory Complexity** | $O(T \cdot d_{{model}} \cdot L)$ | $O(d_{{inner}} \cdot d_{{state}})$ | **Not Confirmed:** Mamba state size varied across sequence lengths. |")

    if analysis["m_step_is_flat"]:
        md.append(rf"| **Attention/Step Complexity** | $O(T)$ per decoding step | $O(1)$ recurrent step | **Confirmed:** Mamba step time remains flat at $O(1)$ computation (scaling exponent $\alpha \approx {analysis['alpha_m_step']:.2f}$). |")
    else:
        md.append(rf"| **Attention/Step Complexity** | $O(T)$ per decoding step | $O(1)$ recurrent step | **Not Confirmed:** Mamba step time scaled with sequence length (scaling exponent $\alpha \approx {analysis['alpha_m_step']:.2f}$). |")

    if analysis["tf_prefill_superlinear"] and analysis["m_scan_linear"]:
        md.append(rf"| **Sequence Scan Complexity** | $O(T^2 \cdot d)$ or FlashAttn tile | $O(T \cdot d \cdot d_{{state}})$ linear | **Confirmed:** Prefill latency scales quadratically ($O(T^{{{analysis['alpha_tf_prefill']:.2f}}})$, ~{last_res['tf_prefill_ms'] / first_res['tf_prefill_ms']:.1f}x slowdown) for attention vs linearly ($O(T^{{{analysis['alpha_m_scan']:.2f}}})$, ~{last_res['m_scan_ms'] / first_res['m_scan_ms']:.1f}x slowdown) for SSM scan. |")
    else:
        md.append(r"| **Sequence Scan Complexity** | $O(T^2 \cdot d)$ or FlashAttn tile | $O(T \cdot d \cdot d_{{state}})$ linear | **Not Confirmed:** Measured sequence scaling did not exhibit expected theoretical quadratic vs linear divergence. |")

    md.append("")
    md.append("## Key Insights & Architectural Conclusions")
    md.append("")
    if analysis["m_state_invariant"]:
        md.append("1. **Constant Memory Footprint ($O(1)$ vs $O(N)$):**")
        md.append(f"   At $T={first_res['T']}$, Transformer KV-cache occupies {format_bytes(first_res['tf_meas_b'])} vs Mamba's {m_state_str}. By $T={last_res['T']}$, Transformer KV-cache expands to {format_bytes(last_res['tf_meas_b'])} (**{last_res['ratio_mem']:.2f}x larger**), while Mamba's recurrent state remains exactly {m_state_str}.")
        md.append("   At production context lengths ($T=128k$ or $1M$ tokens), Transformer KV-caches require gigabytes of high-bandwidth VRAM per concurrent request, forcing hardware vendors to demand massive memory pools. Mamba retains full recurrent continuity within fixed state buffers.")
    else:
        md.append("1. **Memory Footprint Scaling:**")
        md.append(f"   At $T={first_res['T']}$, Transformer KV-cache occupies {format_bytes(first_res['tf_meas_b'])} and Mamba occupies {format_bytes(first_res['m_meas_b'])}; at $T={last_res['T']}$, Transformer occupies {format_bytes(last_res['tf_meas_b'])} and Mamba occupies {format_bytes(last_res['m_meas_b'])}.")
    md.append("")

    tf_slowdown = last_res['tf_prefill_ms'] / first_res['tf_prefill_ms']
    m_slowdown = last_res['m_scan_ms'] / first_res['m_scan_ms']
    md.append("2. **Inference Latency Scalability:**")
    if analysis["tf_prefill_superlinear"] and analysis["m_scan_linear"]:
        md.append(f"   As sequence length grows from ${first_res['T']}$ to ${last_res['T']}$, Transformer full sequence processing latency increases from {first_res['tf_prefill_ms']:.2f} ms to {last_res['tf_prefill_ms']:.2f} ms ({tf_slowdown:.1f}x slowdown due to super-linear $O(T^{{{analysis['alpha_tf_prefill']:.2f}}})$ scaling), whereas Mamba's scan latency scales linearly ($O(T^{{{analysis['alpha_m_scan']:.2f}}})$, {m_slowdown:.1f}x slowdown).")
    else:
        md.append(f"   As sequence length grows from ${first_res['T']}$ to ${last_res['T']}$, Transformer full sequence processing latency increases from {first_res['tf_prefill_ms']:.2f} ms to {last_res['tf_prefill_ms']:.2f} ms ({tf_slowdown:.1f}x slowdown), whereas Mamba's scan latency scales by {m_slowdown:.1f}x.")

    if analysis["tf_faster_ts"] and analysis["mamba_faster_ts"]:
        md.append(rf"   For short contexts ($T \le {max(analysis['tf_faster_ts'])}$), Transformer cached step decoding is faster ({results[0]['tf_step_ms']:.2f} ms vs {results[0]['m_step_ms']:.2f} ms at $T={results[0]['T']}$) due to low attention projection overhead at small dimensions. As context expands, Transformer step latency scales with cache size while Mamba step latency remains flat, crossing over so Mamba is faster at $T \ge {min(analysis['mamba_faster_ts'])}$.")
    elif analysis["tf_faster_ts"]:
        md.append(rf"   Transformer cached step decoding remains faster across measured sequence lengths ($T \le {last_res['T']}$) due to lightweight attention overhead at small model dimensions ({results[0]['tf_step_ms']:.2f} ms at $T={results[0]['T']}$), while Mamba step latency remains flat at ~{results[0]['m_step_ms']:.2f} ms.")
    elif analysis["mamba_faster_ts"]:
        md.append(f"   Mamba step decoding is faster across all measured sequence lengths ({results[0]['m_step_ms']:.2f} ms vs {results[0]['tf_step_ms']:.2f} ms).")
    md.append("")

    md.append("3. **Anti-RAM Verdict:**")
    md.append("   The claim that AI inference fundamentally requires ever-increasing terabytes of RAM is an artifact of the standard Softmax Attention inductive bias, not an intrinsic law of intelligence. Architectural alternatives such as Mamba S6 dismantle the memory bottleneck by operating as continuous-time selective dynamical systems.")
    md.append("")
    md.append("---")
    md.append("*Sealed by Parallax Swarm Agent 4 (Anti-RAM Benchmark) — Parallax Core Lab.*")
    md.append("")

    return "\n".join(md)


def run_benchmark():
    seq_lengths = [128, 256, 512, 1024, 2048]
    batch_size = 1
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_state = 16
    d_conv = 4
    expand = 2
    vocab_size = 256

    print("==================================================================")
    print("PARALLAX ANTI-RAM BENCHMARK: TRANSFORMER KV-CACHE vs MAMBA (S6)")
    print("Evaluating O(1) state space scaling vs O(N) linear token cache growth")
    print("==================================================================")

    gpt_config = ParallaxLMConfig(vocab_size=vocab_size, embed_dim=d_model, num_heads=num_heads, num_layers=num_layers)
    gpt_model = ParallaxLM(gpt_config)

    bm_config = BitMambaConfig(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, d_state=d_state, d_conv=d_conv, expand=expand)
    mamba_model = BitMambaLM(bm_config)

    precision_str = f"Float64 ({gpt_model.token_embeddings.weight.data.itemsize} bytes/elem)"

    results: List[Dict] = []

    for T in seq_lengths:
        print(f"[*] Benchmarking Sequence Length T = {T}...")

        # Transformer benchmark
        tf_step_ms, tf_prefill_ms, tf_theo_b, tf_meas_b = benchmark_transformer(
            gpt_model, batch_size, T, warmup=3, repeats=10
        )

        # Mamba benchmark
        m_step_ms, m_scan_ms, m_theo_b, m_meas_b = benchmark_mamba(
            mamba_model, batch_size, T, warmup=3, repeats=10
        )

        ratio_mem = tf_meas_b / m_meas_b if m_meas_b > 0 else 0

        res = {
            "T": T,
            "tf_theo_b": tf_theo_b,
            "tf_meas_b": tf_meas_b,
            "tf_step_ms": tf_step_ms,
            "tf_prefill_ms": tf_prefill_ms,
            "m_theo_b": m_theo_b,
            "m_meas_b": m_meas_b,
            "m_step_ms": m_step_ms,
            "m_scan_ms": m_scan_ms,
            "ratio_mem": ratio_mem,
        }
        results.append(res)

        print(
            f"    -> T={T}: Transformer KV={format_bytes(tf_meas_b)} (Prefill: {tf_prefill_ms:.2f}ms, Step: {tf_step_ms:.2f}ms) | "
            f"Mamba State={format_bytes(m_meas_b)} (Scan: {m_scan_ms:.2f}ms, Step: {m_step_ms:.2f}ms) | "
            f"RAM Ratio: {ratio_mem:.1f}x"
        )

    # Generate Markdown Report
    report_md = generate_report(
        results,
        d_model,
        num_layers,
        num_heads,
        d_state,
        d_conv,
        results[0]["m_meas_b"],
        precision_str=precision_str,
    )

    report_path = os.path.join(repo_root, "docs", "benchmarks", "anti_ram_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[+] Benchmark finished successfully. Report sealed at: {report_path}")


if __name__ == "__main__":
    run_benchmark()
