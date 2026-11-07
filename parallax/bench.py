"""
Performance Benchmarking and FLOPs Profiling Suite.

Provides precise runtime instrumentation, parameter counting,
and theoretical FLOPs calculations for ParallaxLM forward/backward passes
and autoregressive inference (with and without KV cache).
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.generator import generate


@dataclass
class BenchmarkResult:
    """Stores benchmark metrics for a specific workload."""
    workload: str
    tokens_per_second: float
    latency_ms: float
    total_tokens: int
    memory_mb_approx: float


def count_parameters(model: ParallaxLM) -> Dict[str, int]:
    """Counts total learnable parameters grouped by structural submodules."""
    breakdown: Dict[str, int] = {
        "embedding": sum(int(np.prod(p.shape)) for p in model.token_embeddings.parameters()),
        "attention": 0,
        "mlp": 0,
        "norm": sum(int(np.prod(p.shape)) for p in model.final_norm.parameters()),
        "head": sum(int(np.prod(p.shape)) for p in model.lm_head.parameters()),
        "total": 0,
    }

    for block in model.blocks:
        breakdown["attention"] += sum(int(np.prod(p.shape)) for p in block.attn.parameters())
        breakdown["mlp"] += sum(int(np.prod(p.shape)) for p in block.mlp.parameters())
        breakdown["norm"] += sum(int(np.prod(p.shape)) for p in block.attn_norm.parameters())
        breakdown["norm"] += sum(int(np.prod(p.shape)) for p in block.mlp_norm.parameters())

    breakdown["total"] = sum(int(np.prod(p.shape)) for p in model.parameters())
    return breakdown


def estimate_flops_per_token(config: ParallaxLMConfig) -> int:
    """Estimates theoretical FLOPs required per token in forward pass.

    Standard Transformer FLOP approximation:
    - Token embedding lookup: ~0 (memory bound)
    - Attention Q, K, V + Output projections: 4 * 2 * d_model^2 * n_layers
    - MLP layers (fc1 + fc2): 2 * 2 * (4 * d_model) * d_model * n_layers = 16 * d_model^2 * n_layers
    - LM Head projection: 2 * d_model * vocab_size
    Total FLOPs/token ~= 24 * n_layers * d_model^2 + 2 * d_model * vocab_size
    """
    d = config.embed_dim
    L = config.num_layers
    V = config.vocab_size

    attn_flops = 4 * 2 * (d**2) * L
    mlp_flops = 16 * (d**2) * L
    head_flops = 2 * d * V

    return attn_flops + mlp_flops + head_flops


def benchmark_inference(
    model: ParallaxLM,
    prompt_len: int = 8,
    gen_len: int = 16,
    num_runs: int = 3,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Compares inference throughput with and without KV cache."""
    prompt = list(range(1, prompt_len + 1))

    # 1. Benchmark uncached inference
    latencies_uncached: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = generate(model, prompt, max_new_tokens=gen_len, temperature=0.0, use_cache=False)
        latencies_uncached.append(time.perf_counter() - t0)

    mean_uncached_s = float(np.mean(latencies_uncached))
    tok_sec_uncached = gen_len / mean_uncached_s if mean_uncached_s > 0 else 0.0

    uncached_result = BenchmarkResult(
        workload="Autoregressive Inference (No Cache - O(N^2))",
        tokens_per_second=tok_sec_uncached,
        latency_ms=mean_uncached_s * 1000.0,
        total_tokens=gen_len,
        memory_mb_approx=0.0,
    )

    # 2. Benchmark cached inference
    latencies_cached: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = generate(model, prompt, max_new_tokens=gen_len, temperature=0.0, use_cache=True)
        latencies_cached.append(time.perf_counter() - t0)

    mean_cached_s = float(np.mean(latencies_cached))
    tok_sec_cached = gen_len / mean_cached_s if mean_cached_s > 0 else 0.0

    cached_result = BenchmarkResult(
        workload="Autoregressive Inference (KV Cache - O(1) step)",
        tokens_per_second=tok_sec_cached,
        latency_ms=mean_cached_s * 1000.0,
        total_tokens=gen_len,
        memory_mb_approx=0.0,
    )

    return uncached_result, cached_result


def format_benchmark_table(results: List[BenchmarkResult]) -> str:
    """Formats benchmark results into a clean markdown table."""
    lines = [
        "| Workload | Latency (ms) | Tokens / Sec | Speedup |",
        "| :--- | :---: | :---: | :---: |",
    ]

    base_speed = results[0].tokens_per_second if results else 1.0
    for r in results:
        speedup = f"{r.tokens_per_second / base_speed:.2f}x" if base_speed > 0 else "1.00x"
        lines.append(f"| {r.workload} | {r.latency_ms:.2f} ms | {r.tokens_per_second:.1f} tok/s | {speedup} |")

    return "\n".join(lines)
