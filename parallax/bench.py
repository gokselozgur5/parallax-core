"""
Performance Benchmarking and FLOPs Profiling Suite.

Provides precise runtime instrumentation, parameter counting,
and theoretical FLOPs calculations for ParallaxLM forward/backward passes
and autoregressive inference (with and without KV cache).
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple
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

    unique_params = {id(p): p for p in model.parameters()}
    breakdown["total"] = sum(int(np.prod(p.shape)) for p in unique_params.values())
    return breakdown


def estimate_flops_per_token(config: ParallaxLMConfig) -> int:
    """Estimates theoretical FLOPs required per token in forward pass.

    Transformer FLOP accounting:
    - Attention linear projections: Q (2*d^2), K & V (2 * 2 * d * d_kv), Out (2*d^2)
    - MLP projections:
        Dense (SwiGLU: 6*d*intermediate, Standard: 4*d*intermediate)
        MoE: Gate (2*d*num_experts) + top_k * expert_mlp_flops
    - LM Head projection: 2 * d * vocab_size
    """
    d = config.embed_dim
    L = config.num_layers
    V = config.vocab_size

    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else num_heads
    d_kv = (num_kv_heads * d) // num_heads

    q_flops = 2 * d * d
    k_flops = 2 * d * d_kv
    v_flops = 2 * d * d_kv
    out_flops = 2 * d * d
    attn_flops_per_layer = q_flops + k_flops + v_flops + out_flops

    if config.intermediate_size is not None:
        intermediate_size = config.intermediate_size
    elif config.use_swiglu:
        intermediate_size = 64 * ((int(2 * (4 * d) / 3) + 63) // 64)
    else:
        intermediate_size = 4 * d

    if config.use_swiglu:
        single_mlp_flops = 6 * d * intermediate_size
    else:
        single_mlp_flops = 4 * d * intermediate_size

    if config.num_experts is not None and config.num_experts > 1:
        top_k = config.top_k_experts if config.top_k_experts is not None else 2
        gate_flops = 2 * d * config.num_experts
        mlp_flops_per_layer = gate_flops + top_k * single_mlp_flops
    else:
        mlp_flops_per_layer = single_mlp_flops

    head_flops = 2 * d * V

    return (attn_flops_per_layer + mlp_flops_per_layer) * L + head_flops


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
    actual_uncached_tokens = 0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = generate(model, prompt, max_new_tokens=gen_len, temperature=0.0, use_cache=False)
        latencies_uncached.append(time.perf_counter() - t0)
        actual_uncached_tokens = len(out) - len(prompt)

    mean_uncached_s = float(np.mean(latencies_uncached))
    tok_sec_uncached = actual_uncached_tokens / mean_uncached_s if mean_uncached_s > 0 else 0.0

    uncached_result = BenchmarkResult(
        workload="Autoregressive Inference (No Cache - O(N^2))",
        tokens_per_second=tok_sec_uncached,
        latency_ms=mean_uncached_s * 1000.0,
        total_tokens=actual_uncached_tokens,
        memory_mb_approx=0.0,
    )

    # 2. Benchmark cached inference
    latencies_cached: List[float] = []
    actual_cached_tokens = 0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = generate(model, prompt, max_new_tokens=gen_len, temperature=0.0, use_cache=True)
        latencies_cached.append(time.perf_counter() - t0)
        actual_cached_tokens = len(out) - len(prompt)

    mean_cached_s = float(np.mean(latencies_cached))
    tok_sec_cached = actual_cached_tokens / mean_cached_s if mean_cached_s > 0 else 0.0

    cached_result = BenchmarkResult(
        workload="Autoregressive Inference (KV Cache - O(1) step)",
        tokens_per_second=tok_sec_cached,
        latency_ms=mean_cached_s * 1000.0,
        total_tokens=actual_cached_tokens,
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
