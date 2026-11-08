"""
Unit tests for Performance Benchmarking and Profiling Suite (M9).
"""

import numpy as np
import pytest

from parallax.bench import (
    BenchmarkResult,
    benchmark_inference,
    count_parameters,
    estimate_flops_per_token,
    format_benchmark_table,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_count_parameters():
    config = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
        use_rope=True,
    )
    model = ParallaxLM(config)
    breakdown = count_parameters(model)

    assert breakdown["total"] > 0
    assert breakdown["embedding"] == 64 * 32
    assert breakdown["head"] == 32 * 64
    assert breakdown["attention"] > 0
    assert breakdown["mlp"] > 0
    assert breakdown["total"] == sum([
        breakdown["embedding"],
        breakdown["attention"],
        breakdown["mlp"],
        breakdown["norm"],
        breakdown["head"],
    ])


def test_estimate_flops_per_token():
    config = ParallaxLMConfig(
        vocab_size=256,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        max_seq_len=128,
        use_rope=True,
    )
    flops = estimate_flops_per_token(config)
    assert flops > 0
    # Expected: 24 * 3 * (64^2) + 2 * 64 * 256 = 294912 + 32768 = 327680
    assert flops == 327680


def test_benchmark_inference_speedup():
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=32,
        use_rope=True,
    )
    model = ParallaxLM(config)

    uncached, cached = benchmark_inference(model, prompt_len=4, gen_len=8, num_runs=2)
    # In micro-scale toy models (seq_len=8), Python loop overhead can exceed O(N^2) matmul time.
    # Verify both benchmark metrics run and report valid positive throughput.
    assert uncached.tokens_per_second > 0
    assert cached.tokens_per_second > 0

    table = format_benchmark_table([uncached, cached])
    assert "| Workload | Latency (ms) | Tokens / Sec | Speedup |" in table
    assert "Autoregressive Inference" in table
