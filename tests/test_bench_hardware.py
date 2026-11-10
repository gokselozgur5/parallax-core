"""
Tests for Parallax Hardware Backends & Architectural Regression Suite.

Verifies:
- Hardware GEMM & RMSNorm benchmarks across NumPy, C-Engine, Rust, and Metal GPU.
- Architectural memory footprint scaling (O(T) vs O(1)).
- INT4 layer compression ratio verification.
- FlashDecoding v1 vs v2 Split-K execution and numerical parity.
- Markdown report generation and disk serialization.
"""

import os
import tempfile
import numpy as np
import pytest

from scripts.bench_all_hardware_backends import (
    bench_gemm,
    bench_rmsnorm,
    bench_memory_scaling,
    calculate_architecture_weights,
    bench_int4_layer,
    bench_flash_decoding,
    get_system_hardware_info,
    generate_markdown_report,
    save_markdown_report,
    format_bytes,
)
from parallax.c_engine import is_c_engine_available
from parallax.metal_engine import is_metal_available


def test_system_hardware_info():
    """Verify hardware info detection contains valid system keys."""
    info = get_system_hardware_info()
    assert "os" in info
    assert "arch" in info
    assert "processor" in info
    assert "python_version" in info
    assert "c_engine_available" in info
    assert "metal_device" in info


def test_bench_gemm_execution():
    """Verify GEMM benchmark executes accurately across all active backends."""
    sizes = [64, 128]
    records = bench_gemm(sizes=sizes, repeats=2, warmup=1)
    assert len(records) == len(sizes)

    for rec in records:
        assert "numpy" in rec
        assert rec["numpy"]["ms"] > 0
        assert rec["numpy"]["gflops"] > 0

        if is_c_engine_available():
            assert "c_engine" in rec
            assert rec["c_engine"]["ms"] > 0
            assert rec["c_engine"]["diff"] < 1e-4

        if is_metal_available():
            assert "metal" in rec
            assert rec["metal"]["ms"] > 0
            assert rec["metal"]["diff"] < 1e-4

        if "rust" in rec:
            assert rec["rust"]["ms"] > 0
            assert rec["rust"]["diff"] < 1e-4


def test_bench_rmsnorm_execution():
    """Verify RMSNorm benchmark produces matching numerical outputs."""
    shapes = [(64, 128), (128, 256)]
    records = bench_rmsnorm(shapes=shapes, repeats=2, warmup=1)
    assert len(records) == len(shapes)

    for rec in records:
        assert "numpy" in rec
        assert rec["numpy"]["ms"] > 0
        assert rec["numpy"]["gflops"] > 0

        if is_c_engine_available():
            assert "c_engine" in rec
            assert rec["c_engine"]["ms"] > 0
            assert rec["c_engine"]["diff"] < 1e-4

        if is_metal_available():
            assert "metal" in rec
            assert rec["metal"]["ms"] > 0
            assert rec["metal"]["diff"] < 1e-4


def test_memory_scaling_complexity():
    """Verify Transformer KV-cache scales linearly O(T) while Mamba is O(1) constant."""
    seq_lengths = [128, 256, 512, 1024]
    records = bench_memory_scaling(seq_lengths=seq_lengths, batch_size=1, d_model=64, num_layers=2)

    assert len(records) == len(seq_lengths)

    mamba_bytes_0 = records[0]["mamba_state_bytes"]
    bitmamba_bytes_0 = records[0]["bitmamba_state_bytes"]

    prev_tf_kv = 0
    for r in records:
        # KV cache strictly grows linearly with sequence length
        assert r["tf_kv_bytes"] > prev_tf_kv
        prev_tf_kv = r["tf_kv_bytes"]

        # Mamba and BitMamba state must remain strictly constant invariant of T
        assert r["mamba_state_bytes"] == mamba_bytes_0
        assert r["bitmamba_state_bytes"] == bitmamba_bytes_0

    # Multiplier at T=1024 must be 8x higher than at T=128
    assert records[-1]["ratio_kv_to_mamba"] == pytest.approx(records[0]["ratio_kv_to_mamba"] * 8.0, rel=1e-3)


def test_architecture_weights_and_int4():
    """Verify parameter calculations and INT4 compression factors."""
    weights = calculate_architecture_weights(vocab_size=256, d_model=64, num_layers=2)
    assert "transformer_fp32" in weights
    assert "transformer_int4" in weights
    assert "mamba_fp32" in weights
    assert "bitmamba" in weights

    assert weights["transformer_int4"]["bytes"] < weights["transformer_fp32"]["bytes"]
    assert weights["bitmamba"]["bytes"] < weights["mamba_fp32"]["bytes"]

    # INT4 Layer benchmark
    int4_stats = bench_int4_layer(in_features=256, out_features=256, group_size=32)
    assert int4_stats["weight_compression_ratio"] == 8.0
    assert int4_stats["total_compression_ratio"] > 4.5


def test_flash_decoding_v1_vs_v2():
    """Verify FlashDecoding v2 numerical parity with v1 and dynamic Split-K logic."""
    seq_lengths = [128, 256, 512]
    records = bench_flash_decoding(seq_lengths=seq_lengths, d=32, repeats=2, warmup=1)

    assert len(records) == len(seq_lengths)
    for r in records:
        assert r["v1_ms"] > 0
        assert r["v2_ms"] > 0
        assert r["diff"] < 1e-10  # Numerical parity
        assert r["v2_splits"] >= 1
        assert r["v1_splits"] >= 1


def test_markdown_report_generation_and_saving():
    """Verify markdown report generates valid headers, tables, and writes to disk."""
    hw_info = get_system_hardware_info()
    gemm_results = bench_gemm(sizes=[64], repeats=1, warmup=1)
    rmsnorm_results = bench_rmsnorm(shapes=[(64, 64)], repeats=1, warmup=1)
    weights_summary = calculate_architecture_weights(vocab_size=256, d_model=64, num_layers=2)
    int4_stats = bench_int4_layer(128, 128, group_size=32)
    scaling_results = bench_memory_scaling([128, 256], batch_size=1, d_model=64, num_layers=2)
    flash_results = bench_flash_decoding([128], d=32, repeats=1, warmup=1)

    report_str = generate_markdown_report(
        hw_info=hw_info,
        gemm_results=gemm_results,
        rmsnorm_results=rmsnorm_results,
        weights_summary=weights_summary,
        scaling_results=scaling_results,
        int4_stats=int4_stats,
        flash_results=flash_results,
    )

    assert "# Parallax Hardware Matrix & Architectural Regression Benchmark Report" in report_str
    assert "## 1. Hardware GEMM Compute Benchmark" in report_str
    assert "## 2. Hardware RMSNorm Compute Benchmark" in report_str
    assert "## 3. Architecture Weight Memory & INT4 Quantization" in report_str
    assert "## 4. Context State RAM Scaling" in report_str
    assert "## 5. FlashDecoding v1 vs FlashDecoding v2" in report_str
    assert "Sealed by Parallax Swarm Agent 18" in report_str
    assert f"- **Operating System:** `{hw_info['os']}`" in report_str
    assert f"**{gemm_results[0]['size'][0]} x {gemm_results[0]['size'][1]}**" in report_str
    assert f"**{rmsnorm_results[0]['shape'][0]} x {rmsnorm_results[0]['shape'][1]}**" in report_str
    assert f"{weights_summary['transformer_fp32']['params']:,}" in report_str
    assert f"**{int4_stats['total_compression_ratio']:.2f}x**" in report_str

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "report.md")
        save_markdown_report(report_str, tmp_file)
        assert os.path.isfile(tmp_file)
        with open(tmp_file, "r", encoding="utf-8") as f:
            read_back = f.read()
        assert read_back == report_str
