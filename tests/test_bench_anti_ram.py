"""
Tests for Parallax Anti-RAM Benchmark (Transformer vs Mamba S6).

Verifies:
- Every 'Meas.' value equals nbytes of model arrays and matches float64 dtype (#421).
- Mamba step timing evaluates state produced by scanning tokens (#421).
- Report conclusions follow dynamically from measured rows (#439).
- Contradictory synthetic rows omit 'Confirmed' claims (#439).
"""

from typing import Any, Dict, List
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.bitmamba import BitMambaLM, BitMambaConfig
from scripts.bench_ram_transformer_vs_mamba import (
    benchmark_transformer,
    benchmark_mamba,
    generate_report,
    _measure_mamba_model_state,
)


def test_transformer_meas_equals_model_array_nbytes():
    """Verify Transformer measured KV cache size equals exact nbytes of returned caches."""
    config = ParallaxLMConfig(vocab_size=128, embed_dim=32, num_heads=2, num_layers=2)
    model = ParallaxLM(config)
    batch_size = 1
    seq_len = 64

    step_ms, prefill_ms, theo_b, meas_b = benchmark_transformer(
        model, batch_size=batch_size, seq_len=seq_len, warmup=1, repeats=2
    )

    idx = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))
    initial_caches = [
        (np.zeros((batch_size, 2, 0, 16), dtype=model.token_embeddings.weight.data.dtype),
         np.zeros((batch_size, 2, 0, 16), dtype=model.token_embeddings.weight.data.dtype))
        for _ in range(config.num_layers)
    ]
    _, full_caches = model(idx, kv_caches=initial_caches)
    expected_nbytes = sum(k.nbytes + v.nbytes for k, v in full_caches)

    assert meas_b == expected_nbytes
    assert full_caches[0][0].dtype == np.float64
    assert theo_b == meas_b
    assert step_ms > 0
    assert prefill_ms > 0


def test_mamba_meas_equals_scanned_state_nbytes():
    """Verify Mamba measured state size equals exact nbytes of model recurrent states."""
    config = BitMambaConfig(vocab_size=128, d_model=32, num_layers=2, d_state=16, d_conv=4, expand=2)
    model = BitMambaLM(config)
    batch_size = 1
    seq_len = 32

    step_ms, scan_ms, theo_b, meas_b = benchmark_mamba(
        model, batch_size=batch_size, seq_len=seq_len, warmup=1, repeats=2
    )

    theo_calc, _ = _measure_mamba_model_state(model, batch_size, dtype_bytes=model.token_embeddings.weight.data.itemsize)
    assert meas_b == theo_calc
    assert theo_b == theo_calc
    assert step_ms > 0
    assert scan_ms > 0


def test_generate_report_confirmed_when_data_supports():
    """Verify 'Confirmed' proofs appear when empirical data matches theoretical scaling."""
    results: List[Dict[str, Any]] = [
        {
            "T": 128,
            "tf_theo_b": 262144,
            "tf_meas_b": 262144,
            "tf_step_ms": 0.30,
            "tf_prefill_ms": 3.0,
            "m_theo_b": 40960,
            "m_meas_b": 40960,
            "m_step_ms": 0.50,
            "m_scan_ms": 8.0,
            "ratio_mem": 6.4,
        },
        {
            "T": 256,
            "tf_theo_b": 524288,
            "tf_meas_b": 524288,
            "tf_step_ms": 0.35,
            "tf_prefill_ms": 8.0,
            "m_theo_b": 40960,
            "m_meas_b": 40960,
            "m_step_ms": 0.51,
            "m_scan_ms": 16.0,
            "ratio_mem": 12.8,
        },
        {
            "T": 512,
            "tf_theo_b": 1048576,
            "tf_meas_b": 1048576,
            "tf_step_ms": 0.40,
            "tf_prefill_ms": 25.0,
            "m_theo_b": 40960,
            "m_meas_b": 40960,
            "m_step_ms": 0.49,
            "m_scan_ms": 32.0,
            "ratio_mem": 25.6,
        },
        {
            "T": 1024,
            "tf_theo_b": 2097152,
            "tf_meas_b": 2097152,
            "tf_step_ms": 0.55,
            "tf_prefill_ms": 85.0,
            "m_theo_b": 40960,
            "m_meas_b": 40960,
            "m_step_ms": 0.50,
            "m_scan_ms": 64.0,
            "ratio_mem": 51.2,
        },
        {
            "T": 2048,
            "tf_theo_b": 4194304,
            "tf_meas_b": 4194304,
            "tf_step_ms": 0.75,
            "tf_prefill_ms": 300.0,
            "m_theo_b": 40960,
            "m_meas_b": 40960,
            "m_step_ms": 0.50,
            "m_scan_ms": 130.0,
            "ratio_mem": 102.4,
        },
    ]

    report = generate_report(
        results,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_state=16,
        d_conv=4,
        mamba_state_bytes=40960,
    )

    assert "**Confirmed:** Mamba state size is strictly invariant" in report
    assert "**Confirmed:** Mamba step time remains flat" in report
    assert "**Confirmed:** Prefill latency scales quadratically" in report


def test_generate_report_omits_confirmed_when_contradicted():
    """Verify 'Confirmed' proofs are NOT printed when synthetic data contradicts them."""
    contradictory_results: List[Dict[str, Any]] = [
        {
            "T": 128,
            "tf_theo_b": 262144,
            "tf_meas_b": 262144,
            "tf_step_ms": 1.0,
            "tf_prefill_ms": 100.0,
            "m_theo_b": 40960,
            "m_meas_b": 10000,
            "m_step_ms": 1.0,
            "m_scan_ms": 50.0,
            "ratio_mem": 26.2,
        },
        {
            "T": 256,
            "tf_theo_b": 524288,
            "tf_meas_b": 524288,
            "tf_step_ms": 1.0,
            "tf_prefill_ms": 90.0,
            "m_theo_b": 40960,
            "m_meas_b": 20000,
            "m_step_ms": 4.0,
            "m_scan_ms": 50.0,
            "ratio_mem": 26.2,
        },
        {
            "T": 512,
            "tf_theo_b": 1048576,
            "tf_meas_b": 1048576,
            "tf_step_ms": 1.0,
            "tf_prefill_ms": 80.0,
            "m_theo_b": 40960,
            "m_meas_b": 40000,
            "m_step_ms": 16.0,
            "m_scan_ms": 50.0,
            "ratio_mem": 26.2,
        },
        {
            "T": 1024,
            "tf_theo_b": 2097152,
            "tf_meas_b": 2097152,
            "tf_step_ms": 1.0,
            "tf_prefill_ms": 70.0,
            "m_theo_b": 40960,
            "m_meas_b": 80000,
            "m_step_ms": 64.0,
            "m_scan_ms": 50.0,
            "ratio_mem": 26.2,
        },
        {
            "T": 2048,
            "tf_theo_b": 4194304,
            "tf_meas_b": 4194304,
            "tf_step_ms": 1.0,
            "tf_prefill_ms": 60.0,
            "m_theo_b": 40960,
            "m_meas_b": 160000,
            "m_step_ms": 256.0,
            "m_scan_ms": 50.0,
            "ratio_mem": 26.2,
        },
    ]

    report = generate_report(
        contradictory_results,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_state=16,
        d_conv=4,
        mamba_state_bytes=40960,
    )

    assert "**Confirmed:** Mamba state size is strictly invariant" not in report
    assert "**Confirmed:** Mamba step time remains flat" not in report
    assert "**Confirmed:** Prefill latency scales quadratically" not in report
    assert "Not Confirmed" in report
