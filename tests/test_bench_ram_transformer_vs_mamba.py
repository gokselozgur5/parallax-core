from scripts.bench_ram_transformer_vs_mamba import generate_report


def _row(seq_len: int) -> dict:
    return {
        "T": seq_len,
        "tf_theo_b": 1024,
        "tf_meas_b": 1024,
        "tf_step_ms": 1.0,
        "tf_prefill_ms": 1.0,
        "m_theo_b": seq_len,
        "m_meas_b": seq_len,  # state grows with T, so every claim is "Not Confirmed"
        "m_step_ms": 1.0,
        "m_scan_ms": 1.0,
        "ratio_mem": 1.0,
    }


def test_not_confirmed_rows_render_single_braces():
    report = generate_report([_row(128), _row(256)], 64, 2, 4, 16, 4, 128)
    assert "Not Confirmed" in report
    assert "{{" not in report
