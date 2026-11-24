"""Tests for scripts.demo_showcase."""

from __future__ import annotations

import json

from scripts import demo_showcase


def test_demo_showcase_main_quick_json(capsys) -> None:
    """Verify demo_showcase main with --quick and --json produces valid JSON metrics."""
    code = demo_showcase.main(["--quick", "--json"])
    assert code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["version"] == "0.2.0"
    assert data["autograd"]["status"] == "PASSED"
    assert data["autograd"]["metrics"]["max_grad_error"] < 1e-4

    assert len(data["memory_scaling"]) == 3
    for row in data["memory_scaling"]:
        assert row["transformer_kv_bytes"] > 0
        assert row["mamba_state_bytes"] > 0
        assert row["ratio_transformer_over_mamba"] > 0

    assert "active_gemm_backend" in data["dispatch"]
    assert "active_rmsnorm_backend" in data["dispatch"]
    from parallax.backends.dispatch import get_last_backend
    assert data["dispatch"]["active_rmsnorm_backend"] == (get_last_backend("rmsnorm") or "numpy")
    assert data["dispatch"]["active_gemm_backend"] == (get_last_backend("gemm") or "numpy")
    assert data["dispatch"]["gemm_latency_ms"] >= 0
    assert data["dispatch"]["rmsnorm_latency_ms"] >= 0


def test_demo_showcase_human_output(capsys) -> None:
    """Verify demo_showcase human-readable stdout execution."""
    code = demo_showcase.main(["--quick"])
    assert code == 0

    captured = capsys.readouterr()
    assert "PARALLAX v0.2.0 SHOWCASE DEMO" in captured.out
    assert "Autograd" in captured.out
    assert "Memory" in captured.out
    assert "Dispatch" in captured.out


def test_experimental_modules_lazy_resolution() -> None:
    """Verify every module declared in EXPERIMENTAL_MODULES resolves cleanly."""
    import parallax.experimental

    for mod_name in parallax.experimental.EXPERIMENTAL_MODULES:
        mod = getattr(parallax.experimental, mod_name)
        assert mod is not None
