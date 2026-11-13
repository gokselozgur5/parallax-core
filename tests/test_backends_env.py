import pytest

from parallax.backends.env import backend_override


def test_no_variables_means_no_override():
    assert backend_override("gemm", {}) is None


def test_global_override_applies_to_every_op():
    env = {"PARALLAX_BACKEND": "rust"}
    assert backend_override("gemm", env) == "rust"
    assert backend_override("softmax", env) == "rust"


def test_per_op_override_wins_over_global():
    env = {"PARALLAX_BACKEND": "rust", "PARALLAX_BACKEND_GEMM": "metal"}
    assert backend_override("gemm", env) == "metal"
    assert backend_override("softmax", env) == "rust"


def test_values_are_trimmed_and_case_insensitive():
    assert backend_override("gemm", {"PARALLAX_BACKEND": "  NumPy \n"}) == "numpy"


def test_empty_per_op_value_falls_back_to_global():
    env = {"PARALLAX_BACKEND": "c-avx", "PARALLAX_BACKEND_GEMM": "  "}
    assert backend_override("gemm", env) == "c-avx"


def test_unknown_backend_raises_with_valid_names():
    with pytest.raises(ValueError, match="PARALLAX_BACKEND_GEMM='cuda'.*numpy, rust"):
        backend_override("gemm", {"PARALLAX_BACKEND_GEMM": "cuda"})


def test_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("PARALLAX_BACKEND", "c")
    assert backend_override("rmsnorm") == "c"
