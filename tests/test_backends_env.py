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
