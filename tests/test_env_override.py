"""Tests for PARALLAX_BACKEND environment overrides."""

from __future__ import annotations

import pytest

from parallax.core.env_override import get_backend_override


def test_get_backend_override_from_env():
    env = {"PARALLAX_BACKEND": "rust"}
    assert get_backend_override(env=env) == "rust"


def test_per_op_override_takes_precedence():
    env = {"PARALLAX_BACKEND": "rust", "PARALLAX_BACKEND_GEMM": "metal"}
    assert get_backend_override(op="gemm", env=env) == "metal"
    assert get_backend_override(op="softmax", env=env) == "rust"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="is not a known backend"):
        get_backend_override(env={"PARALLAX_BACKEND": "unknown_backend"})


def test_empty_or_whitespace_override():
    assert get_backend_override(env={"PARALLAX_BACKEND": "   "}) is None
    assert get_backend_override(env={"PARALLAX_BACKEND": "  CuDa  "}) == "cuda"


def test_backend_override_alias_and_is_override_active():
    from parallax.core.env_override import backend_override, is_override_active

    env = {"PARALLAX_BACKEND": "numpy"}
    assert backend_override("gemm", env) == "numpy"
    assert is_override_active("gemm", env) is True
    assert is_override_active("gemm", {}) is False


def test_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("PARALLAX_BACKEND", "c")
    assert get_backend_override("rmsnorm") == "c"
