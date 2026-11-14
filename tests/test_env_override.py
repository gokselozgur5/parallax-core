"""Tests for PARALLAX_BACKEND environment overrides."""

from __future__ import annotations

from parallax.core.env_override import get_backend_override


def test_get_backend_override_from_env():
    env = {"PARALLAX_BACKEND": "rust"}
    assert get_backend_override(env=env) == "rust"
