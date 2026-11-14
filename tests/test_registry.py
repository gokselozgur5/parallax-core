"""Tests for core backend registry."""

from parallax.core.registry import BackendRegistry


def test_register_and_get():
    reg = BackendRegistry()
    engine = object()
    reg.register("cpu", engine)
    assert reg.get("cpu") is engine
