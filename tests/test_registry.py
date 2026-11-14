"""Tests for core backend registry."""

import pytest

from parallax.core.registry import BackendRegistry


def test_register_and_get():
    reg = BackendRegistry()
    engine = object()
    reg.register("cpu", engine)
    assert reg.get("cpu") is engine


def test_register_duplicate_raises():
    reg = BackendRegistry()
    reg.register("cpu", object())
    with pytest.raises(ValueError):
        reg.register("cpu", object())


def test_get_unknown_raises():
    reg = BackendRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")
