"""Tests for the backend registry."""

import pytest

from parallax.backends import clear, get, list_backends, register, unregister


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear()
    yield
    clear()


def test_register_and_get():
    backend = object()
    register("cpu", backend)
    assert get("cpu") is backend


def test_register_duplicate_raises():
    register("cpu", object())
    with pytest.raises(ValueError):
        register("cpu", object())
