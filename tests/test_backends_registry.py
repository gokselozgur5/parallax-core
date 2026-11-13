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


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get("missing")


def test_unregister_removes_and_unknown_raises():
    register("cpu", object())
    unregister("cpu")
    assert list_backends() == []
    with pytest.raises(KeyError):
        unregister("cpu")


def test_list_backends_priority_then_registration_order():
    register("a", object(), priority=1)
    register("b", object(), priority=5)
    register("c", object(), priority=1)
    register("d", object())
    assert list_backends() == ["b", "a", "c", "d"]


def test_clear_empties_registry():
    register("cpu", object())
    clear()
    assert list_backends() == []
