"""Tests for thread-local backend scope in parallax.core."""

from __future__ import annotations

import pytest

from parallax.core.scope import current_backend, use_backend


def test_use_backend_basic_scope():
    assert current_backend() is None
    with use_backend("numpy") as b:
        assert b == "numpy"
        assert current_backend() == "numpy"
    assert current_backend() is None



def test_nested_scopes_restore_correctly():
    with use_backend("numpy"):
        assert current_backend() == "numpy"
        with use_backend("rust"):
            assert current_backend() == "rust"
        assert current_backend() == "numpy"
    assert current_backend() is None


def test_scope_as_function_decorator():
    @use_backend("c")
    def compute():
        return current_backend()

    assert compute() == "c"
    assert current_backend() is None
