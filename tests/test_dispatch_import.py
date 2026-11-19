"""Tests for parallax.dispatch bridge module imports."""

import inspect
import parallax.dispatch


def test_dispatch_import_all_symbols():
    for name in parallax.dispatch.__all__:
        obj = getattr(parallax.dispatch, name)
        assert callable(obj) or inspect.isclass(obj)
