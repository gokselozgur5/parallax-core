from __future__ import annotations

import ctypes
from unittest.mock import MagicMock, patch

from parallax.native import bind, get_c_lib


def test_get_c_lib_reuses_c_engine() -> None:
    fake_lib = MagicMock(spec=ctypes.CDLL)
    with patch("parallax.native.loader._find_or_build_library", return_value=fake_lib):
        assert get_c_lib() is fake_lib


def test_bind_missing_when_lib_none() -> None:
    with patch("parallax.native.loader.get_c_lib", return_value=None):
        assert bind("some_symbol", [ctypes.c_int], ctypes.c_int) is None
