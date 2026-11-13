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


def test_bind_missing_symbol_returns_none() -> None:
    fake_lib = MagicMock(spec=ctypes.CDLL)
    del fake_lib.missing_symbol
    with patch("parallax.native.loader.get_c_lib", return_value=fake_lib):
        assert bind("missing_symbol", [ctypes.c_int], None) is None


def test_bind_success_configures_signature() -> None:
    fake_lib = MagicMock(spec=ctypes.CDLL)
    mock_fn = MagicMock()
    fake_lib.c_kernel_test = mock_fn
    with patch("parallax.native.loader.get_c_lib", return_value=fake_lib):
        fn = bind("c_kernel_test", [ctypes.c_float], ctypes.c_int)
        assert fn is mock_fn
        assert fn.argtypes == [ctypes.c_float]
        assert fn.restype == ctypes.c_int


def test_bind_real_c_engine_symbol() -> None:
    lib = get_c_lib()
    if lib is None:
        return
    fn = bind("c_gemm_f32", [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3, None)
    assert fn is not None
    assert bind("non_existent_kernel_symbol", [], None) is None
