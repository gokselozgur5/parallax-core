"""Python wrapper for the NEON 8x8 SGEMM microkernel."""

from __future__ import annotations

import ctypes

from parallax.native.loader import bind

_c_gemm_neon_f32 = bind(
    "c_gemm_neon_f32",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int],
    ctypes.c_int,
)


def is_gemm_neon_available() -> bool:
    """Return True if the NEON SGEMM symbol is bound."""
    return _c_gemm_neon_f32 is not None
