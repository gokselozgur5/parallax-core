"""AVX-512 FlashAttention-1 single-head microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np
from parallax.native.loader import bind

_FN_FLASHATTN_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512_FLASHATTN: Optional[Callable[..., Any]] = None


def _get_flashattn_avx512_fn() -> Optional[Callable[..., Any]]:
    global _FN_FLASHATTN_AVX512
    if _FN_FLASHATTN_AVX512 is None:
        args = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
        ]
        _FN_FLASHATTN_AVX512 = bind("c_flashattn_avx512_f32", args, ctypes.c_int)
    return _FN_FLASHATTN_AVX512


def is_avx512_flashattn_available() -> bool:
    """Return True if AVX-512 FlashAttention is supported by CPU."""
    global _FN_HAS_AVX512_FLASHATTN
    if _FN_HAS_AVX512_FLASHATTN is None:
        _FN_HAS_AVX512_FLASHATTN = bind(
            "c_has_avx512_flashattn_f32", [], ctypes.c_int
        )
    if _FN_HAS_AVX512_FLASHATTN is None:
        return False
    return bool(_FN_HAS_AVX512_FLASHATTN() != 0)


def _validate_inputs(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, and v must be 2-dimensional tensors")
    seq_len, head_dim = q.shape
    if k.shape != (seq_len, head_dim) or v.shape != (seq_len, head_dim):
        raise ValueError("q, k, and v shapes must match (seq_len, head_dim)")
    return seq_len, head_dim


def _prep_arrays(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    return (
        np.ascontiguousarray(q, dtype=np.float32),
        np.ascontiguousarray(k, dtype=np.float32),
        np.ascontiguousarray(v, dtype=np.float32),
    )


def _invoke_c_kernel(fn, q, k, v, out, seq_len, head_dim, scale, is_causal):
    c_flag = 1 if is_causal else 0
    status = fn(
        q.ctypes.data, k.ctypes.data, v.ctypes.data, out.ctypes.data,
        seq_len, head_dim, ctypes.c_float(scale), c_flag
    )
    if status == -5:
        raise RuntimeError("AVX-512F is not supported on this CPU")
    if status != 0:
        raise RuntimeError(f"c_flashattn_avx512_f32 failed: {status}")


def flashattn_avx512(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: Optional[float] = None,
    is_causal: bool = False,
) -> np.ndarray:
    """Compute single-head FlashAttention-1 forward with online softmax."""
    seq_len, head_dim = _validate_inputs(q, k, v)
    fn = _get_flashattn_avx512_fn()
    if fn is None:
        raise RuntimeError("c_flashattn_avx512_f32 symbol not available")
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim))
    qc, kc, vc = _prep_arrays(q, k, v)
    out = np.empty((seq_len, head_dim), dtype=np.float32)
    _invoke_c_kernel(fn, qc, kc, vc, out, seq_len, head_dim, scale, is_causal)
    return out
