from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import bind

_FP = ctypes.POINTER(ctypes.c_float)
_c_gqa = bind("c_gqa_proj_f32", [_FP] * 8 + [ctypes.c_int] * 5, ctypes.c_int)


def is_gqa_proj_available() -> bool:
    """Return True if native c_gqa_proj_f32 is bound."""
    return _c_gqa is not None


def c_gqa_proj(x, wq, wk, wv, n_heads: int, n_kv: int, head_dim: int):
    """Project x [rows, dim] to Q and KV-replicated K, V, each [rows, n_heads*head_dim]."""
    if _c_gqa is None:
        raise RuntimeError("c_gqa_proj_f32 is not available")
    arrs = [np.ascontiguousarray(a, dtype=np.float32) for a in (x, wq, wk, wv)]
    rows, dim = arrs[0].shape
    out_cols = n_heads * head_dim
    q, k, v = (np.empty((rows, out_cols), dtype=np.float32) for _ in range(3))
    tmp = np.empty((rows, n_kv * head_dim), dtype=np.float32)
    ptrs = [a.ctypes.data_as(_FP) for a in (*arrs, q, k, v, tmp)]
    rc = _c_gqa(*ptrs, rows, dim, n_heads, n_kv, head_dim)
    if rc != 0:
        raise ValueError("invalid GQA projection arguments")
    return q, k, v
