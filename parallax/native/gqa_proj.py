from __future__ import annotations

import ctypes

import numpy as np

from parallax.native.loader import check_int32_bounds, lazy_bind

_FP = ctypes.POINTER(ctypes.c_float)
_c_gqa = lazy_bind("c_gqa_proj_f32", [_FP] * 8 + [ctypes.c_int] * 5, ctypes.c_int)


def is_gqa_proj_available() -> bool:
    """Return True if native c_gqa_proj_f32 is bound."""
    return bool(_c_gqa)


def c_gqa_proj(x, wq, wk, wv, n_heads: int, n_kv: int, head_dim: int):
    """Project x [rows, dim] to Q and KV-replicated K, V, each [rows, n_heads*head_dim]."""
    if not _c_gqa:
        raise RuntimeError("c_gqa_proj_f32 is not available")
    if getattr(x, "ndim", None) == 2:
        r0, d0 = x.shape
        oc0 = max(n_heads, n_kv) * head_dim
        check_int32_bounds(r0, d0, n_heads, n_kv, head_dim, oc0, r0 * d0, d0 * oc0, r0 * oc0)
    arrs = [np.ascontiguousarray(a, dtype=np.float32) for a in (x, wq, wk, wv)]
    if arrs[0].ndim != 2:
        raise ValueError("x must be a 2D tensor [rows, dim]")
    rows, dim = arrs[0].shape
    if arrs[1].shape != (dim, n_heads * head_dim):
        raise ValueError(f"wq shape {arrs[1].shape} does not match (dim, n_heads * head_dim)")
    if arrs[2].shape != (dim, n_kv * head_dim):
        raise ValueError(f"wk shape {arrs[2].shape} does not match (dim, n_kv * head_dim)")
    if arrs[3].shape != (dim, n_kv * head_dim):
        raise ValueError(f"wv shape {arrs[3].shape} does not match (dim, n_kv * head_dim)")
    out_cols = n_heads * head_dim
    q, k, v = (np.empty((rows, out_cols), dtype=np.float32) for _ in range(3))
    tmp = np.empty((rows, n_kv * head_dim), dtype=np.float32)
    ptrs = [a.ctypes.data_as(_FP) for a in (*arrs, q, k, v, tmp)]
    rc = _c_gqa(*ptrs, rows, dim, n_heads, n_kv, head_dim)
    if rc != 0:
        raise ValueError("invalid GQA projection arguments")
    return q, k, v
