from __future__ import annotations

import ctypes
from typing import Tuple
import numpy as np
from parallax.native.loader import bind

_FP = ctypes.POINTER(ctypes.c_float)
_c_mha = bind(
    "c_mha_split_qkv_f32",
    [_FP] * 4 + [ctypes.c_int] * 5,
    ctypes.c_int,
)


def is_mha_proj_available() -> bool:
    """Return True if native c_mha_split_qkv_f32 is bound."""
    return _c_mha is not None


def _check_mha_args(qkv: np.ndarray, hq: int, hkv: int, d: int) -> Tuple[int, int]:
    if _c_mha is None:
        raise RuntimeError("c_mha_split_qkv_f32 is not available")
    if qkv.ndim != 3:
        raise ValueError("qkv must have shape [B, T, (H_q + 2*H_kv)*D]")
    b, t, fused_dim = qkv.shape
    if fused_dim != (hq + 2 * hkv) * d or hq <= 0 or hkv <= 0 or d <= 0:
        raise ValueError("dimension mismatch or invalid head counts")
    return b, t


def c_mha_split_qkv(
    qkv: np.ndarray, hq: int, hkv: int, d: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split fused QKV [B, T, (H_q + 2*H_kv)*D] into Q, K, V head tensors."""
    b, t = _check_mha_args(qkv, hq, hkv, d)
    assert _c_mha is not None
    qkv_c = np.ascontiguousarray(qkv, dtype=np.float32)
    q = np.empty((b, hq, t, d), dtype=np.float32)
    k = np.empty((b, hkv, t, d), dtype=np.float32)
    v = np.empty((b, hkv, t, d), dtype=np.float32)
    rc = _c_mha(
        qkv_c.ctypes.data_as(_FP),
        q.ctypes.data_as(_FP),
        k.ctypes.data_as(_FP),
        v.ctypes.data_as(_FP),
        b,
        t,
        hq,
        hkv,
        d,
    )
    if rc != 0:
        raise ValueError("c_mha_split_qkv_f32 failed")
    return q, k, v
