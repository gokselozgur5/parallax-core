"""Runtime CPU kernel dispatcher across AVX-512, AVX2, NEON, scalar C, and NumPy."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from parallax.backends.cpuinfo import CpuInfo, detect_cpu
from parallax.native.avx2_rmsnorm import is_avx2_rmsnorm_available, rmsnorm_avx2
from parallax.native.avx2_softmax import is_avx2_softmax_available, softmax_avx2
from parallax.native.flashattn_avx512 import flashattn_avx512, is_avx512_flashattn_available
from parallax.native.flashattn_neon import flashattn_neon, is_flashattn_neon_available
from parallax.native.gelu import _get_gelu_fn, c_gelu, gelu_numpy
from parallax.native.gemm_avx2 import gemm_avx2, is_avx2_gemm_available
from parallax.native.gemm_avx512 import gemm_avx512, is_avx512_available
from parallax.native.gemm_neon import gemm_neon, is_gemm_neon_available
from parallax.native.layernorm import c_layernorm, is_layernorm_available
from parallax.native.loader import check_int32_bounds, lazy_bind
from parallax.native.neon_rmsnorm import is_neon_rmsnorm_available, rmsnorm_neon
from parallax.native.rmsnorm_avx512 import is_avx512_rmsnorm_available, rmsnorm_avx512
from parallax.native.rope import is_rope_available, rope_in_place
from parallax.native.softmax_avx512 import is_avx512_softmax_available, softmax_avx512
from parallax.native.softmax_neon import is_softmax_neon_available, softmax_neon
from parallax.native.swiglu import _get_swiglu_fn, c_swiglu, swiglu_numpy

_FP = ctypes.POINTER(ctypes.c_float)
_c_gemm_scalar = lazy_bind("c_gemm_scalar_f32", [_FP, _FP, _FP, ctypes.c_int, ctypes.c_int, ctypes.c_int], None)
_c_rmsnorm_scalar = lazy_bind("c_rmsnorm_scalar_f32", [_FP, _FP, _FP, ctypes.c_int, ctypes.c_int, ctypes.c_float], None)
_c_softmax_scalar = lazy_bind("c_softmax_scalar_f32", [_FP, _FP, ctypes.c_int, ctypes.c_int], None)

OPS: tuple[str, ...] = ("gemm", "rmsnorm", "softmax", "flashattn", "layernorm", "rope", "gelu", "swiglu")
_HOST_CPU: Optional[CpuInfo] = None


@dataclass(frozen=True)
class KernelSelection:
    """Selected kernel tier and callable for a native operation."""

    op: str
    isa: str
    fn: Callable[..., Any]


def get_host_cpu() -> CpuInfo:
    """Return cached CpuInfo for the current host."""
    global _HOST_CPU
    if _HOST_CPU is None:
        _HOST_CPU = detect_cpu()
    return _HOST_CPU


def reset_dispatch_cache() -> None:
    """Clear cached host CPU detection state."""
    global _HOST_CPU
    _HOST_CPU = None


def _gemm_scalar_c(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, k = a.shape
    n = b.shape[1]
    ac, bc = np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)
    out = np.empty((m, n), dtype=np.float32)
    _c_gemm_scalar(ac.ctypes.data_as(_FP), bc.ctypes.data_as(_FP), out.ctypes.data_as(_FP), m, n, k)
    return out


def _gemm_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32), dtype=np.float32)


def _rmsnorm_scalar_c(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rows, cols = x.shape
    xc, wc = np.ascontiguousarray(x, dtype=np.float32), np.ascontiguousarray(weight, dtype=np.float32)
    out = np.empty_like(xc)
    _c_rmsnorm_scalar(xc.ctypes.data_as(_FP), wc.ctypes.data_as(_FP), out.ctypes.data_as(_FP), rows, cols, float(eps))
    return out


def _rmsnorm_numpy(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    inv_rms = (1.0 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + float(eps))).astype(np.float32)
    return np.asarray((np.asarray(x, dtype=np.float32) * inv_rms) * np.asarray(weight, dtype=np.float32), dtype=np.float32)


def _softmax_scalar_c(x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    xc = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(xc)
    _c_softmax_scalar(xc.ctypes.data_as(_FP), out.ctypes.data_as(_FP), rows, cols)
    return out


def _softmax_numpy(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    exps = np.exp(x64 - np.max(x64, axis=-1, keepdims=True))
    return np.asarray(exps / np.sum(exps, axis=-1, keepdims=True), dtype=np.float32)


def _flashattn_numpy(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    scale: Optional[float] = None, is_causal: bool = False,
) -> np.ndarray:
    qc, kc, vc = np.asarray(q, dtype=np.float64), np.asarray(k, dtype=np.float64), np.asarray(v, dtype=np.float64)
    seq_len, head_dim = qc.shape
    sc = float(1.0 / np.sqrt(head_dim)) if scale is None else float(scale)
    scores = (qc @ kc.T) * sc
    if is_causal:
        scores = np.where(np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1), -np.inf, scores)
    row_max = np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores - np.where(np.isneginf(row_max), 0.0, row_max))
    denom = np.sum(probs, axis=-1, keepdims=True)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    weights = np.where(denom > 0.0, probs / safe_denom, np.where(np.isnan(denom), np.nan, 0.0))
    return np.asarray(weights @ vc, dtype=np.float32)


def _layernorm_numpy(
    x: np.ndarray, weight: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None, eps: float = 1e-5
) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    mean = np.mean(x64, axis=-1, keepdims=True)
    var = np.mean((x64 - mean) ** 2, axis=-1, keepdims=True)
    out = (np.asarray(x, dtype=np.float32) - mean.astype(np.float32)) / np.sqrt(var + float(eps)).astype(np.float32)
    if weight is not None:
        out = out * np.asarray(weight, dtype=np.float32)
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _rope_numpy(
    x: np.ndarray, start_pos: int = 0, theta: float = 10000.0, interleaved: bool = True
) -> np.ndarray:
    t, _, d = x.shape
    half = d // 2
    inv_freq = 1.0 / (float(theta) ** (np.arange(half, dtype=np.float64) / half))
    ang = (np.arange(start_pos, start_pos + t, dtype=np.float64)[:, None] * inv_freq[None, :])[:, None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x1, x2 = (x[..., 0::2], x[..., 1::2]) if interleaved else (x[..., :half], x[..., half:])
    r1, r2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    if interleaved:
        x[..., 0::2], x[..., 1::2] = r1, r2
    else:
        x[..., :half], x[..., half:] = r1, r2
    return x


def select_kernel(op: str, cpu: Optional[CpuInfo] = None) -> KernelSelection:
    """Select the highest-priority available kernel for ``op`` on ``cpu``."""
    info = get_host_cpu() if cpu is None else cpu
    if op == "gemm":
        if info.avx512f and is_avx512_available():
            return KernelSelection("gemm", "avx512", gemm_avx512)
        if info.avx2 and is_avx2_gemm_available():
            return KernelSelection("gemm", "avx2", gemm_avx2)
        if info.neon and is_gemm_neon_available():
            return KernelSelection("gemm", "neon", gemm_neon)
        if _c_gemm_scalar:
            return KernelSelection("gemm", "scalar_c", _gemm_scalar_c)
        return KernelSelection("gemm", "numpy", _gemm_numpy)
    if op == "rmsnorm":
        if info.avx512f and is_avx512_rmsnorm_available():
            return KernelSelection("rmsnorm", "avx512", rmsnorm_avx512)
        if info.avx2 and is_avx2_rmsnorm_available():
            return KernelSelection("rmsnorm", "avx2", rmsnorm_avx2)
        if info.neon and is_neon_rmsnorm_available():
            return KernelSelection("rmsnorm", "neon", rmsnorm_neon)
        if _c_rmsnorm_scalar:
            return KernelSelection("rmsnorm", "scalar_c", _rmsnorm_scalar_c)
        return KernelSelection("rmsnorm", "numpy", _rmsnorm_numpy)
    if op == "softmax":
        if info.avx512f and is_avx512_softmax_available():
            return KernelSelection("softmax", "avx512", softmax_avx512)
        if info.avx2 and is_avx2_softmax_available():
            return KernelSelection("softmax", "avx2", softmax_avx2)
        if info.neon and is_softmax_neon_available():
            return KernelSelection("softmax", "neon", softmax_neon)
        if _c_softmax_scalar:
            return KernelSelection("softmax", "scalar_c", _softmax_scalar_c)
        return KernelSelection("softmax", "numpy", _softmax_numpy)
    if op == "flashattn":
        if info.avx512f and is_avx512_flashattn_available():
            return KernelSelection("flashattn", "avx512", flashattn_avx512)
        if info.neon and is_flashattn_neon_available():
            return KernelSelection("flashattn", "neon", flashattn_neon)
        return KernelSelection("flashattn", "numpy", _flashattn_numpy)
    if op == "layernorm":
        if is_layernorm_available():
            return KernelSelection("layernorm", "scalar_c", c_layernorm)
        return KernelSelection("layernorm", "numpy", _layernorm_numpy)
    if op == "rope":
        if is_rope_available():
            return KernelSelection("rope", "scalar_c", rope_in_place)
        return KernelSelection("rope", "numpy", _rope_numpy)
    if op == "gelu":
        if _get_gelu_fn() is not None:
            return KernelSelection("gelu", "scalar_c", c_gelu)
        return KernelSelection("gelu", "numpy", gelu_numpy)
    if op == "swiglu":
        if _get_swiglu_fn() is not None:
            return KernelSelection("swiglu", "scalar_c", c_swiglu)
        return KernelSelection("swiglu", "numpy", swiglu_numpy)
    raise ValueError(f"Unsupported kernel operation: {op!r}")


def _reject_complex(*arrays: Any) -> None:
    if any(np.iscomplexobj(arr) for arr in arrays):
        raise TypeError("Complex dtypes are not supported")


def gemm(a: np.ndarray, b: np.ndarray, *, cpu: Optional[CpuInfo] = None) -> np.ndarray:
    """Dispatch float32 2D matrix multiplication to the fastest available ISA."""
    _reject_complex(a, b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible GEMM shapes: {a.shape} and {b.shape}")
    m, k = a.shape
    _, n = b.shape
    check_int32_bounds(m, n, k, m * k, k * n, m * n, label="Matrix dimensions")
    if m == 0 or n == 0 or k == 0:
        return np.zeros((m, n), dtype=np.float32)
    return np.asarray(select_kernel("gemm", cpu).fn(a, b), dtype=np.float32)


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6, *, cpu: Optional[CpuInfo] = None) -> np.ndarray:
    """Dispatch RMSNorm along the last axis to the fastest available ISA."""
    _reject_complex(x, weight)
    if x.ndim == 0 or x.shape[-1] <= 0:
        raise ValueError("x must have at least 1 dimension with positive width")
    cols = x.shape[-1]
    if weight.shape != (cols,):
        raise ValueError(f"Weight shape {weight.shape} incompatible with feature dim {cols}")
    rows = x.size // cols
    check_int32_bounds(rows, cols, rows * cols)
    if rows == 0:
        return np.empty(x.shape, dtype=np.float32)
    x_2d = np.ascontiguousarray(x, dtype=np.float32).reshape(rows, cols)
    w_1d = np.ascontiguousarray(weight, dtype=np.float32)
    return np.asarray(select_kernel("rmsnorm", cpu).fn(x_2d, w_1d, eps), dtype=np.float32).reshape(x.shape)


def softmax(x: np.ndarray, axis: int = -1, *, cpu: Optional[CpuInfo] = None) -> np.ndarray:
    """Dispatch numerically stable softmax to the fastest available ISA."""
    _reject_complex(x)
    if x.ndim == 0 or axis < -x.ndim or axis >= x.ndim:
        raise ValueError(f"Invalid softmax shape={x.shape}, axis={axis}")
    norm_axis = axis if axis >= 0 else axis + x.ndim
    cols = x.shape[norm_axis]
    if cols == 0:
        raise ValueError("Cannot compute softmax along empty dimension")
    rows = x.size // cols
    check_int32_bounds(rows, cols, rows * cols)
    if rows == 0:
        return np.empty(x.shape, dtype=np.float32)
    moved = np.moveaxis(np.asarray(x, dtype=np.float32), norm_axis, -1)
    flat = np.ascontiguousarray(moved).reshape(rows, cols)
    out = np.asarray(select_kernel("softmax", cpu).fn(flat), dtype=np.float32).reshape(moved.shape)
    return np.moveaxis(out, -1, norm_axis)


def flashattn(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    scale: Optional[float] = None, is_causal: bool = False, *, cpu: Optional[CpuInfo] = None,
) -> np.ndarray:
    """Dispatch single-head FlashAttention forward pass to the fastest available ISA."""
    _reject_complex(q, k, v)
    if q.ndim != 2 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have matching 2D shape (seq_len, head_dim)")
    seq_len, head_dim = q.shape
    if seq_len <= 0 or head_dim <= 0:
        raise ValueError("seq_len and head_dim must be positive")
    check_int32_bounds(seq_len, head_dim, seq_len * head_dim)
    return np.asarray(select_kernel("flashattn", cpu).fn(q, k, v, scale=scale, is_causal=is_causal), dtype=np.float32)


def layernorm(
    x: np.ndarray, weight: Optional[np.ndarray] = None, bias: Optional[np.ndarray] = None,
    eps: float = 1e-5, *, cpu: Optional[CpuInfo] = None,
) -> np.ndarray:
    """Dispatch LayerNorm along the last axis to native C or NumPy."""
    _reject_complex(x, weight, bias)
    if x.ndim == 0 or x.shape[-1] <= 0:
        raise ValueError("x must have at least 1 dimension with positive width")
    cols = x.shape[-1]
    rows = x.size // cols
    if weight is not None and weight.shape != (cols,):
        raise ValueError(f"weight shape mismatch: expected ({cols},), got {weight.shape}")
    if bias is not None and bias.shape != (cols,):
        raise ValueError(f"bias shape mismatch: expected ({cols},), got {bias.shape}")
    check_int32_bounds(rows, cols, rows * cols)
    if rows == 0:
        return np.empty(x.shape, dtype=np.float32)
    xc = np.ascontiguousarray(x, dtype=np.float32)
    wc = np.ascontiguousarray(weight, dtype=np.float32) if weight is not None else None
    bc = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else None
    return np.asarray(select_kernel("layernorm", cpu).fn(xc, weight=wc, bias=bc, eps=eps), dtype=np.float32)


def rope(
    x: np.ndarray, start_pos: int = 0, theta: float = 10000.0, interleaved: bool = True,
    *, in_place: bool = False, cpu: Optional[CpuInfo] = None,
) -> np.ndarray:
    """Dispatch RoPE rotation on [T, H, D] tensor to native C or NumPy."""
    _reject_complex(x)
    if x.ndim != 3 or x.shape[-1] % 2 != 0:
        raise ValueError("x must have shape [T, H, D] with even D")
    t, h, d = x.shape
    check_int32_bounds(t, h, d, t * h * d, start_pos)
    if in_place and (x.dtype != np.float32 or not x.flags.c_contiguous):
        raise ValueError("in_place requires a C-contiguous float32 array")
    buf = x if in_place else np.array(x, dtype=np.float32, copy=True, order="C")
    if t == 0 or h == 0 or d == 0:
        return buf
    return np.asarray(select_kernel("rope", cpu).fn(buf, start_pos, theta, interleaved), dtype=np.float32)


def gelu(x: np.ndarray, *, cpu: Optional[CpuInfo] = None) -> np.ndarray:
    """Dispatch fused GELU tanh approximation to native C or NumPy."""
    _reject_complex(x)
    check_int32_bounds(int(getattr(x, "size", 0)), label="Input size")
    xc = np.ascontiguousarray(x, dtype=np.float32)
    if xc.size == 0:
        return np.empty_like(xc)
    return np.asarray(select_kernel("gelu", cpu).fn(xc), dtype=np.float32)


def swiglu(a: np.ndarray, b: np.ndarray, *, cpu: Optional[CpuInfo] = None) -> np.ndarray:
    """Dispatch fused SwiGLU activation to native C or NumPy."""
    _reject_complex(a, b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    check_int32_bounds(int(a.size), label="Input size")
    ac, bc = np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)
    if ac.size == 0:
        return np.empty_like(ac)
    return np.asarray(select_kernel("swiglu", cpu).fn(ac, bc), dtype=np.float32)
