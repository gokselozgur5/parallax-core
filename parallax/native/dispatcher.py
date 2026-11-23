"""Runtime CPU kernel dispatcher across AVX-512, AVX2, NEON, scalar C, and NumPy."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from parallax.backends.cpuinfo import CpuInfo, detect_cpu
from parallax.native.avx2_rmsnorm import is_avx2_rmsnorm_available, rmsnorm_avx2
from parallax.native.avx2_softmax import is_avx2_softmax_available, softmax_avx2
from parallax.native.gemm_avx2 import gemm_avx2, is_avx2_gemm_available
from parallax.native.gemm_avx512 import gemm_avx512, is_avx512_available
from parallax.native.gemm_neon import gemm_neon, is_gemm_neon_available
from parallax.native.loader import lazy_bind
from parallax.native.neon_rmsnorm import is_neon_rmsnorm_available, rmsnorm_neon
from parallax.native.rmsnorm_avx512 import is_avx512_rmsnorm_available, rmsnorm_avx512
from parallax.native.softmax_avx512 import is_avx512_softmax_available, softmax_avx512
from parallax.native.softmax_neon import is_softmax_neon_available, softmax_neon

_FP = ctypes.POINTER(ctypes.c_float)
_c_gemm_scalar = lazy_bind("c_gemm_scalar_f32", [_FP, _FP, _FP, ctypes.c_int, ctypes.c_int, ctypes.c_int], None)
_c_rmsnorm_scalar = lazy_bind("c_rmsnorm_scalar_f32", [_FP, _FP, _FP, ctypes.c_int, ctypes.c_int, ctypes.c_float], None)
_c_softmax_scalar = lazy_bind("c_softmax_scalar_f32", [_FP, _FP, ctypes.c_int, ctypes.c_int], None)

OPS: tuple[str, ...] = ("gemm", "rmsnorm", "softmax")
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
    raise ValueError(f"Unsupported kernel operation: {op!r}")
