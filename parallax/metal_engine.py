"""
Parallax Metal & CUDA JIT Engine.

Lightweight FFI bridge for compiling and running GPU kernels at runtime:
- Apple Silicon (Metal Performance / MSL JIT) via ctypes Objective-C runtime
- CUDA JIT compilation fallback / abstraction hooks
- Graceful fallback to CPU/NumPy if Metal or CUDA is not available.
"""

from __future__ import annotations

import sys
import threading
import ctypes
from ctypes import util
from typing import Optional, ContextManager
from contextlib import contextmanager
import numpy as np

# Global cached Metal state
_METAL_INITIALIZED: bool = False
_IS_METAL_AVAILABLE: bool = False
_INT32_MAX: int = 2147483647

_OBJC: Optional[ctypes.CDLL] = None
_METAL: Optional[ctypes.CDLL] = None
_CF: Optional[ctypes.CDLL] = None
_DEVICE: Optional[ctypes.c_void_p] = None
_COMMAND_QUEUE: Optional[ctypes.c_void_p] = None

_PSO_RMSNORM: Optional[ctypes.c_void_p] = None
_PSO_GEMM: Optional[ctypes.c_void_p] = None
_INIT_LOCK = threading.Lock()


class MTLSize(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_ulonglong),
        ("height", ctypes.c_ulonglong),
        ("depth", ctypes.c_ulonglong),
    ]


# Metal Shading Language (MSL) Kernel Source
MSL_KERNELS = """
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_f32(
    device const float *in [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint base = id * cols;
    float sum_sq = 0.0f;
    for (uint i = 0; i < cols; ++i) {
        float val = in[base + i];
        sum_sq += val * val;
    }
    float scale = rsqrt(sum_sq / float(cols) + eps);
    for (uint i = 0; i < cols; ++i) {
        out[base + i] = in[base + i] * scale * weight[i];
    }
}

kernel void gemm_f32(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;
    uint col = tid.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""


def _objc_send(ret_type, target, sel_name: str | bytes, *args):
    """Invoke Objective-C method via objc_msgSend with dynamic signature."""
    global _OBJC
    if _OBJC is None:
        raise RuntimeError("Objective-C runtime not loaded")

    sel_bytes = sel_name.encode("utf-8") if isinstance(sel_name, str) else sel_name
    sel = _OBJC.sel_registerName(sel_bytes)

    argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    for arg in args:
        if isinstance(arg, int):
            argtypes.append(ctypes.c_ulonglong)
        elif isinstance(arg, float):
            argtypes.append(ctypes.c_float)
        elif (
            isinstance(arg, (ctypes.c_void_p, ctypes.c_char_p))
            or isinstance(arg, ctypes.Structure)
            or isinstance(arg, ctypes._Pointer)
        ):
            argtypes.append(type(arg))
        else:
            argtypes.append(ctypes.c_void_p)

    func_type = ctypes.CFUNCTYPE(ret_type, *argtypes)
    fn = func_type(ctypes.cast(_OBJC.objc_msgSend, ctypes.c_void_p).value)
    return fn(target, sel, *args)


def _release_buffers(*buffers: Optional[ctypes.c_void_p]) -> None:
    """Release Objective-C / Metal buffers created with retain count +1."""
    for buf in buffers:
        if buf:
            _objc_send(None, buf, "release")


@contextmanager
def _managed_buffers(*buffers: Optional[ctypes.c_void_p]):
    try:
        yield
    finally:
        _release_buffers(*buffers)


def _cf_string(s: str) -> ctypes.c_void_p:
    """Create NSString / CFStringRef from Python string."""
    global _CF
    if _CF is None:
        raise RuntimeError("CoreFoundation not loaded")
    _CF.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    _CF.CFStringCreateWithCString.restype = ctypes.c_void_p
    return _CF.CFStringCreateWithCString(None, s.encode("utf-8"), 0x08000100)


def _init_metal() -> bool:
    """Initialize Metal device, compile JIT kernels, and prepare pipelines."""
    if _METAL_INITIALIZED:
        return _IS_METAL_AVAILABLE
    with _INIT_LOCK:
        return _init_metal_unlocked()


def _init_metal_unlocked() -> bool:
    global _METAL_INITIALIZED, _IS_METAL_AVAILABLE
    global _OBJC, _METAL, _CF, _DEVICE, _COMMAND_QUEUE
    global _PSO_RMSNORM, _PSO_GEMM

    if _METAL_INITIALIZED:
        return _IS_METAL_AVAILABLE

    if sys.platform != "darwin":
        _IS_METAL_AVAILABLE = False
        _METAL_INITIALIZED = True
        return False

    try:
        objc_lib = util.find_library("objc")
        metal_lib = util.find_library("Metal")
        cf_lib = util.find_library("CoreFoundation")

        if not objc_lib or not metal_lib or not cf_lib:
            return False

        _OBJC = ctypes.cdll.LoadLibrary(objc_lib)
        _METAL = ctypes.cdll.LoadLibrary(metal_lib)
        _CF = ctypes.cdll.LoadLibrary(cf_lib)

        _OBJC.sel_registerName.argtypes = [ctypes.c_char_p]
        _OBJC.sel_registerName.restype = ctypes.c_void_p

        _METAL.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        _DEVICE = _METAL.MTLCreateSystemDefaultDevice()
        if not _DEVICE:
            return False

        _COMMAND_QUEUE = _objc_send(ctypes.c_void_p, _DEVICE, "newCommandQueue")
        if not _COMMAND_QUEUE:
            return False

        # JIT compile MSL kernel source
        err = ctypes.c_void_p()
        src_cf = _cf_string(MSL_KERNELS)
        library = _objc_send(
            ctypes.c_void_p,
            _DEVICE,
            "newLibraryWithSource:options:error:",
            src_cf,
            None,
            ctypes.byref(err),
        )
        if not library:
            return False

        # Compile RMSNorm pipeline
        fn_rms = _objc_send(
            ctypes.c_void_p, library, "newFunctionWithName:", _cf_string("rmsnorm_f32")
        )
        if not fn_rms:
            return False
        _PSO_RMSNORM = _objc_send(
            ctypes.c_void_p,
            _DEVICE,
            "newComputePipelineStateWithFunction:error:",
            fn_rms,
            ctypes.byref(err),
        )

        # Compile GEMM pipeline
        fn_gemm = _objc_send(
            ctypes.c_void_p, library, "newFunctionWithName:", _cf_string("gemm_f32")
        )
        if not fn_gemm:
            return False
        _PSO_GEMM = _objc_send(
            ctypes.c_void_p,
            _DEVICE,
            "newComputePipelineStateWithFunction:error:",
            fn_gemm,
            ctypes.byref(err),
        )

        _IS_METAL_AVAILABLE = bool(_PSO_RMSNORM and _PSO_GEMM)
        return _IS_METAL_AVAILABLE
    except Exception:
        _IS_METAL_AVAILABLE = False
        return False
    finally:
        _METAL_INITIALIZED = True


def is_metal_available() -> bool:
    """Return True if running on Apple Silicon / macOS with Metal JIT available."""
    return _init_metal()


def get_metal_device_name() -> Optional[str]:
    """Return the name of the active Metal device or None if unavailable."""
    if not is_metal_available():
        return None
    try:
        nsstr = _objc_send(ctypes.c_void_p, _DEVICE, "name")
        utf8_fn = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(
            ctypes.cast(_OBJC.objc_msgSend, ctypes.c_void_p).value
        )
        c_str = utf8_fn(nsstr, _OBJC.sel_registerName(b"UTF8String"))
        return c_str.decode("utf-8") if c_str else "Apple Metal GPU"
    except Exception:
        return "Apple Metal GPU"


def is_cuda_available() -> bool:
    """
    Return True if CUDA runtime / driver is available.
    Designed for unified Parallax JIT acceleration interface.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        pass

    cuda_lib = util.find_library("cuda") or util.find_library("cudart")
    return cuda_lib is not None


def metal_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication C = A @ B accelerated via Metal compute shader.
    A: 2D array of shape (M, K), float32
    B: 2D array of shape (K, N), float32
    Returns: C of shape (M, N), float32
    """
    if np.iscomplexobj(A) or np.iscomplexobj(B):
        raise TypeError("Complex dtypes are not supported")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D matrices, got shapes {A.shape} and {B.shape}"
        )

    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Shape mismatch for GEMM: ({M}, {K}) @ ({K2}, {N})")

    if M > _INT32_MAX or N > _INT32_MAX or K > _INT32_MAX or (M * K) > _INT32_MAX or (K * N) > _INT32_MAX or (M * N) > _INT32_MAX:
        raise ValueError(f"Matrix dimensions exceed INT32_MAX: ({M}, {K}) @ ({K2}, {N})")

    if M == 0 or N == 0 or K == 0 or A.dtype != np.float32 or B.dtype != np.float32 or not is_metal_available():
        return np.matmul(A, B)

    A_c = np.ascontiguousarray(A, dtype=np.float32)
    B_c = np.ascontiguousarray(B, dtype=np.float32)

    buf_A = _objc_send(
        ctypes.c_void_p,
        _DEVICE,
        "newBufferWithBytes:length:options:",
        A_c.ctypes.data_as(ctypes.c_void_p),
        A_c.nbytes,
        0,
    )
    buf_B = _objc_send(
        ctypes.c_void_p,
        _DEVICE,
        "newBufferWithBytes:length:options:",
        B_c.ctypes.data_as(ctypes.c_void_p),
        B_c.nbytes,
        0,
    )
    out_bytes = M * N * 4
    buf_C = _objc_send(
        ctypes.c_void_p, _DEVICE, "newBufferWithLength:options:", out_bytes, 0
    )

    cmd_buf = _objc_send(ctypes.c_void_p, _COMMAND_QUEUE, "commandBuffer")
    encoder = _objc_send(ctypes.c_void_p, cmd_buf, "computeCommandEncoder")

    _objc_send(None, encoder, "setComputePipelineState:", _PSO_GEMM)
    _objc_send(None, encoder, "setBuffer:offset:atIndex:", buf_A, 0, 0)
    _objc_send(None, encoder, "setBuffer:offset:atIndex:", buf_B, 0, 1)
    _objc_send(None, encoder, "setBuffer:offset:atIndex:", buf_C, 0, 2)

    u_M = ctypes.c_uint32(M)
    u_N = ctypes.c_uint32(N)
    u_K = ctypes.c_uint32(K)
    _objc_send(None, encoder, "setBytes:length:atIndex:", ctypes.byref(u_M), 4, 3)
    _objc_send(None, encoder, "setBytes:length:atIndex:", ctypes.byref(u_N), 4, 4)
    _objc_send(None, encoder, "setBytes:length:atIndex:", ctypes.byref(u_K), 4, 5)

    grid_size = MTLSize(N, M, 1)
    tg_w = min(N, 16)
    tg_h = min(M, 16)
    threadgroup_size = MTLSize(tg_w, tg_h, 1)

    dispatch_fn = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize
    )(ctypes.cast(_OBJC.objc_msgSend, ctypes.c_void_p).value)
    dispatch_fn(
        encoder,
        _OBJC.sel_registerName(b"dispatchThreads:threadsPerThreadgroup:"),
        grid_size,
        threadgroup_size,
    )

    _objc_send(None, encoder, "endEncoding")
    _objc_send(None, cmd_buf, "commit")
    _objc_send(None, cmd_buf, "waitUntilCompleted")

    contents_ptr = _objc_send(ctypes.c_void_p, buf_C, "contents")
    res = np.ctypeslib.as_array(
        ctypes.cast(contents_ptr, ctypes.POINTER(ctypes.c_float)), shape=(M, N)
    ).copy()
    _release_buffers(buf_A, buf_B, buf_C)
    return res


def metal_rmsnorm(
    x: np.ndarray, weight: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """
    RMSNorm accelerated via Metal compute shader.
    x: input array of shape (..., cols), float32
    weight: scale vector of shape (cols,), float32
    eps: numerical epsilon
    Returns: normalized array of same shape as x, float32
    """
    if np.iscomplexobj(x) or np.iscomplexobj(weight):
        raise TypeError("Complex dtypes are not supported")
    if x.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension")

    orig_shape = x.shape
    cols = orig_shape[-1]
    if cols == 0:
        raise ValueError("Feature dimension cannot be 0")
    if weight.shape != (cols,):
        raise ValueError(
            f"Weight shape {weight.shape} does not match last dimension {cols}"
        )

    if x.size == 0:
        return np.empty(orig_shape, dtype=x.dtype if x.dtype == np.float32 else np.float32)

    rows = x.size // cols
    if rows > _INT32_MAX or cols > _INT32_MAX or (rows * cols) > _INT32_MAX:
        raise ValueError(f"Dimensions exceed INT32_MAX: rows={rows}, cols={cols}")

    if not is_metal_available() or x.dtype != np.float32 or weight.dtype != np.float32:
        variance = np.mean(x**2, axis=-1, keepdims=True)
        return (x / np.sqrt(variance + eps)) * weight

    x_c = np.ascontiguousarray(x.reshape(rows, cols), dtype=np.float32)
    w_c = np.ascontiguousarray(weight, dtype=np.float32)

    buf_x = _objc_send(
        ctypes.c_void_p,
        _DEVICE,
        "newBufferWithBytes:length:options:",
        x_c.ctypes.data_as(ctypes.c_void_p),
        x_c.nbytes,
        0,
    )
    buf_w = _objc_send(
        ctypes.c_void_p,
        _DEVICE,
        "newBufferWithBytes:length:options:",
        w_c.ctypes.data_as(ctypes.c_void_p),
        w_c.nbytes,
        0,
    )
    buf_out = _objc_send(
        ctypes.c_void_p, _DEVICE, "newBufferWithLength:options:", x_c.nbytes, 0
    )

    cmd_buf = _objc_send(ctypes.c_void_p, _COMMAND_QUEUE, "commandBuffer")
    encoder = _objc_send(ctypes.c_void_p, cmd_buf, "computeCommandEncoder")

    _objc_send(None, encoder, "setComputePipelineState:", _PSO_RMSNORM)
    _objc_send(None, encoder, "setBuffer:offset:atIndex:", buf_x, 0, 0)
    _objc_send(None, encoder, "setBuffer:offset:atIndex:", buf_w, 0, 1)
    _objc_send(None, encoder, "setBuffer:offset:atIndex:", buf_out, 0, 2)

    u_cols = ctypes.c_uint32(cols)
    u_eps = ctypes.c_float(eps)
    _objc_send(None, encoder, "setBytes:length:atIndex:", ctypes.byref(u_cols), 4, 3)
    _objc_send(None, encoder, "setBytes:length:atIndex:", ctypes.byref(u_eps), 4, 4)

    grid_size = MTLSize(rows, 1, 1)
    tg_size = MTLSize(min(rows, 32), 1, 1)

    dispatch_fn = ctypes.CFUNCTYPE(
        None, ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize
    )(ctypes.cast(_OBJC.objc_msgSend, ctypes.c_void_p).value)
    dispatch_fn(
        encoder,
        _OBJC.sel_registerName(b"dispatchThreads:threadsPerThreadgroup:"),
        grid_size,
        tg_size,
    )

    _objc_send(None, encoder, "endEncoding")
    _objc_send(None, cmd_buf, "commit")
    _objc_send(None, cmd_buf, "waitUntilCompleted")

    contents_ptr = _objc_send(ctypes.c_void_p, buf_out, "contents")
    res = np.ctypeslib.as_array(
        ctypes.cast(contents_ptr, ctypes.POINTER(ctypes.c_float)),
        shape=(rows, cols),
    ).copy()
    _release_buffers(buf_x, buf_w, buf_out)
    return res.reshape(orig_shape)
