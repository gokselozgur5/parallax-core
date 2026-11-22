"""Dispatched tensor operations with autograd awareness."""

from __future__ import annotations

import numpy as np
from parallax.backends import registry
from parallax.backends.dispatch import (
    _record_fallback,
    _resolve_backend,
    candidate_backends,
    detect_cpu,
    dispatch,
    get_scope_backend,
)
from parallax.backends.env import backend_override
from parallax.backends.policy import native_order
from parallax.backends.scope import current_backend
from parallax.tensor import Tensor, _accumulate_grad


def dispatched_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication of two Tensors routed through dispatch('gemm')."""
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError("dispatched_matmul requires Tensor arguments")

    a_data, b_data = a.data, b.data
    target_f32 = False
    for name in candidate_backends("gemm", n_elements=a_data.size):
        if name == "numpy":
            break
        be = _resolve_backend(name)
        if be is not None and be.available():
            if not be.supports("gemm", a_data.dtype, a_data.shape) and be.supports("gemm", np.float32, a_data.shape):
                target_f32 = True
                break

    if target_f32:
        out_data = dispatch("gemm", a_data.astype(np.float32, copy=False), b_data.astype(np.float32, copy=False))
        out_data = out_data.astype(a_data.dtype, copy=False)
    else:
        out_data = dispatch("gemm", a_data, b_data)

    out = Tensor(out_data, (a, b), "dispatched_matmul")

    if not out.requires_grad:
        return out

    def _backward() -> None:
        if out.grad is None:
            return
        a_is_1d, b_is_1d = (a.ndim == 1), (b.ndim == 1)
        a_2d = a.data[None, :] if a_is_1d else a.data
        b_2d = b.data[..., :, None] if b_is_1d else b.data
        if a_is_1d and b_is_1d:
            g_2d = np.array([[float(out.grad)]])
        elif a_is_1d:
            g_2d = out.grad[..., None, :]
        elif b_is_1d:
            g_2d = out.grad[..., None]
        else:
            g_2d = out.grad

        if a.requires_grad:
            other_t = np.swapaxes(b_2d, -1, -2)
            grad_self = g_2d @ other_t
            if a_is_1d:
                grad_self = np.squeeze(grad_self, axis=-2)
            _accumulate_grad(a, grad_self)
        if b.requires_grad:
            self_t = np.swapaxes(a_2d, -1, -2)
            grad_other = self_t @ g_2d
            if b_is_1d:
                grad_other = np.squeeze(grad_other, axis=-1)
            _accumulate_grad(b, grad_other)

    out._backward = _backward
    return out


def _dispatch_gather_data(weight_data: np.ndarray, idx_arr: np.ndarray) -> np.ndarray:
    """Route row-gather lookup through backend dispatch with NumPy fallback."""
    sb = current_backend("gather") or get_scope_backend()
    ov = backend_override("gather")
    forced = {x for x in (sb, ov) if x is not None}
    candidates: list[str] = []
    for name in (sb, ov, *native_order(detect_cpu()), *registry.list_backends(), "numpy"):
        if name is not None and name not in candidates:
            candidates.append(name)
    last_err: Exception | None = None
    for name in candidates:
        be = _resolve_backend(name)
        failed = False
        if be is not None and be.available():
            fn = getattr(be, "gather", None)
            if callable(fn):
                for dt in (weight_data.dtype, np.dtype(np.float32)):
                    if be.supports("gather", dt, weight_data.shape):
                        try:
                            arg = weight_data.astype(dt, copy=False)
                            return np.asarray(fn(arg, idx_arr)).astype(weight_data.dtype, copy=False)
                        except Exception as exc:
                            last_err = exc
                            failed = True
                            _record_fallback("gather", name, str(exc), exc=exc)
                        break
        if failed:
            continue
        if name == "numpy":
            return weight_data[idx_arr]
        if name in forced:
            _record_fallback("gather", name, "backend not available or unsupported")
    raise RuntimeError(f"All backends failed for gather: {candidates}") from last_err


def dispatched_gather(
    weight: Tensor,
    indices: Tensor | np.ndarray,
    padding_idx: int | None = None,
) -> Tensor:
    """Gather rows of a 2D weight Tensor routed through backend dispatch."""
    if not isinstance(weight, Tensor):
        raise TypeError("dispatched_gather requires a Tensor weight")
    if weight.ndim != 2:
        raise ValueError("dispatched_gather requires a 2D weight Tensor")

    raw_idx = indices.data if isinstance(indices, Tensor) else indices
    idx_arr = np.asarray(raw_idx, dtype=np.int64)
    if np.any(idx_arr < 0) or np.any(idx_arr >= weight.data.shape[0]):
        raise IndexError("Index out of range for DispatchedEmbedding")

    out_data = _dispatch_gather_data(weight.data, idx_arr)
    parents = (weight,) if not isinstance(indices, Tensor) else (indices, weight)
    out = Tensor(out_data, parents, "dispatched_gather")
    if not out.requires_grad:
        return out

    def _backward() -> None:
        if out.grad is None or not weight.requires_grad:
            return
        grad_w = np.zeros_like(weight.data)
        np.add.at(grad_w, idx_arr, out.grad)
        if padding_idx is not None:
            grad_w[padding_idx] = 0.0
        _accumulate_grad(weight, grad_w)

    out._backward = _backward
    return out


