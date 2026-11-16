"""Fused RMSNorm dispatch with epsilon stabilization."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring positive strictly-bounded value."""
    eps_val = float(eps)
    return max(eps_val, min_eps)


def _dispatch_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("rmsnorm", x, weight, eps=eps)
    except Exception:
        variance = np.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(np.maximum(variance, 0.0) + eps)
        return (x * inv_rms) * weight


def fused_rmsnorm(
    x: np.ndarray,
    weight: np.ndarray,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    return_residual: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute RMSNorm with fused residual addition and epsilon stabilization."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual if residual is not None else x
    orig_dtype = fused.dtype
    acc = fused.astype(np.float32) if orig_dtype in (np.float16, np.int32) else fused
    normed = _dispatch_rmsnorm(acc, weight.astype(acc.dtype), eps_val)
    out = normed.astype(orig_dtype)
    if return_residual:
        return out, fused
    return out


class FusedRMSNorm(Module):
    """Fused Root Mean Square Layer Normalization module."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = Parameter(np.ones(self.dim, dtype=np.float32), label="RMSNorm.weight")

    def forward(
        self,
        x: Tensor | np.ndarray,
        residual: Optional[Tensor | np.ndarray] = None,
    ) -> Tensor:
        raw_x = x.data if isinstance(x, Tensor) else x
        raw_res = residual.data if isinstance(residual, Tensor) else residual
        out_data = fused_rmsnorm(raw_x, self.weight.data, residual=raw_res, eps=self.eps)
        parents = [p for p in (x, residual, self.weight) if isinstance(p, Tensor)]
        return Tensor(out_data, tuple(parents), "FusedRMSNorm")

    @classmethod
    def from_norm(cls, norm: Any) -> FusedRMSNorm:
        dim = getattr(norm, "dim", getattr(norm, "normalized_shape", (0,))[0] if hasattr(norm, "normalized_shape") else 0)
        eps = getattr(norm, "eps", 1e-6)
        mod = cls(dim=dim, eps=eps)
        if hasattr(norm, "weight") and norm.weight is not None:
            mod.weight = Parameter(np.asarray(getattr(norm.weight, "data", norm.weight)).copy())
        return mod

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def __repr__(self) -> str:
        return f"FusedRMSNorm({self.extra_repr()})"






