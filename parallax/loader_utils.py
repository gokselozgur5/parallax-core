from typing import Dict, Optional
import numpy as np
from parallax.nn import Linear, Parameter


def _assign_param(param: Parameter, tensors: Dict[str, np.ndarray], key: str) -> None:
    """Assigns tensor data from a weight dictionary into a Parameter, checking shapes."""
    if key not in tensors:
        return
    src = tensors[key]
    if param.data.shape != src.shape:
        raise ValueError(
            f"Shape mismatch for '{key}': expected {param.data.shape}, got {src.shape}"
        )
    param.data = src.copy()
    if param.grad is not None and param.grad.shape != param.data.shape:
        param.grad = np.zeros_like(param.data)


def _assign_bias(proj: Linear, tensors: Dict[str, np.ndarray], key: str) -> None:
    """Assigns bias data from a weight dictionary into a Linear projection layer."""
    if key not in tensors:
        return
    src = tensors[key]
    expected_shape = (proj.out_features,)
    if src.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch for '{key}': expected {expected_shape}, got {src.shape}"
        )
    if proj.bias is None:
        proj.bias = Parameter(src.copy(), label="Linear.bias")
    else:
        proj.bias.data = src.copy()
        if proj.bias.grad is not None and proj.bias.grad.shape != expected_shape:
            proj.bias.grad = np.zeros_like(proj.bias.data)
