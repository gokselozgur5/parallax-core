"""DispatchedConv2d module routing 2D convolution through backend dispatch."""

from __future__ import annotations

from typing import Any, Tuple, Union
import numpy as np

from parallax.backends.dispatch import dispatch
from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def _pair(val: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert int or tuple to a 2-tuple of ints."""
    if isinstance(val, (int, np.integer)):
        return (int(val), int(val))
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return (int(val[0]), int(val[1]))
    raise ValueError(f"Expected int or pair of ints, got {val}")


def conv2d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Forward 2D convolution over 4D input [B, C_in, H, W]."""
    b_sz, _, h_in, w_in = x.shape
    out_c, _, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x
    h_out = (x_pad.shape[2] - kh) // sh + 1
    w_out = (x_pad.shape[3] - kw) // sw + 1
    out = np.zeros((b_sz, out_c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        for j in range(w_out):
            patch = x_pad[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
            out[:, :, i, j] = np.tensordot(patch, weight, axes=((1, 2, 3), (1, 2, 3)))
    if bias is not None:
        out += bias[None, :, None, None]
    return out
