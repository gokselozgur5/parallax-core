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


def conv2d_backward(
    dout: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradients w.r.t input and weight for 2D convolution."""
    sh, sw = stride
    ph, pw = padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x.copy()
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(weight)
    h_out, w_out = dout.shape[2], dout.shape[3]
    for i in range(h_out):
        for j in range(w_out):
            cur = dout[:, :, i, j]
            kh, kw = weight.shape[2], weight.shape[3]
            patch = x_pad[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
            dw += np.tensordot(cur, patch, axes=(0, 0))
            dx_pad[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw] += np.tensordot(cur, weight, axes=(1, 0))
    dx = dx_pad[:, :, ph : ph + x.shape[2], pw : pw + x.shape[3]] if (ph or pw) else dx_pad
    return dx, dw


def _backward_conv2d(
    grad: np.ndarray,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> None:
    """Accumulate autograd gradients for conv2d inputs."""
    dx, dw = conv2d_backward(grad, x.data, weight.data, stride, padding)
    if x.requires_grad:
        _accumulate_grad(x, dx)
    if weight.requires_grad:
        _accumulate_grad(weight, dw)
    if bias is not None and bias.requires_grad:
        _accumulate_grad(bias, np.sum(grad, axis=(0, 2, 3)))


def dispatched_conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    """Dispatched 2D convolution with autograd backward support."""
    if not isinstance(x, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("dispatched_conv2d requires Tensor inputs for x and weight")
    st = _pair(stride)
    pd = _pair(padding)
    b_data = bias.data if bias is not None else None
    try:
        out_data = dispatch("conv2d", x.data, weight.data, b_data, st, pd)
    except Exception:
        out_data = conv2d_forward(x.data, weight.data, b_data, st, pd)
    parents = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(out_data, parents, "dispatched_conv2d")
    if not out.requires_grad:
        return out
    out._backward = lambda: _backward_conv2d(out.grad, x, weight, bias, st, pd) if out.grad is not None else None
    return out


class DispatchedConv2d(Module):
    """Drop-in 2D convolution layer backed by backend dispatch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        kh, kw = self.kernel_size
        limit = 1.0 / np.sqrt(in_channels * kh * kw)
        w_init = np.random.uniform(-limit, limit, (out_channels, in_channels, kh, kw)).astype(np.float32)
        self.weight = Parameter(w_init, label="Conv2d.weight")
        self.bias: Parameter | None = (
            Parameter(np.zeros(out_channels, dtype=np.float32), label="Conv2d.bias") if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return dispatched_conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    @classmethod
    def from_conv2d(cls, conv: Any) -> DispatchedConv2d:
        bias_present = getattr(conv, "bias", None) is not None
        mod = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=bias_present)
        mod.weight = conv.weight
        mod.bias = conv.bias
        return mod

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None}"
        )

    def __repr__(self) -> str:
        return f"DispatchedConv2d({self.extra_repr()})"
