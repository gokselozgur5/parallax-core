"""
BitNet b1.58 (1.58-bit LLMs: Ma et al., 2024) mathematical primitives.
"""

from typing import Tuple
import numpy as np
from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def quantize_weights_ternary(weight: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantizes continuous weights into ternary values {-1, 0, +1} using average absolute mean scaling."""
    # gamma = mean(|W|)
    gamma = float(np.mean(np.abs(weight))) + 1e-12
    # Scaled and rounded into {-1, 0, +1}
    w_scaled = weight / gamma
    w_ternary = np.clip(np.round(w_scaled), -1.0, 1.0)
    return w_ternary, gamma


def quantize_activations_int8(x: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Quantizes activation vectors to 8-bit integers using per-token Absmax scaling."""
    # scale = 127 / max(|x|) per token along feature axis
    max_val = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True), eps)
    scale = 127.0 / max_val
    x_quant = np.clip(np.round(x * scale), -128.0, 127.0)
    return x_quant, scale


class BitLinear(Module):
    """1.58-bit Ternary Linear layer with Straight-Through Estimator (STE) backpropagation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(-limit, limit, (out_features, in_features))
        self.weight = Parameter(weight_data, label="BitLinear.weight")

        if bias:
            self.bias = Parameter(np.zeros(out_features), label="BitLinear.bias")
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: quantizes W to {-1, 0, 1} and x to INT8, performing addition-based matmul."""
        w_data = self.weight.data
        x_data = x.data

        # 1. Quantize weights and activations
        w_ternary, gamma = quantize_weights_ternary(w_data)
        x_quant, x_scale = quantize_activations_int8(x_data)

        # 2. Integer addition-dominated matrix multiplication:
        # y_int = x_quant @ w_ternary^T
        y_int = np.matmul(x_quant, w_ternary.T)

        # 3. Dequantize: y = (gamma / x_scale) * y_int
        dequant_scale = gamma / x_scale
        out_data = y_int * dequant_scale

        if self.bias is not None:
            out_data += self.bias.data

        children = (x, self.weight, self.bias) if self.bias is not None else (x, self.weight)
        out = Tensor(out_data, children, "BitLinear")

        def _backward() -> None:
            if out.grad is None:
                return

            # Straight-Through Estimator (STE): gradient passes directly to continuous weights & input
            # Forward computes: y = (gamma / x_scale) * (x_quant @ W_ternary^T) = (x_quant / x_scale) @ (gamma * W_ternary^T)
            # Under STE with respect to continuous W, grad_w uses the dequantized activations (x_quant / x_scale)
            x_dequant = x_quant / x_scale
            grad_w = np.matmul(out.grad.reshape(-1, self.out_features).T, x_dequant.reshape(-1, self.in_features))
            if self.weight.requires_grad:
                _accumulate_grad(self.weight, grad_w)

            # dL/dX = dL/dY @ W_eff = dL/dY @ (W_ternary * gamma)
            grad_x = np.matmul(out.grad, w_ternary) * gamma
            if x.requires_grad:
                _accumulate_grad(x, grad_x)

            if self.bias is not None and self.bias.requires_grad:
                _accumulate_grad(self.bias, np.sum(out.grad.reshape(-1, self.out_features), axis=0))

        out._backward = _backward
        return out


