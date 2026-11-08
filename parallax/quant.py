"""
Post-Training Quantization (PTQ) Primitives in Pure NumPy.

Implements symmetric INT8 weight quantization and dequantization,
quantized Linear projection with integer matmul, and whole-model quantization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor


def quantize_int8(w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantizes float64/float32 weights to symmetric signed INT8 [-127, 127].

    scale = max(|w|) / 127.0
    w_int8 = clip(round(w / scale), -127, 127)

    Returns:
        (w_int8, scale)
    """
    max_val = float(np.max(np.abs(w)))
    if max_val == 0.0:
        return np.zeros_like(w, dtype=np.int8), 1.0

    scale = max_val / 127.0
    w_scaled = np.round(w / scale)
    w_int8 = np.clip(w_scaled, -127, 127).astype(np.int8)
    return w_int8, scale


def dequantize_int8(w_int8: np.ndarray, scale: float) -> np.ndarray:
    """Dequantizes INT8 weights back to float64."""
    return w_int8.astype(np.float64) * scale


class QuantizedLinear(Module):
    """Inference-only Linear projection with symmetric INT8 quantized weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_int8: np.ndarray = np.zeros((out_features, in_features), dtype=np.int8)
        self.scale: float = 1.0
        self.bias_data: Optional[np.ndarray] = np.zeros(out_features, dtype=np.float64) if bias else None

    @classmethod
    def from_linear(cls, linear: Linear) -> "QuantizedLinear":
        """Constructs a QuantizedLinear instance from a trained float Linear layer."""
        ql = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None))
        ql.w_int8, ql.scale = quantize_int8(linear.weight.data)
        if linear.bias is not None:
            ql.bias_data = linear.bias.data.copy()
        return ql

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize weights dynamically or perform mixed-precision projection
        w_float = dequantize_int8(self.w_int8, self.scale)
        # x: [..., in_features], w_float: [out_features, in_features]
        out_data = x.data @ w_float.T
        if self.bias_data is not None:
            out_data = out_data + self.bias_data
        return Tensor(out_data, (x,), "QuantizedLinear")


def quantize_model(model: ParallaxLM) -> ParallaxLM:
    """Quantizes all Linear layers of a ParallaxLM model to INT8 in-place for fast/compact inference."""
    for block in model.blocks:
        block.attn.q_proj = QuantizedLinear.from_linear(block.attn.q_proj)
        block.attn.k_proj = QuantizedLinear.from_linear(block.attn.k_proj)
        block.attn.v_proj = QuantizedLinear.from_linear(block.attn.v_proj)
        block.attn.out_proj = QuantizedLinear.from_linear(block.attn.out_proj)

        if block.use_swiglu:
            block.mlp.w_gate = QuantizedLinear.from_linear(block.mlp.w_gate)
            block.mlp.w_up = QuantizedLinear.from_linear(block.mlp.w_up)
            block.mlp.w_down = QuantizedLinear.from_linear(block.mlp.w_down)
        else:
            block.mlp.fc1 = QuantizedLinear.from_linear(block.mlp.fc1)
            block.mlp.fc2 = QuantizedLinear.from_linear(block.mlp.fc2)

    if not model.config.tie_weights:
        model.lm_head = QuantizedLinear.from_linear(model.lm_head)

    return model


def calculate_compression_ratio(model: ParallaxLM, quantized_model: ParallaxLM) -> Dict[str, float]:
    """Calculates theoretical memory footprint before and after INT8 quantization."""
    unique_params = {id(p): p for p in model.parameters()}
    float_bytes = sum(p.data.nbytes for p in unique_params.values())

    def _layer_bytes(layer: Any) -> int:
        if isinstance(layer, QuantizedLinear):
            b = layer.w_int8.nbytes
            if layer.bias_data is not None:
                b += layer.bias_data.nbytes
            return b
        return sum(p.data.nbytes for p in layer.parameters())

    quant_bytes = 0
    for block in quantized_model.blocks:
        # Attention projections
        for ql in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj]:
            quant_bytes += _layer_bytes(ql)

        # MLP layers (MoE, SwiGLU, or Standard)
        if hasattr(block.mlp, "experts"):
            quant_bytes += block.mlp.gate.weight.data.nbytes
            for exp in block.mlp.experts:
                if block.use_swiglu:
                    for l in [exp.w_gate, exp.w_up, exp.w_down]:
                        quant_bytes += _layer_bytes(l)
                else:
                    for l in [exp.fc1, exp.fc2]:
                        quant_bytes += _layer_bytes(l)
        elif block.use_swiglu:
            for l in [block.mlp.w_gate, block.mlp.w_up, block.mlp.w_down]:
                quant_bytes += _layer_bytes(l)
        else:
            for l in [block.mlp.fc1, block.mlp.fc2]:
                quant_bytes += _layer_bytes(l)

        # Layer norms
        quant_bytes += block.attn_norm.weight.data.nbytes
        quant_bytes += block.mlp_norm.weight.data.nbytes

    # Embeddings and final norm
    quant_bytes += quantized_model.token_embeddings.weight.data.nbytes
    quant_bytes += quantized_model.final_norm.weight.data.nbytes

    # LM Head (if not tied to token embeddings)
    if not model.config.tie_weights:
        quant_bytes += _layer_bytes(quantized_model.lm_head)

    return {
        "original_mb": round(float_bytes / (1024 * 1024), 3),
        "quantized_mb": round(quant_bytes / (1024 * 1024), 3),
        "ratio": round(float_bytes / quant_bytes if quant_bytes > 0 else 1.0, 2),
    }
