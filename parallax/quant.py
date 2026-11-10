"""
Post-Training Quantization (PTQ) Primitives in Pure NumPy.

Implements symmetric INT8 weight quantization and dequantization,
quantized Linear projection with cached dequantized weights, and whole-model quantization.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from parallax.models.gpt import ParallaxLM
from parallax.nn import Linear, Module
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
        self._cached_weight: Optional[np.ndarray] = None

    def invalidate_cache(self) -> None:
        """Invalidates cached dequantized weights."""
        self._cached_weight = None

    @classmethod
    def from_linear(cls, linear: Linear) -> "QuantizedLinear":
        """Constructs a QuantizedLinear instance from a trained float Linear layer."""
        ql = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None))
        ql.w_int8, ql.scale = quantize_int8(linear.weight.data)
        if linear.bias is not None:
            ql.bias_data = linear.bias.data.copy()
        return ql

    def forward(self, x: Tensor) -> Tensor:
        # Cache dequantized weights during inference to avoid redundant dequantization
        if self._cached_weight is None:
            self._cached_weight = dequantize_int8(self.w_int8, self.scale)
        # x: [..., in_features], w_float: [out_features, in_features]
        out_data = x.data @ self._cached_weight.T
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

        if hasattr(block.mlp, "experts"):
            for exp in block.mlp.experts:
                if block.use_swiglu:
                    exp.w_gate = QuantizedLinear.from_linear(exp.w_gate)
                    exp.w_up = QuantizedLinear.from_linear(exp.w_up)
                    exp.w_down = QuantizedLinear.from_linear(exp.w_down)
                else:
                    exp.fc1 = QuantizedLinear.from_linear(exp.fc1)
                    exp.fc2 = QuantizedLinear.from_linear(exp.fc2)
        elif block.use_swiglu:
            block.mlp.w_gate = QuantizedLinear.from_linear(block.mlp.w_gate)
            block.mlp.w_up = QuantizedLinear.from_linear(block.mlp.w_up)
            block.mlp.w_down = QuantizedLinear.from_linear(block.mlp.w_down)
        else:
            block.mlp.fc1 = QuantizedLinear.from_linear(block.mlp.fc1)
            block.mlp.fc2 = QuantizedLinear.from_linear(block.mlp.fc2)

    if not model.config.tie_weights:
        model.lm_head = QuantizedLinear.from_linear(model.lm_head)

    return model


def calculate_compression_ratio(
    model: ParallaxLM,
    quantized_model: ParallaxLM,
    baseline_dtype: str = "float32",
) -> Dict[str, float]:
    """Calculates theoretical memory footprint before and after INT8 quantization.

    Args:
        model: Baseline model or original model.
        quantized_model: Quantized model.
        baseline_dtype: Explicit baseline dtype ('float32' [4 bytes] or 'float16' [2 bytes]).
    """
    baseline_bytes_per_elem = 2 if baseline_dtype in ("float16", "fp16") else 4

    def _layer_bytes(layer: Any) -> int:
        if isinstance(layer, QuantizedLinear):
            b = layer.w_int8.nbytes
            if layer.bias_data is not None:
                b += layer.bias_data.nbytes
            return b
        return sum(p.data.nbytes for p in layer.parameters())

    def _layer_numel(layer: Any) -> int:
        if isinstance(layer, QuantizedLinear):
            n = layer.w_int8.size
            if layer.bias_data is not None:
                n += layer.bias_data.size
            return n
        return sum(p.data.size for p in layer.parameters())

    quant_bytes = 0
    total_elements = 0

    for block in quantized_model.blocks:
        # Attention projections
        for ql in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj]:
            quant_bytes += _layer_bytes(ql)
            total_elements += _layer_numel(ql)

        # MLP layers (MoE, SwiGLU, or Standard)
        if hasattr(block.mlp, "experts"):
            quant_bytes += block.mlp.gate.weight.data.nbytes
            total_elements += block.mlp.gate.weight.data.size
            for exp in block.mlp.experts:
                if block.use_swiglu:
                    for l in [exp.w_gate, exp.w_up, exp.w_down]:
                        quant_bytes += _layer_bytes(l)
                        total_elements += _layer_numel(l)
                else:
                    for l in [exp.fc1, exp.fc2]:
                        quant_bytes += _layer_bytes(l)
                        total_elements += _layer_numel(l)
        elif block.use_swiglu:
            for l in [block.mlp.w_gate, block.mlp.w_up, block.mlp.w_down]:
                quant_bytes += _layer_bytes(l)
                total_elements += _layer_numel(l)
        else:
            for l in [block.mlp.fc1, block.mlp.fc2]:
                quant_bytes += _layer_bytes(l)
                total_elements += _layer_numel(l)

        # Layer norms
        quant_bytes += block.attn_norm.weight.data.nbytes
        total_elements += block.attn_norm.weight.data.size
        quant_bytes += block.mlp_norm.weight.data.nbytes
        total_elements += block.mlp_norm.weight.data.size

    # Embeddings and final norm
    quant_bytes += quantized_model.token_embeddings.weight.data.nbytes
    total_elements += quantized_model.token_embeddings.weight.data.size
    quant_bytes += quantized_model.final_norm.weight.data.nbytes
    total_elements += quantized_model.final_norm.weight.data.size

    # LM Head (if not tied to token embeddings)
    if not quantized_model.config.tie_weights:
        quant_bytes += _layer_bytes(quantized_model.lm_head)
        total_elements += _layer_numel(quantized_model.lm_head)

    float_bytes = total_elements * baseline_bytes_per_elem

    return {
        "original_mb": round(float_bytes / (1024 * 1024), 3),
        "quantized_mb": round(quant_bytes / (1024 * 1024), 3),
        "ratio": round(float_bytes / quant_bytes if quant_bytes > 0 else 1.0, 2),
    }
