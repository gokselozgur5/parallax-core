"""Direct dispatch layernorm compatibility alias."""

from python.parallax.dispatch.layernorm import (
    FusedLayerNorm,
    fused_layernorm,
    stabilize_epsilon,
)

__all__ = [
    "FusedLayerNorm",
    "fused_layernorm",
    "stabilize_epsilon",
]
