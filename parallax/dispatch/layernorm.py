"""Direct dispatch layernorm compatibility alias."""

from parallax.dispatch.layernorm import (
    FusedLayerNorm,
    fused_layernorm,
    stabilize_epsilon,
)

__all__ = [
    "FusedLayerNorm",
    "fused_layernorm",
    "stabilize_epsilon",
]
