"""Direct dispatch fused layernorm compatibility alias."""

from python.parallax.dispatch.fused_layernorm import (
    DispatchedFusedLayerNorm,
    FusedLayerNormRMS,
    dispatched_fused_layernorm,
    fused_layernorm_backward,
    fused_layernorm_forward,
    rms_weight_scale,
    rms_weight_scale_backward,
    stabilize_epsilon,
)

__all__ = [
    "DispatchedFusedLayerNorm",
    "FusedLayerNormRMS",
    "dispatched_fused_layernorm",
    "fused_layernorm_backward",
    "fused_layernorm_forward",
    "rms_weight_scale",
    "rms_weight_scale_backward",
    "stabilize_epsilon",
]
