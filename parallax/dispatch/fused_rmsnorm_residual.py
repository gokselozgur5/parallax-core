"""Direct dispatch fused RMSNorm residual compatibility alias."""

from python.parallax.dispatch.fused_rmsnorm_residual import (
    FusedRMSNormResidual,
    dispatched_fused_rmsnorm_residual,
    fused_rmsnorm_residual_backward,
    fused_rmsnorm_residual_backward_reference,
    fused_rmsnorm_residual_forward,
    fused_rmsnorm_residual_forward_reference,
    stabilize_epsilon,
)

__all__ = [
    "FusedRMSNormResidual",
    "dispatched_fused_rmsnorm_residual",
    "fused_rmsnorm_residual_backward",
    "fused_rmsnorm_residual_backward_reference",
    "fused_rmsnorm_residual_forward",
    "fused_rmsnorm_residual_forward_reference",
    "stabilize_epsilon",
]
