"""Direct dispatch rmsnorm backward compatibility alias."""

from parallax.dispatch.rmsnorm_backward import (
    dispatched_rmsnorm,
    fused_rmsnorm_backward,
    rmsnorm_backward,
    rmsnorm_backward_reference,
    stabilize_epsilon,
)

__all__ = [
    "dispatched_rmsnorm",
    "fused_rmsnorm_backward",
    "rmsnorm_backward",
    "rmsnorm_backward_reference",
    "stabilize_epsilon",
]
