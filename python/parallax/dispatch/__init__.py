"""Dispatch modules for hardware-accelerated and optimized neural network layers."""

from python.parallax.dispatch.dropout import (
    DispatchedDropout,
    DropoutModule,
    dispatched_dropout,
    dropout_backward,
    dropout_backward_reference,
    dropout_forward,
    dropout_forward_reference,
    dropout_mask_generator,
)
from python.parallax.dispatch.softmax_backward import (
    DispatchedSoftmax,
    SoftmaxModule,
    dispatched_softmax,
    fused_masked_softmax_backward,
    softmax_backward,
    softmax_backward_from_logits,
    softmax_backward_jvp,
    softmax_backward_reference,
    softmax_forward_reference,
)

from python.parallax.dispatch.fused_rmsnorm_residual import (
    FusedRMSNormResidual,
    dispatched_fused_rmsnorm_residual,
    fused_rmsnorm_residual_backward,
    fused_rmsnorm_residual_backward_reference,
    fused_rmsnorm_residual_forward,
    fused_rmsnorm_residual_forward_reference,
)

__all__ = [
    "FusedRMSNormResidual",
    "dispatched_fused_rmsnorm_residual",
    "fused_rmsnorm_residual_backward",
    "fused_rmsnorm_residual_backward_reference",
    "fused_rmsnorm_residual_forward",
    "fused_rmsnorm_residual_forward_reference",
    "DispatchedSoftmax",
    "SoftmaxModule",
    "dispatched_softmax",
    "fused_masked_softmax_backward",
    "softmax_backward",
    "softmax_backward_from_logits",
    "softmax_backward_jvp",
    "softmax_backward_reference",
    "softmax_forward_reference",
    "DispatchedDropout",
    "DropoutModule",
    "dispatched_dropout",
    "dropout_backward",
    "dropout_backward_reference",
    "dropout_forward",
    "dropout_forward_reference",
    "dropout_mask_generator",
]
