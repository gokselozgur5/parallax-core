"""Dispatch modules for hardware-accelerated and optimized neural network layers."""

from python.parallax.dispatch.cross_entropy import (
    CrossEntropyLossModule,
    DispatchedCrossEntropyLoss,
    cross_entropy_backward_reference,
    cross_entropy_forward_reference,
    fast_cross_entropy,
    online_logsumexp,
)
from python.parallax.dispatch.conv2d import (
    DispatchedConv2d,
    conv2d_backward,
    conv2d_forward,
    dispatched_conv2d,
)
from python.parallax.dispatch.embedding_backward import (
    DispatchedEmbeddingBackward,
    accumulate_embedding_backward,
    embedding_backward,
    sparse_embedding_backward,
)
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
    "CrossEntropyLossModule",
    "DispatchedCrossEntropyLoss",
    "cross_entropy_backward_reference",
    "cross_entropy_forward_reference",
    "fast_cross_entropy",
    "online_logsumexp",
    "FusedRMSNormResidual",
    "dispatched_fused_rmsnorm_residual",
    "fused_rmsnorm_residual_backward",
    "fused_rmsnorm_residual_backward_reference",
    "fused_rmsnorm_residual_forward",
    "fused_rmsnorm_residual_forward_reference",
    "DispatchedConv2d",
    "DispatchedEmbeddingBackward",
    "accumulate_embedding_backward",
    "embedding_backward",
    "sparse_embedding_backward",
    "conv2d_backward",
    "conv2d_forward",
    "dispatched_conv2d",
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
