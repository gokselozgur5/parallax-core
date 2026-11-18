"""Direct dispatch softmax backward compatibility alias."""

from parallax.dispatch.softmax_backward import (
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

__all__ = [
    "DispatchedSoftmax",
    "SoftmaxModule",
    "dispatched_softmax",
    "fused_masked_softmax_backward",
    "softmax_backward",
    "softmax_backward_from_logits",
    "softmax_backward_jvp",
    "softmax_backward_reference",
    "softmax_forward_reference",
]
