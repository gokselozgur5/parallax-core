"""Direct dispatch cross entropy compatibility alias."""

from python.parallax.dispatch.cross_entropy import (
    CrossEntropyLossModule,
    DispatchedCrossEntropyLoss,
    cross_entropy_backward_reference,
    cross_entropy_forward_reference,
    fast_cross_entropy,
    online_logsumexp,
)

__all__ = [
    "CrossEntropyLossModule",
    "DispatchedCrossEntropyLoss",
    "cross_entropy_backward_reference",
    "cross_entropy_forward_reference",
    "fast_cross_entropy",
    "online_logsumexp",
]
