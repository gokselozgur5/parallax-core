"""Direct dispatch conv2d compatibility alias."""

from python.parallax.dispatch.conv2d import (
    DispatchedConv2d,
    conv2d_backward,
    conv2d_forward,
    dispatched_conv2d,
)

__all__ = [
    "DispatchedConv2d",
    "conv2d_backward",
    "conv2d_forward",
    "dispatched_conv2d",
]
