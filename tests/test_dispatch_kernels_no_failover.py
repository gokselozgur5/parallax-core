"""Reproduction test verifying dispatched kernels do not fail over on every call."""

import numpy as np

from parallax.backends import registry
from parallax.backends.dispatch import (
    clear_fallback_history,
    get_fallback_history,
)
from parallax.backends.numpy_backend import NumpyBackend
from parallax.dispatch import (
    fused_layernorm,
    softmax_backward,
)
from parallax.dispatch.dropout import dropout_forward


def test_dispatched_kernels_do_not_fail_over_on_every_call():
    added = "numpy" not in registry.list_backends()
    if added:
        registry.register("numpy", NumpyBackend())
    clear_fallback_history()
    x = np.full((2, 4), 0.5)
    try:
        fused_layernorm(x)
        dropout_forward(x, p=0.5, seed=0)
        softmax_backward(x, x)
        failed = sorted({record.op for record in get_fallback_history()})
    finally:
        clear_fallback_history()
        if added:
            registry.unregister("numpy")
    assert failed == [], f"no registered backend implements: {failed}"
