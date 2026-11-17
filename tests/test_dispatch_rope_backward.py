"""Tests for Rotary Positional Embedding (RoPE) backward functional kernel."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
from python.parallax.dispatch.rope import apply_rotary_emb, precompute_freqs
from python.parallax.dispatch.rope_backward import (
    DispatchedRoPEBackward,
    dispatched_rope,
    rope_backward,
    rope_backward_reference,
    stabilize_theta,
)


def test_stabilize_theta() -> None:
    assert stabilize_theta(10000.0) == 10000.0
    assert stabilize_theta(0.0) == 1e-4
    assert stabilize_theta(-50.0) == 1e-4
