"""Tests for Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor
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


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-2.0) == 1e-12
