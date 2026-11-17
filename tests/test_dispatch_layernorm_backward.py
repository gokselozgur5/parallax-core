"""Tests for LayerNorm backward autograd functional kernel."""

from __future__ import annotations

import numpy as np
import pytest

from python.parallax.dispatch.layernorm_backward import (
    stabilize_epsilon,
)


def test_stabilize_epsilon():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(0.0) == 1e-12
    assert stabilize_epsilon(-5.0) == 1e-12
