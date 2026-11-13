from __future__ import annotations

import sys
import numpy as np
import pytest
from parallax.backends.metal_bridge import MetalTensorBridge
from parallax.metal_engine import is_metal_available
from parallax.tensor import Tensor


def test_metal_bridge_availability() -> None:
    bridge = MetalTensorBridge()
    if sys.platform != "darwin":
        assert not bridge.available()
    else:
        assert bridge.available() == is_metal_available()


def test_metal_bridge_unwrap_array() -> None:
    bridge = MetalTensorBridge()
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = Tensor(arr)
    assert np.array_equal(bridge.unwrap_array(t), arr)
    assert np.array_equal(bridge.unwrap_array(arr), arr)
    assert np.array_equal(
        bridge.unwrap_array([4, 5]), np.array([4.0, 5.0], dtype=np.float32)
    )
