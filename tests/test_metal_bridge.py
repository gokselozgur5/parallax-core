from parallax.backends.metal_bridge import MetalTensorBridge
from parallax.tensor import Tensor
import numpy as np


def test_from_metal_buffer_dtype() -> None:
    bridge = MetalTensorBridge()
    t = Tensor([1.0, 2.0])
    buf, shape, dt = bridge.to_metal_buffer(t)
    assert buf is not None
    rec = bridge.from_metal_buffer(buf, shape, dt)
    assert rec is not None
    assert np.allclose(t.data, rec)
    bridge.release_buffer(buf)
