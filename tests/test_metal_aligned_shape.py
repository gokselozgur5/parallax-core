import numpy as np
from parallax.metal.aligned import aligned_zeros

def test_aligned_zeros_returns_requested_shape() -> None:
    arr = aligned_zeros((3, 5), dtype=np.float32)
    assert arr.shape == (3, 5)
