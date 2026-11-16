"""Unit tests for GQA head repeat dispatch helpers."""

import numpy as np
import pytest
from python.parallax.dispatch.gqa import repeat_kv


def test_repeat_kv_shape_and_values() -> None:
    x = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
    # Shape: (1, 2, 1, 2) -> (batch=1, num_kv_heads=2, seq_len=1, head_dim=2)
    out = repeat_kv(x, n_rep=3)
    assert out.shape == (1, 6, 1, 2)
    np.testing.assert_allclose(out[:, 0:3, :, :], np.repeat(x[:, 0:1, :, :], 3, axis=1))
    np.testing.assert_allclose(out[:, 3:6, :, :], np.repeat(x[:, 1:2, :, :], 3, axis=1))
