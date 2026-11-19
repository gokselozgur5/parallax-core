from __future__ import annotations

import numpy as np

from parallax.native.swiglu import swiglu_numpy


def test_swiglu_numpy_propagates_nan_inputs() -> None:
    a = np.array([np.nan, 1.0], dtype=np.float32)
    b = np.array([1.0, np.nan], dtype=np.float32)
    out = swiglu_numpy(a, b)
    assert np.isnan(out).all()
