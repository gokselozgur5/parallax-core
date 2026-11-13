from __future__ import annotations

import numpy as np

from parallax.native.swiglu import c_swiglu, swiglu_numpy


def test_swiglu_parity_small() -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((32, 64), dtype=np.float32)
    b = rng.standard_normal((32, 64), dtype=np.float32)

    res = c_swiglu(a, b)
    ref = swiglu_numpy(a, b)
    max_err = float(np.max(np.abs(res - ref)))
    assert max_err < 1e-6
