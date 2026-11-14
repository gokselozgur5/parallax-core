from __future__ import annotations

import pytest
from parallax.native.gelu import c_gelu, gelu_numpy


import numpy as np


def test_gelu_parity_small() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((32, 64), dtype=np.float32)

    res = c_gelu(x)
    ref = gelu_numpy(x)
    max_err = float(np.max(np.abs(res - ref)))
    assert max_err < 1e-6
