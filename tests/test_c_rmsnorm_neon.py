import numpy as np
import pytest

from parallax.native.neon_rmsnorm import is_neon_rmsnorm_available, rmsnorm_neon


def test_neon_rmsnorm_availability_probe():
    assert isinstance(is_neon_rmsnorm_available(), bool)


def test_rmsnorm_neon_shape_validation():
    x = np.ones((2, 8), dtype=np.float32)
    w_bad = np.ones((7,), dtype=np.float32)
    if not is_neon_rmsnorm_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            rmsnorm_neon(x, w_bad)
    else:
        with pytest.raises(ValueError, match="shape mismatch"):
            rmsnorm_neon(x, w_bad)
