import numpy as np
import pytest

from parallax.native.avx2_rmsnorm import is_avx2_rmsnorm_available, rmsnorm_avx2


def test_avx2_rmsnorm_availability_probe():
    assert isinstance(is_avx2_rmsnorm_available(), bool)


def test_rmsnorm_avx2_shape_validation():
    x = np.ones((2, 8), dtype=np.float32)
    w_bad = np.ones((7,), dtype=np.float32)
    if not is_avx2_rmsnorm_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            rmsnorm_avx2(x, w_bad)
    else:
        with pytest.raises(ValueError, match="shape mismatch"):
            rmsnorm_avx2(x, w_bad)
