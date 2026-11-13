import numpy as np
import pytest

from parallax.backends.layout import ensure_f32_contiguous


def test_compliant_array_passes_through():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    out, copied = ensure_f32_contiguous(arr)
    assert out is arr
    assert copied is False


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
        np.asfortranarray(np.ones((3, 5), dtype=np.float32)),
        np.linspace(0.0, 1.0, 9).reshape(3, 3),
        np.arange(10, dtype=np.int32),
    ],
)
def test_non_compliant_arrays_are_copied(arr):
    out, copied = ensure_f32_contiguous(arr)
    assert copied is True
    assert out.dtype == np.float32 and out.flags.c_contiguous
    np.testing.assert_allclose(out, arr.astype(np.float32))
