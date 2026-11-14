import numpy as np
import pytest
from parallax.native.flashattn_avx512 import (
    _get_flashattn_avx512_fn,
    is_avx512_flashattn_available,
    flashattn_avx512,
)


def test_flashattn_avx512_symbols_bound():
    fn = _get_flashattn_avx512_fn()
    assert fn is not None
    assert isinstance(is_avx512_flashattn_available(), bool)
