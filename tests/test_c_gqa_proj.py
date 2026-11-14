from __future__ import annotations

import numpy as np
import pytest

from parallax.native.gqa_proj import c_gqa_proj, is_gqa_proj_available


def test_available() -> None:
    assert is_gqa_proj_available()


def test_parity_and_replication() -> None:
    rng = np.random.default_rng(0)
    rows, dim, nh, nkv, hd = 5, 16, 8, 2, 4
    x = rng.standard_normal((rows, dim)).astype(np.float32)
    wq = rng.standard_normal((dim, nh * hd)).astype(np.float32)
    wk = rng.standard_normal((dim, nkv * hd)).astype(np.float32)
    wv = rng.standard_normal((dim, nkv * hd)).astype(np.float32)
    q, k, v = c_gqa_proj(x, wq, wk, wv, nh, nkv, hd)
    np.testing.assert_allclose(q, x @ wq, rtol=1e-4, atol=1e-4)
    for out, w in ((k, wk), (v, wv)):
        ref = np.repeat((x @ w).reshape(rows, nkv, hd), nh // nkv, axis=1)
        np.testing.assert_allclose(out.reshape(rows, nh, hd), ref, rtol=1e-4, atol=1e-4)


def test_invalid_heads() -> None:
    x = np.ones((2, 4), dtype=np.float32)
    w = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        c_gqa_proj(x, w, w, w, 3, 2, 2)
