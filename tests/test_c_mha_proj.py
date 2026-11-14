from __future__ import annotations

import numpy as np

from parallax.native.mha_proj import (
    c_mha_split_qkv,
    is_mha_proj_available,
)


def test_mha_proj_available() -> None:
    assert is_mha_proj_available()


def test_mha_split_qkv_parity() -> None:
    rng = np.random.default_rng(42)
    b, t, hq, hkv, d = 2, 4, 4, 2, 8
    fused_dim = (hq + 2 * hkv) * d
    qkv = rng.standard_normal((b, t, fused_dim), dtype=np.float32)
    q, k, v = c_mha_split_qkv(qkv, hq, hkv, d)
    q_exp = qkv[:, :, : hq * d].reshape(b, t, hq, d).transpose(0, 2, 1, 3)
    k_exp = qkv[:, :, hq * d : (hq + hkv) * d].reshape(b, t, hkv, d).transpose(0, 2, 1, 3)
    v_exp = qkv[:, :, (hq + hkv) * d :].reshape(b, t, hkv, d).transpose(0, 2, 1, 3)
    np.testing.assert_allclose(q, q_exp)
    np.testing.assert_allclose(k, k_exp)
    np.testing.assert_allclose(v, v_exp)
