from parallax.unified_norm import UnifiedNorm


def test_unified_norm_initialization() -> None:
    norm_rms = UnifiedNorm(dim=32, norm_type="rmsnorm")
    assert norm_rms.dim == 32
    assert norm_rms.bias is None

    norm_ln = UnifiedNorm(dim=32, norm_type="layernorm")
    assert norm_ln.bias is not None
    assert norm_ln.bias.shape == (32,)

import numpy as np


def test_unified_norm_fused_residual_rmsnorm() -> None:
    norm = UnifiedNorm(dim=4, norm_type="rmsnorm")
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    res = np.array([[1.0, 1.0, 1.0, 1.0]])
    normed, fused = norm.forward(x, residual=res)
    assert np.allclose(fused, [[2.0, 3.0, 4.0, 5.0]])
    assert normed.shape == (1, 4)
    assert np.all(np.isfinite(normed))
