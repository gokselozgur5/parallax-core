from parallax.unified_norm import UnifiedNorm


def test_unified_norm_initialization() -> None:
    norm_rms = UnifiedNorm(dim=32, norm_type="rmsnorm")
    assert norm_rms.dim == 32
    assert norm_rms.bias is None

    norm_ln = UnifiedNorm(dim=32, norm_type="layernorm")
    assert norm_ln.bias is not None
    assert norm_ln.bias.shape == (32,)
