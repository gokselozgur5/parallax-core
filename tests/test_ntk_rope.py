from parallax.ntk_rope import NTKAwareRoPE


def test_ntk_rope_initialization() -> None:
    rope = NTKAwareRoPE(dim=64, base=10000.0, alpha=2.0)
    assert rope.dim == 64
    assert rope.base == 10000.0
    assert rope.alpha == 2.0
