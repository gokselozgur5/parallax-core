import numpy as np
from parallax.mla import MultiHeadLatentAttention


def test_mla_initialization_attributes() -> None:
    mla = MultiHeadLatentAttention(embed_dim=64, num_heads=4, latent_dim=16)
    assert mla.embed_dim == 64
    assert mla.num_heads == 4
    assert mla.latent_dim == 16
    assert mla.head_dim == 16

def test_mla_compression_and_forward() -> None:
    np.random.seed(42)
    seq_len, embed_dim, latent_dim = 8, 32, 8
    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim, num_heads=2, latent_dim=latent_dim
    )
    x = np.random.randn(seq_len, embed_dim)
    c_kv = mla.compress_kv(x)
    assert c_kv.shape == (seq_len, latent_dim)

    k, v = mla.decompress_kv(c_kv)
    assert k.shape == (seq_len, embed_dim)
    assert v.shape == (seq_len, embed_dim)

    out = mla.forward(x, causal=True)
    assert out.shape == (seq_len, embed_dim)
    assert np.all(np.isfinite(out))


def test_mla_batched_forward() -> None:
    np.random.seed(42)
    batch_size, seq_len, embed_dim, latent_dim = 2, 6, 32, 8
    mla = MultiHeadLatentAttention(
        embed_dim=embed_dim, num_heads=4, latent_dim=latent_dim, rope_dim=8
    )
    x = np.random.randn(batch_size, seq_len, embed_dim)
    out = mla.forward(x, causal=True)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert np.all(np.isfinite(out))
