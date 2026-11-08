import numpy as np
from parallax.mla import MultiHeadLatentAttention


def test_mla_initialization_attributes() -> None:
    mla = MultiHeadLatentAttention(embed_dim=64, num_heads=4, latent_dim=16)
    assert mla.embed_dim == 64
    assert mla.num_heads == 4
    assert mla.latent_dim == 16
    assert mla.head_dim == 16
