import numpy as np
from parallax.ring_attention import RingAttentionContext


def test_ring_attention_context_init() -> None:
    ctx = RingAttentionContext(block_size=32)
    assert ctx.block_size == 32

from parallax.ring_attention import ring_flash_update


def test_ring_flash_update_rescale() -> None:
    acc_o = np.ones((2, 4))
    acc_m = np.zeros(2)
    acc_l = np.ones(2)
    block_o = np.full((2, 4), 2.0)
    block_m = np.ones(2)
    block_l = np.ones(2)
    merged_o, merged_m, merged_l = ring_flash_update(
        acc_o, acc_m, acc_l, block_o, block_m, block_l
    )
    assert merged_o.shape == (2, 4)
    assert np.all(merged_m == 1.0)
