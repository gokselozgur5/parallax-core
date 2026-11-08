import numpy as np
from parallax.ring_attention import RingAttentionContext


def test_ring_attention_context_init() -> None:
    ctx = RingAttentionContext(block_size=32)
    assert ctx.block_size == 32
