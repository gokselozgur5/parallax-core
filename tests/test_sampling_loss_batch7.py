import numpy as np
import pytest

from parallax.contrastive_decoding import ContrastiveDecoder
from parallax.flash_loss import flash_cross_entropy_forward
from parallax.generator import sample_multinomial, sample_next_token
from parallax.loss import cross_entropy_loss
from parallax.paged_attention import BlockAllocator, SequenceManager
from parallax.tensor import Tensor
from parallax.top_a import TopASampler


def test_cross_entropy_loss_target_out_of_bounds_validation():
    # Issue #296: validate targets outside [0, V)
    logits = Tensor(np.array([[0.0, 0.0, 5.0]], dtype=np.float32))

    # Target -1 should raise ValueError instead of silently indexing class 2
    with pytest.raises(ValueError, match=r"Target index -1 out of bounds for vocab_size 3"):
        cross_entropy_loss(logits, np.array([-1]))

    # Target >= vocab_size should raise ValueError
    with pytest.raises(ValueError, match=r"Target index 3 out of bounds for vocab_size 3"):
        cross_entropy_loss(logits, np.array([3]))

    # Target == ignore_index should be safely ignored
    loss_ignored = cross_entropy_loss(logits, np.array([-100]), ignore_index=-100)
    assert loss_ignored.data == 0.0

    # Negative ignore_index other than -100
    loss_custom_ignore = cross_entropy_loss(logits, np.array([-1]), ignore_index=-1)
    assert loss_custom_ignore.data == 0.0


def test_flash_cross_entropy_float64_large_scale_precision():
    # Issue #309: keep target logits in float64 during online cross entropy
    logits = np.array([[1000.1234567, 0.0, 0.0]])
    targets = np.array([0])
    loss, m_vec, l_vec = flash_cross_entropy_forward(logits, targets, chunk_size=2)
    assert abs(loss) < 1e-12


def test_sample_multinomial_temperature_zero_greedy_argmax():
    # Issue #319: temperature 0 should be argmax
    logits = np.array([1.0, 1.0 - 1e-6, -5.0])
    tok = sample_multinomial(logits, temperature=0.0)
    assert tok == 0

    tok_subnormal = sample_multinomial(logits, temperature=1e-8)
    assert tok_subnormal == 0


def test_top_a_and_contrastive_temperature_zero_argmax():
    # Issue #319: top_a and contrastive_decoding with temperature <= 0
    top_a = TopASampler()
    logits = np.array([1.0, 1.0 - 1e-6, -5.0])
    for _ in range(50):
        assert top_a.sample(logits, temperature=0.0) == 0

    cd = ContrastiveDecoder()
    exp_logits = np.array([1.0, 1.0 - 1e-6, -5.0])
    amat_logits = np.array([0.0, 0.0, 0.0])
    for _ in range(50):
        assert cd.sample_next_token(exp_logits, amat_logits, temperature=0.0) == 0


def test_top_a_safeguard_alpha_greater_than_one():
    # Issue #320: alpha > 1 always keeps at least top candidate token
    sampler = TopASampler(alpha=2.0)
    logits = np.array([6.0, 0.0, 0.0, 0.0, -5.0])
    filtered = sampler.filter_logits(logits)
    assert filtered[0] == 6.0
    assert np.all(filtered[1:] == -1e9)

    for _ in range(50):
        assert sampler.sample(logits, temperature=1.0) == 0


def test_paged_attention_block_validation_and_sequence_release():
    # Issue #311: validate block ids and release sequences
    alloc = BlockAllocator(num_blocks=2, block_size=2, head_dim=4)

    # Validate block id bounds
    with pytest.raises(ValueError, match=r"Block 7 out of bounds for num_blocks 2"):
        alloc.free_block(7)

    with pytest.raises(ValueError, match=r"Block -1 out of bounds for num_blocks 2"):
        alloc.free_block(-1)

    mgr = SequenceManager(alloc)
    for _ in range(4):
        mgr.append_token(seq_id=10, k_vec=np.zeros(4), v_vec=np.zeros(4))

    assert len(alloc.free_blocks) == 0
    assert len(mgr.block_tables[10]) == 2

    # Release sequence 10
    mgr.release(10)
    assert 10 not in mgr.block_tables
    assert 10 not in mgr.seq_lengths
    assert len(alloc.free_blocks) == 2

    # Re-allocating blocks works after release
    b0 = alloc.allocate_block()
    b1 = alloc.allocate_block()
    assert {b0, b1} == {0, 1}
