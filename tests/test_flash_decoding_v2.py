import numpy as np
import pytest

from parallax.flash_decoding_v2 import (
    select_split_k,
    reduce_split_k_online,
    flash_decode_split_k_v2,
    flash_decode_split_k_paged,
)


def standard_attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    """Exact full standard attention for a single query token."""
    q_vec = q.squeeze()
    logits = np.dot(k, q_vec) * scale
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    return np.dot(probs, v)


def test_select_split_k_dynamic():
    """Verify dynamic Split-K partition factor scaling."""
    min_chunk = 64
    assert select_split_k(0, min_chunk_size=min_chunk) == 1
    assert select_split_k(32, min_chunk_size=min_chunk) == 1
    assert select_split_k(64, min_chunk_size=min_chunk) == 1
    assert select_split_k(100, min_chunk_size=min_chunk) == 1
    assert select_split_k(128, min_chunk_size=min_chunk) == 2
    assert select_split_k(192, min_chunk_size=min_chunk) == 2
    assert select_split_k(200, min_chunk_size=min_chunk) == 2
    assert select_split_k(256, min_chunk_size=min_chunk) == 4
    assert select_split_k(512, min_chunk_size=min_chunk) == 8
    assert select_split_k(1024, min_chunk_size=min_chunk) == 8
    # Custom max_splits clamp
    assert select_split_k(1024, min_chunk_size=min_chunk, max_splits=4) == 4
    # Support max_splits > 8 when sequence length permits
    assert select_split_k(1024, min_chunk_size=min_chunk, max_splits=16) == 16
    assert select_split_k(2048, min_chunk_size=min_chunk, max_splits=32) == 32
    # Invariant: split_k * min_chunk_size <= seq_len
    for s in [1, 50, 64, 127, 128, 192, 255, 256, 500, 1024]:
        k = select_split_k(s, min_chunk_size=64, max_splits=16)
        if s >= 64:
            assert k * 64 <= s


def test_reduce_split_k_online_numerical_stability():
    """Verify numerical stability of reduce_split_k_online even with large magnitude logits."""
    np.random.seed(42)
    d = 16
    num_splits = 4
    local_outs = np.random.randn(num_splits, d)
    # Huge logits that would overflow naive exp
    local_maxes = np.array([1000.0, 1005.0, 990.0, 1004.5])
    local_lsums = np.array([10.0, 15.0, 8.0, 12.0])

    out, lse = reduce_split_k_online(local_outs, local_maxes, local_lsums)

    assert out.shape == (d,)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.isfinite(lse)
    assert lse > 1005.0  # Logsumexp must be >= max


@pytest.mark.parametrize("seq_len", [16, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("d", [32, 64])
def test_flash_decode_split_k_v2_numeric_equivalence(seq_len: int, d: int):
    """Verify FlashDecoding-v2 matches standard attention reference with atol <= 1e-5."""
    np.random.seed(42 + seq_len)
    q = np.random.randn(d)
    k = np.random.randn(seq_len, d)
    v = np.random.randn(seq_len, d)
    scale = 1.0 / np.sqrt(d)

    ref_out = standard_attention_reference(q, k, v, scale)

    # Test with automatic dynamic split
    fd2_auto = flash_decode_split_k_v2(q, k, v, k_splits=None, scale=scale)
    assert np.allclose(fd2_auto, ref_out, atol=1e-5, rtol=1e-5)

    # Test with explicit split values (2, 4, 8)
    for splits in [2, 4, 8]:
        fd2_split = flash_decode_split_k_v2(q, k, v, k_splits=splits, scale=scale)
        assert np.allclose(fd2_split, ref_out, atol=1e-5, rtol=1e-5)


def test_flash_decode_split_k_paged():
    """Verify flash_decode_split_k_paged produces identical outputs to reference over paged memory."""
    np.random.seed(1337)
    d = 32
    block_size = 16
    num_blocks = 10
    seq_len = 70  # Spans 5 blocks (4 full 16s + 6 tokens in the 5th)

    k_blocks = np.random.randn(num_blocks, block_size, d)
    v_blocks = np.random.randn(num_blocks, block_size, d)
    block_table = [3, 7, 1, 9, 2]
    q = np.random.randn(d)
    scale = 1.0 / np.sqrt(d)

    # Gather reference tensors
    k_full = np.concatenate([k_blocks[b] for b in block_table], axis=0)[:seq_len]
    v_full = np.concatenate([v_blocks[b] for b in block_table], axis=0)[:seq_len]
    ref_out = standard_attention_reference(q, k_full, v_full, scale)

    paged_out = flash_decode_split_k_paged(
        q=q,
        block_table=block_table,
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        seq_len=seq_len,
        scale=scale,
    )

    assert np.allclose(paged_out, ref_out, atol=1e-5, rtol=1e-5)


def test_flash_decode_empty_or_single_token():
    """Edge cases: empty sequence and single token sequence."""
    d = 16
    q = np.random.randn(d)
    k_empty = np.zeros((0, d))
    v_empty = np.zeros((0, d))

    out_empty = flash_decode_split_k_v2(q, k_empty, v_empty)
    assert np.all(out_empty == 0.0)

    # Single token
    k_single = np.random.randn(1, d)
    v_single = np.random.randn(1, d)
    out_single = flash_decode_split_k_v2(q, k_single, v_single)
    # With 1 token, softmax probability is 1.0, so output must equal v[0]
    assert np.allclose(out_single, v_single[0], atol=1e-5)
