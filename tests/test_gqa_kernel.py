import numpy as np
from parallax.gqa_kernel import repeat_kv_heads


def test_repeat_kv_heads_broadcasting():
    b, h_kv, s, d = 2, 2, 4, 8
    num_groups = 3
    x = np.random.randn(b, h_kv, s, d).astype(np.float32)

    repeated = repeat_kv_heads(x, num_groups=num_groups)

    assert repeated.shape == (b, 6, s, d)
    # Group 0: heads 0, 1, 2 must equal original head 0
    for g in range(num_groups):
        assert np.allclose(repeated[:, g, :, :], x[:, 0, :, :])
    # Group 1: heads 3, 4, 5 must equal original head 1
    for g in range(num_groups):
        assert np.allclose(repeated[:, num_groups + g, :, :], x[:, 1, :, :])


from parallax.gqa_kernel import reduce_gqa_kv_gradients


def test_reduce_gqa_kv_gradients_accumulation():
    b, h_kv, s, d = 2, 2, 4, 8
    num_groups = 3
    # Gradient of ones across 6 repeated heads
    grad_rep = np.ones((b, 6, s, d), dtype=np.float32)

    grad_kv = reduce_gqa_kv_gradients(grad_rep, num_groups=num_groups)

    assert grad_kv.shape == (b, h_kv, s, d)
    # Summing across 3 heads should yield exact value 3.0
    assert np.allclose(grad_kv, 3.0)
