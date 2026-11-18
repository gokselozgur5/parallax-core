import numpy as np

from parallax.dispatch import fused_masked_softmax_backward


def test_masked_softmax_backward_mask_polarity_is_dtype_independent():
    probs = np.array([[0.1, 0.2, 0.3, 0.4]])
    dout = np.array([[1.0, -2.0, 3.0, 0.5]])
    mask = np.array([[True, False, False, True]])
    from_bool = fused_masked_softmax_backward(dout, probs, mask)
    from_float = fused_masked_softmax_backward(dout, probs, mask.astype(np.float64))
    np.testing.assert_allclose(from_bool, from_float)
