import numpy as np
import pytest
from parallax.bitnet import BitLinear
from parallax.muon import Muon
from parallax.qgalore import QGaLoreProjector
from parallax.tensor import Tensor


def test_qgalore_rank_larger_than_min_mn():
    M, N = 8, 16
    rank = 32  # rank > min(M, N) == 8
    proj = QGaLoreProjector(rank=rank, update_interval=10)

    grad = np.random.randn(M, N).astype(np.float32)
    q_subspace, scale = proj.project_and_quantize(grad)

    expected_rank = min(M, N)
    assert q_subspace.shape == (expected_rank, N)
    assert proj.U.shape == (M, expected_rank)
    assert proj.V_T.shape == (expected_rank, N)

    recon = proj.deproject(q_subspace, scale)
    assert recon.shape == (M, N)


def test_bitnet_bias_in_autograd_graph_receives_grad():
    np.random.seed(42)
    in_f, out_f = 8, 4
    layer = BitLinear(in_features=in_f, out_features=out_f, bias=True)

    # Bias-only training: freeze input and weights
    x = Tensor(np.random.randn(2, in_f), requires_grad=False)
    layer.weight.requires_grad = False
    layer.bias.requires_grad = True

    out = layer(x)
    assert layer.bias in out._prev

    loss = out.sum()
    loss.backward()

    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (out_f,)
    assert np.allclose(layer.bias.grad, np.array([2.0, 2.0, 2.0, 2.0]))
    assert layer.weight.grad is None


def test_muon_step_mismatched_params_and_grads_raises_value_error():
    p1 = np.ones((4, 4), dtype=np.float32)
    p2 = np.ones((4, 4), dtype=np.float32)
    optimizer = Muon([p1, p2], lr=0.01)

    with pytest.raises(
        ValueError, match=r"Mismatched params \(2\) and grads \(1\)"
    ):
        optimizer.step([np.zeros((4, 4), dtype=np.float32)])

    with pytest.raises(
        ValueError, match=r"Mismatched params \(2\) and grads \(3\)"
    ):
        optimizer.step(
            [
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.float32),
            ]
        )
