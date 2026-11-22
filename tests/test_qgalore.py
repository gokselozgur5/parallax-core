import numpy as np
from parallax.nn import Parameter
from parallax.qgalore import QGaLoreAdamW, QGaLoreProjector


def test_qgalore_projector_init():
    proj = QGaLoreProjector(rank=4, update_interval=50)
    assert proj.rank == 4
    assert proj.effective_rank == 4
    assert proj.update_interval == 50
    assert proj.step_count == 0
    assert proj.U is None


def test_qgalore_quantization_and_reconstruction():
    np.random.seed(42)
    M, N, rank = 32, 32, 4
    proj = QGaLoreProjector(rank=rank, update_interval=10)

    # Low rank gradient
    A = np.random.randn(M, rank).astype(np.float32)
    B = np.random.randn(rank, N).astype(np.float32)
    grad = np.dot(A, B)

    q_subspace, scale = proj.project_and_quantize(grad)

    assert q_subspace.shape == (M, rank)
    assert q_subspace.dtype == np.int8
    assert np.all(q_subspace >= -128) and np.all(q_subspace <= 127)

    recon = proj.deproject(q_subspace, scale)
    assert recon.shape == grad.shape
    # Low rank input should be accurately reconstructed by rank SVD
    rel_error = np.linalg.norm(recon - grad) / np.linalg.norm(grad)
    assert rel_error < 0.1


def test_qgalore_rank_exceeds_dimension():
    import pytest

    proj = QGaLoreProjector(rank=16)
    with pytest.warns(UserWarning, match=r"rank \(16\) exceeds min\(M, N\) \(8\)"):
        proj.update_basis(np.random.randn(8, 12))
    assert proj.effective_rank == 8
    assert proj.U.shape == (8, 8)


def test_qgalore_left_projection_when_m_less_than_n():
    np.random.seed(42)
    M, N, rank = 16, 32, 4  # M < N -> left-sided projection
    proj = QGaLoreProjector(rank=rank, update_interval=10)

    A = np.random.randn(M, rank).astype(np.float32)
    B = np.random.randn(rank, N).astype(np.float32)
    grad = np.dot(A, B)

    q_subspace, scale = proj.project_and_quantize(grad)

    assert q_subspace.shape == (rank, N)
    recon = proj.deproject(q_subspace, scale)
    assert recon.shape == (M, N)
    rel_error = np.linalg.norm(recon - grad) / np.linalg.norm(grad)
    assert rel_error < 0.1


def test_qgalore_adamw_moments_are_smaller_than_dense_adamw_moments():
    rng = np.random.default_rng(0)
    weight = Parameter(rng.standard_normal((64, 64)))
    target = rng.standard_normal((64, 64))
    optimizer = QGaLoreAdamW([weight], lr=1e-2, weight_decay=0.0, rank=4)

    loss_before = 0.5 * float(np.sum((weight.data - target) ** 2))
    weight.grad = weight.data - target
    optimizer.step()
    loss_after = 0.5 * float(np.sum((weight.data - target) ** 2))

    moment_bytes = sum(a.nbytes for a in optimizer.m) + sum(a.nbytes for a in optimizer.v)
    assert moment_bytes <= 2 * 64 * 4 * 8
    assert moment_bytes < 2 * weight.data.nbytes
    assert loss_after < loss_before


def test_qgalore_adamw_left_projection_and_1d_bias_moments():
    rng = np.random.default_rng(1)
    weight = Parameter(rng.standard_normal((16, 32)))
    bias = Parameter(rng.standard_normal((32,)))
    optimizer = QGaLoreAdamW([weight, bias], lr=1e-2, weight_decay=0.0, rank=4)

    weight.grad = rng.standard_normal((16, 32))
    bias.grad = rng.standard_normal((32,))
    optimizer.step()

    assert optimizer.m[0].shape == (4, 32)
    assert optimizer.v[0].shape == (4, 32)
    assert optimizer.m[1].shape == (32,)
    assert optimizer.v[1].shape == (32,)


