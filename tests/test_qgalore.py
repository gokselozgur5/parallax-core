import numpy as np
from parallax.qgalore import QGaLoreProjector


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

