"""Unit tests for DispatchedEmbedding module."""

import numpy as np
import pytest
from parallax.nn import Parameter
from parallax.tensor import Tensor


def test_dispatched_embedding_init():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(num_embeddings=10, embedding_dim=4)
    assert emb.num_embeddings == 10
    assert emb.embedding_dim == 4
    assert emb.weight.shape == (10, 4)
    assert isinstance(emb.weight, Parameter)


def test_dispatched_embedding_forward_shape():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(10, 4)
    idx = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    out = emb(idx)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(out.data[0, 0], emb.weight.data[1])


def test_dispatched_embedding_padding_idx():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(10, 4, padding_idx=0)
    np.testing.assert_allclose(emb.weight.data[0], np.zeros(4))
    idx = Tensor(np.array([0, 1]))
    out = emb(idx)
    np.testing.assert_allclose(out.data[0], np.zeros(4))


def test_dispatched_embedding_bounds_validation():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(5, 3)
    with pytest.raises(IndexError):
        emb(Tensor(np.array([5])))
    with pytest.raises(IndexError):
        emb(Tensor(np.array([-1])))


def test_dispatched_embedding_backward_grad():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(5, 2, padding_idx=0)
    idx = Tensor(np.array([1, 1, 2]))
    out = emb(idx)
    loss = out.sum()
    loss.backward()
    assert emb.weight.grad is not None
    np.testing.assert_allclose(emb.weight.grad[1], [2.0, 2.0])
    np.testing.assert_allclose(emb.weight.grad[0], [0.0, 0.0])


def test_dispatched_embedding_from_pretrained_and_repr():
    from parallax.nn.embedding import DispatchedEmbedding
    weights = np.ones((4, 3))
    emb = DispatchedEmbedding.from_pretrained(weights, freeze=True)
    assert not emb.weight.requires_grad
    assert "num_embeddings=4" in repr(emb)
    assert "embedding_dim=3" in repr(emb)


def test_dispatched_embedding_package_export():
    import parallax.nn as pnn
    assert hasattr(pnn, "DispatchedEmbedding")
    assert "DispatchedEmbedding" in pnn.__all__


def test_dispatched_embedding_routes_through_backend_gather(monkeypatch):
    from parallax.backends import registry
    from parallax.backends.base import Backend
    from parallax.backends.dispatch import clear_fallback_history, get_fallback_history
    from parallax.backends.numpy_backend import NumpyBackend
    from parallax.nn.embedding import DispatchedEmbedding

    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class _GatherBackend(Backend):
        name = "c"

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: object, shape: tuple[int, ...]) -> bool:
            return op == "gather" and np.dtype(dtype) == np.float32  # type: ignore[arg-type]

        def gather(self, weight: np.ndarray, indices: np.ndarray) -> np.ndarray:
            seen.append((weight.shape, indices.shape))
            return weight[indices]

    weights = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    emb = DispatchedEmbedding.from_pretrained(weights, freeze=False, padding_idx=0)
    registry.clear()
    clear_fallback_history()
    registry.register("numpy", NumpyBackend())
    registry.register("c", _GatherBackend())
    try:
        monkeypatch.setenv("PARALLAX_BACKEND", "c")
        out = emb(Tensor(np.array([1, 3])))
        out.sum().backward()
        monkeypatch.delenv("PARALLAX_BACKEND", raising=False)
        registry.unregister("c")
        fallback_out = emb(Tensor(np.array([2])))
        history = get_fallback_history()
    finally:
        registry.clear()
        clear_fallback_history()
    assert seen == [((4, 2), (2,))]
    assert history == []
    np.testing.assert_allclose(out.data, [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(fallback_out.data, [[3.0, 4.0]])
    assert emb.weight.grad is not None
    np.testing.assert_allclose(emb.weight.grad[1], [1.0, 1.0])


def test_dispatched_gather_validation_and_fallback(monkeypatch):
    from parallax.backends import registry
    from parallax.backends.base import Backend
    from parallax.backends.dispatch import clear_fallback_history, get_fallback_history
    from parallax.backends.numpy_backend import NumpyBackend
    from parallax.backends.scope import use_backend
    from parallax.backends.tensor_ops import dispatched_gather
    from parallax.nn.embedding import DispatchedEmbedding

    with pytest.raises(TypeError, match="Tensor weight"):
        dispatched_gather(np.ones((4, 2)), np.array([0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2D weight"):
        dispatched_gather(Tensor(np.ones((4,))), np.array([0]))

    class _FaultyGather(Backend):
        name = "rust"

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: object, shape: tuple[int, ...]) -> bool:
            return op == "gather"

        def gather(self, weight: np.ndarray, indices: np.ndarray) -> np.ndarray:
            raise RuntimeError("gather kernel failure")

    emb = DispatchedEmbedding.from_pretrained(np.array([[1.0, 2.0], [3.0, 4.0]]), freeze=True)
    registry.clear()
    clear_fallback_history()
    registry.register("numpy", NumpyBackend())
    registry.register("rust", _FaultyGather())
    try:
        with use_backend("rust"):
            out = emb(np.array([1]))
        history = get_fallback_history()
        monkeypatch.setenv("PARALLAX_BACKEND", "invalid_backend")
        with pytest.raises(ValueError, match="not a known backend"):
            emb(np.array([0]))
    finally:
        registry.clear()
        clear_fallback_history()
    np.testing.assert_allclose(out.data, [[3.0, 4.0]])
    assert len(history) == 1 and "gather kernel failure" in history[0].reason



