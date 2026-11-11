import sys
import pytest


def test_cuda_engine_import_without_torch(monkeypatch):
    """Test importing parallax.cuda_engine when torch is not available."""
    # Ensure torch cannot be imported
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", None)

    # Remove parallax.cuda_engine if cached
    sys.modules.pop("parallax.cuda_engine", None)

    import parallax.cuda_engine as ce
    assert ce.is_cuda_available() is False

    # Restore module state
    sys.modules.pop("parallax.cuda_engine", None)


def test_cuda_engine_generate_stream_validation():
    """Verify input validation and tiny temperature fallback in Qwen2CUDAModel."""
    torch = pytest.importorskip("torch")
    from parallax.cuda_engine import Qwen2CUDAModel

    model = object.__new__(Qwen2CUDAModel)
    model.device = torch.device("cpu")
    model.vocab_size = 100
    model.forward_step = lambda input_ids, start_pos, kv_cache: (torch.zeros((1, 100)), None)

    with pytest.raises(ValueError, match="prompt_tokens must not be empty"):
        list(model.generate_stream([]))

    with pytest.raises(ValueError, match="out of vocab range"):
        list(model.generate_stream([-1]))

    with pytest.raises(ValueError, match="out of vocab range"):
        list(model.generate_stream([100]))

    with pytest.raises(ValueError, match="repetition_penalty must be positive"):
        list(model.generate_stream([1], repetition_penalty=0.0))

    with pytest.raises(ValueError, match="max_new_tokens must be non-negative"):
        list(model.generate_stream([1], max_new_tokens=-1))

    assert list(model.generate_stream([1], max_new_tokens=0)) == []

    # Tiny temperature < 1e-5 should fall back to greedy without division by zero / overflow
    tokens = list(model.generate_stream([1], max_new_tokens=2, temperature=1e-8))
    assert len(tokens) == 2

