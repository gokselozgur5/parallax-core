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


def test_cuda_engine_forward_step_cached_multi_token_mask(tmp_path):
    """Cover chunked prefill / multi-token continuation causal mask."""
    torch = pytest.importorskip("torch")
    from parallax.cuda_engine import Qwen2CUDAModel
    from tests.test_cuda_qwen3_gguf import _write_tiny_qwen3_gguf

    gguf_path = str(tmp_path / "tiny_qwen3.gguf")
    _write_tiny_qwen3_gguf(gguf_path)
    model = Qwen2CUDAModel(gguf_path, device="cpu", dtype=torch.float32)

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long)
    full_logits, _ = model.forward_step(ids[:, :12], start_pos=0)
    _, kv = model.forward_step(ids[:, :8], start_pos=0)
    chunk_logits, _ = model.forward_step(ids[:, 8:12], start_pos=8, kv_cache=kv)

    diff = (full_logits[0] - chunk_logits[0]).abs().max().item()
    assert diff < 1e-2


