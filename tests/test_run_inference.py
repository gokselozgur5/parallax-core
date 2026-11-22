import numpy as np
import pytest

import parallax.gguf_loader as gguf_loader
from parallax.gguf import save_gguf


def test_gguf_dims_are_validated_before_model_allocation(tmp_path, monkeypatch):
    path = str(tmp_path / "tiny.gguf")
    metadata = {
        "general.architecture": "llama",
        "llama.embedding_length": 2**40,
        "llama.block_count": 1,
        "llama.attention.head_count": 1,
        "tokenizer.ggml.tokens": ["a"],
    }
    save_gguf(path, metadata, {"token_embd.weight": np.zeros((1, 4), dtype=np.float32)})

    built = []

    def fake_model(config):
        built.append(config.embed_dim)
        raise MemoryError("stand-in for the 8 TiB embedding allocation")

    monkeypatch.setattr(gguf_loader, "ParallaxLM", fake_model)
    with pytest.raises(ValueError):
        gguf_loader.load_gguf_pretrained(path)
    assert built == []
