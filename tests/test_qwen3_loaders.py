"""End-to-end HuggingFace SafeTensors and GGUF loader parity tests for Qwen3."""

import json
from pathlib import Path
import numpy as np

from parallax.gguf import save_gguf
from parallax.gguf_loader import load_gguf_pretrained
from parallax.hf_loader import load_hf_pretrained
from parallax.safetensors import save_safetensors


def _make_qwen3_weights() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(9)
    hf: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.standard_normal((32, 16)).astype(np.float32) * 0.1,
        "model.norm.weight": np.ones(16, dtype=np.float32),
        "lm_head.weight": rng.standard_normal((32, 16)).astype(np.float32) * 0.1,
    }
    gg: dict[str, np.ndarray] = {
        "token_embd.weight": hf["model.embed_tokens.weight"],
        "output_norm.weight": hf["model.norm.weight"],
        "output.weight": hf["lm_head.weight"],
    }
    return hf, gg


def _populate_layer0(hf: dict[str, np.ndarray], gg: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(11)
    pairs = [
        ("input_layernorm.weight", "attn_norm.weight", (16,)),
        ("post_attention_layernorm.weight", "ffn_norm.weight", (16,)),
        ("self_attn.q_proj.weight", "attn_q.weight", (24, 16)),
        ("self_attn.k_proj.weight", "attn_k.weight", (12, 16)),
        ("self_attn.v_proj.weight", "attn_v.weight", (12, 16)),
        ("self_attn.o_proj.weight", "attn_output.weight", (16, 24)),
        ("self_attn.q_norm.weight", "attn_q_norm.weight", (6,)),
        ("self_attn.k_norm.weight", "attn_k_norm.weight", (6,)),
        ("mlp.gate_proj.weight", "ffn_gate.weight", (32, 16)),
        ("mlp.up_proj.weight", "ffn_up.weight", (32, 16)),
        ("mlp.down_proj.weight", "ffn_down.weight", (16, 32)),
    ]
    for hf_k, gg_k, shp in pairs:
        hf[f"model.layers.0.{hf_k}"] = gg[f"blk.0.{gg_k}"] = rng.standard_normal(shp).astype(np.float32) * 0.1


def test_qwen3_hf_and_gguf_loaders_match_logits(tmp_path: Path) -> None:
    hf, gg = _make_qwen3_weights()
    _populate_layer0(hf, gg)
    hf_dir = tmp_path / "hf_q3"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "hidden_size": 16, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 6, "num_hidden_layers": 1,
        "intermediate_size": 32, "vocab_size": 32, "tie_word_embeddings": False,
    }))
    save_safetensors(hf, str(hf_dir / "model.safetensors"))
    gg_path = str(tmp_path / "q3.gguf")
    save_gguf(gg_path, {
        "general.architecture": "qwen3", "qwen3.embedding_length": 16, "qwen3.block_count": 1,
        "qwen3.attention.head_count": 4, "qwen3.attention.head_count_kv": 2,
        "qwen3.attention.key_length": 6, "qwen3.feed_forward_length": 32, "general.vocab_size": 32,
    }, gg)
    ids = np.array([[1, 4, 7]], dtype=np.int64)
    np.testing.assert_allclose(load_hf_pretrained(str(hf_dir))(ids).data, load_gguf_pretrained(gg_path)(ids).data, atol=1e-6)
