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
