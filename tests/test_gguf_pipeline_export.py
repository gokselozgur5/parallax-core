"""
Tests for GGUF Export & Quantization Pipeline (export_to_gguf_v3).

Verifies roundtrip export of GPT (ParallaxLM) and BitMamba models into GGUF v3
with F32, Q8_0, and Q4_0 formats, and validates tensor names, shapes, and values
via gguf_loader.
"""

import os
import tempfile
import numpy as np
import pytest

from parallax.bitmamba import BitMambaConfig, BitMambaLM
from parallax.gguf import GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, load_gguf
from parallax.gguf_exporter import export_to_gguf_v3
from parallax.gguf_loader import load_gguf_pretrained
from parallax.gguf_tokenizer import GGUFTokenizer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_export_gpt_f32_roundtrip():
    """Verify ParallaxLM exported as F32 preserves exact weights and loads with load_gguf_pretrained."""
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=64,
        tie_weights=False,
        use_swiglu=True,
    )
    model = ParallaxLM(cfg)

    tokens = [f"tok_{i}" for i in range(64)]
    tokens[0] = "<unk>"
    tokens[1] = "<s>"
    tokens[2] = "</s>"
    tokenizer = GGUFTokenizer(tokens=tokens, bos_token_id=1, eos_token_id=2)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        export_to_gguf_v3(model, tmp_path, quantization="f32", tokenizer=tokenizer)

        # 1. Inspect raw GGUF file
        gguf_file = load_gguf(tmp_path)
        assert gguf_file.version == 3
        assert gguf_file.metadata["general.architecture"] == "llama"
        assert gguf_file.metadata["general.file_type"] == 0  # LLAMA_FTYPE_ALL_F32
        assert "token_embd.weight" in gguf_file.tensors
        assert "output.weight" in gguf_file.tensors
        assert "blk.0.attn_q.weight" in gguf_file.tensors
        assert "blk.0.ffn_gate.weight" in gguf_file.tensors

        # Verify shapes
        assert gguf_file.tensor_infos["token_embd.weight"].shape == (64, 32)
        assert gguf_file.tensor_infos["output.weight"].shape == (64, 32)
        assert gguf_file.tensor_infos["blk.0.attn_q.weight"].shape == (32, 32)

        # 2. Load back via load_gguf_pretrained
        loaded_model, loaded_tok = load_gguf_pretrained(tmp_path, return_tokenizer=True)
        assert loaded_model.config.vocab_size == cfg.vocab_size
        assert loaded_model.config.embed_dim == cfg.embed_dim
        assert loaded_model.config.num_layers == cfg.num_layers
        assert loaded_tok.vocab_size == 64

        # 3. Exact tensor roundtrip check
        np.testing.assert_allclose(
            loaded_model.token_embeddings.weight.data,
            model.token_embeddings.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            loaded_model.blocks[0].attn.q_proj.weight.data,
            model.blocks[0].attn.q_proj.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            loaded_model.lm_head.weight.data,
            model.lm_head.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )

        # 4. Equivalent forward pass
        idx = np.array([[1, 5, 12, 30]], dtype=np.int64)
        np.testing.assert_allclose(
            loaded_model(idx).data,
            model(idx).data,
            atol=1e-5,
            rtol=1e-5,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_export_gpt_q8_0_roundtrip():
    """Verify ParallaxLM exported as Q8_0 quantizes 2D matrices and preserves shapes."""
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        max_seq_len=32,
        intermediate_size=64,
        tie_weights=True,
    )
    model = ParallaxLM(cfg)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        export_to_gguf_v3(model, tmp_path, quantization="q8_0")

        gguf_file = load_gguf(tmp_path)
        assert gguf_file.version == 3
        assert gguf_file.metadata["general.file_type"] == 7  # LLAMA_FTYPE_MOSTLY_Q8_0

        # Check tensor types in descriptors
        assert gguf_file.tensor_infos["token_embd.weight"].ggml_type == GGML_TYPE_Q8_0
        assert gguf_file.tensor_infos["blk.0.attn_q.weight"].ggml_type == GGML_TYPE_Q8_0
        # 1D RMSNorm should stay F32
        assert gguf_file.tensor_infos["output_norm.weight"].ggml_type == GGML_TYPE_F32

        # Check shapes
        assert gguf_file.tensors["token_embd.weight"].shape == (64, 32)
        assert gguf_file.tensors["blk.0.attn_q.weight"].shape == (32, 32)
        assert gguf_file.tensors["output_norm.weight"].shape == (32,)

        # Q8_0 precision check: max error < 0.05 on normalized weights
        w_orig = model.token_embeddings.weight.data
        w_dequant = gguf_file.tensors["token_embd.weight"]
        np.testing.assert_allclose(w_dequant, w_orig, atol=0.05)

        # Load back into ParallaxLM
        loaded_model = load_gguf_pretrained(tmp_path)
        assert loaded_model.config.vocab_size == 64
        assert loaded_model.config.tie_weights is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_export_gpt_q4_0_roundtrip():
    """Verify ParallaxLM exported as Q4_0 quantizes 2D matrices with correct GGML descriptors."""
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        max_seq_len=32,
        intermediate_size=64,
        tie_weights=False,
    )
    model = ParallaxLM(cfg)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        export_to_gguf_v3(model, tmp_path, quantization="q4_0")

        gguf_file = load_gguf(tmp_path)
        assert gguf_file.version == 3
        assert gguf_file.metadata["general.file_type"] == 2  # LLAMA_FTYPE_MOSTLY_Q4_0

        assert gguf_file.tensor_infos["token_embd.weight"].ggml_type == GGML_TYPE_Q4_0
        assert gguf_file.tensor_infos["blk.0.attn_q.weight"].ggml_type == GGML_TYPE_Q4_0
        assert gguf_file.tensor_infos["output_norm.weight"].ggml_type == GGML_TYPE_F32

        assert gguf_file.tensors["token_embd.weight"].shape == (64, 32)
        assert gguf_file.tensors["output.weight"].shape == (64, 32)

        # Q4_0 bounded reconstruction check
        w_orig = model.token_embeddings.weight.data
        w_dequant = gguf_file.tensors["token_embd.weight"]
        np.testing.assert_allclose(w_dequant, w_orig, atol=0.35)

        loaded_model = load_gguf_pretrained(tmp_path)
        assert loaded_model.config.vocab_size == 64
        assert loaded_model.config.tie_weights is False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_export_bitmamba_f32_roundtrip():
    """Verify BitMambaLM exported as GGUF v3 preserves all SSM tensors, shapes, and metadata."""
    np.random.seed(42)
    cfg = BitMambaConfig(
        vocab_size=64,
        d_model=32,
        num_layers=2,
        d_state=8,
        d_conv=4,
        expand=2,
        tie_weights=False,
    )
    model = BitMambaLM(cfg)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        export_to_gguf_v3(model, tmp_path, quantization="f32")

        gguf_file = load_gguf(tmp_path)
        assert gguf_file.version == 3
        assert gguf_file.metadata["general.architecture"] == "mamba"
        assert gguf_file.metadata["general.name"] == "BitMambaLM"
        assert gguf_file.metadata["mamba.embedding_length"] == 32
        assert gguf_file.metadata["mamba.block_count"] == 2
        assert gguf_file.metadata["mamba.ssm.d_state"] == 8
        assert gguf_file.metadata["mamba.ssm.d_conv"] == 4
        assert gguf_file.metadata["mamba.ssm.expand"] == 2

        # Check tensor names existence
        expected_tensors = [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.ssm_in.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_conv1d.bias",
            "blk.0.ssm_x.weight",
            "blk.0.ssm_dt.weight",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_a",
            "blk.0.ssm_d",
            "blk.0.ssm_out.weight",
            "blk.1.ssm_in.weight",
            "blk.1.ssm_out.weight",
        ]
        for t_name in expected_tensors:
            assert t_name in gguf_file.tensors, f"Missing tensor {t_name}"

        # Check shapes
        assert gguf_file.tensors["token_embd.weight"].shape == (64, 32)
        assert gguf_file.tensors["output.weight"].shape == (64, 32)
        assert gguf_file.tensors["blk.0.ssm_in.weight"].shape == (128, 32)
        assert gguf_file.tensors["blk.0.ssm_conv1d.weight"].shape == (64, 4)
        assert gguf_file.tensors["blk.0.ssm_conv1d.bias"].shape == (64,)
        assert gguf_file.tensors["blk.0.ssm_a"].shape == (64, 8)
        assert gguf_file.tensors["blk.0.ssm_d"].shape == (64,)
        assert gguf_file.tensors["blk.0.ssm_out.weight"].shape == (32, 64)

        # Check exact value roundtrip
        np.testing.assert_allclose(
            gguf_file.tensors["token_embd.weight"],
            model.token_embeddings.weight.data,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            gguf_file.tensors["blk.0.ssm_conv1d.weight"],
            model.blocks[0].conv_weight.data,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            gguf_file.tensors["blk.0.ssm_a"],
            model.blocks[0].A_log.data,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            gguf_file.tensors["blk.0.ssm_d"],
            model.blocks[0].D.data,
            atol=1e-6,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_export_bitmamba_block_norms_and_param_parity(layers):
    """Verify BitMamba export includes per-block norms and matches parameter count."""
    cfg = BitMambaConfig(vocab_size=64, d_model=32, num_layers=layers, d_state=8, d_conv=4, expand=2)
    model = BitMambaLM(cfg)
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name
    try:
        export_to_gguf_v3(model, tmp_path, quantization="f32")
        gguf = load_gguf(tmp_path)
        assert gguf.metadata.get("general.version") == "2"
        for i in range(layers):
            norm_k = f"blk.{i}.attn_norm.weight"
            assert norm_k in gguf.tensors
            np.testing.assert_allclose(gguf.tensors[norm_k], model.norms[i].weight.data, atol=1e-6)
        assert sum(t.size for t in gguf.tensors.values()) == sum(p.data.size for p in model.parameters())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_export_bitmamba_q8_0_roundtrip():
    """Verify BitMambaLM exported as Q8_0 quantizes projections with divisible shapes."""
    np.random.seed(42)
    cfg = BitMambaConfig(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        d_state=8,
        d_conv=4,
        expand=2,
        tie_weights=False,
    )
    model = BitMambaLM(cfg)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        export_to_gguf_v3(model, tmp_path, quantization="q8_0")

        gguf_file = load_gguf(tmp_path)
        assert gguf_file.metadata["general.file_type"] == 7  # LLAMA_FTYPE_MOSTLY_Q8_0

        # in_proj: (128, 32) -> trailing dim 32 divisible by 32 -> Q8_0
        assert gguf_file.tensor_infos["blk.0.ssm_in.weight"].ggml_type == GGML_TYPE_Q8_0
        # out_proj: (32, 64) -> trailing dim 64 divisible by 32 -> Q8_0
        assert gguf_file.tensor_infos["blk.0.ssm_out.weight"].ggml_type == GGML_TYPE_Q8_0
        # conv1d weight: (64, 4) -> trailing dim 4 not divisible by 32 -> stays F32
        assert gguf_file.tensor_infos["blk.0.ssm_conv1d.weight"].ggml_type == GGML_TYPE_F32

        # Check values
        assert gguf_file.tensors["blk.0.ssm_in.weight"].shape == (128, 32)
        np.testing.assert_allclose(
            gguf_file.tensors["blk.0.ssm_conv1d.weight"],
            model.blocks[0].conv_weight.data,
            atol=1e-6,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_export_invalid_quantization_raises():
    """Verify that unsupported quantization formats raise ValueError."""
    cfg = ParallaxLMConfig(vocab_size=32, embed_dim=16, num_layers=1)
    model = ParallaxLM(cfg)
    with pytest.raises(ValueError, match="Unsupported quantization"):
        export_to_gguf_v3(model, "dummy.gguf", quantization="q2_k")
