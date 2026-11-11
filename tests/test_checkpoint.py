"""
Unit tests for Model Checkpointing and Serialization (M13).
"""

import os
import tempfile
import numpy as np
import pytest

from parallax.checkpoint import (
    get_model_state_dict,
    load_checkpoint,
    load_model_state_dict,
    save_checkpoint,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW


def test_state_dict_extraction_and_loading():
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        tie_weights=False,
    )
    model1 = ParallaxLM(config)
    model2 = ParallaxLM(config)

    # Weights must be initially distinct
    assert not np.allclose(model1.token_embeddings.weight.data, model2.token_embeddings.weight.data)

    state_dict = get_model_state_dict(model1)
    load_model_state_dict(model2, state_dict)

    # Weights must be identical after load
    assert np.allclose(model1.token_embeddings.weight.data, model2.token_embeddings.weight.data)
    assert np.allclose(model1.final_norm.weight.data, model2.final_norm.weight.data)
    assert np.allclose(model1.blocks[0].attn.q_proj.weight.data, model2.blocks[0].attn.q_proj.weight.data)


def test_checkpoint_roundtrip_with_tied_weights_and_optimizer():
    config = ParallaxLMConfig(
        vocab_size=24,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
        tie_weights=True,
    )
    model = ParallaxLM(config)
    opt = AdamW(model.parameters(), lr=0.005)

    # Fake training step to populate moments
    idx = np.array([[1, 2, 3]])
    logits = model(idx)
    loss = logits.sum()
    loss.backward()
    opt.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.npz")
        save_checkpoint(ckpt_path, model, optimizer=opt, step=42)

        # Restore from checkpoint
        new_model, step = load_checkpoint(ckpt_path)
        assert step == 42
        assert new_model.config.tie_weights is True
        assert new_model.config.vocab_size == 24
        assert new_model.token_embeddings.weight is new_model.lm_head.weight

        # Check numerical parity of forward pass
        out_orig = model(idx).data
        out_restored = new_model(idx).data
        assert np.allclose(out_orig, out_restored, atol=1e-7)


def test_checkpoint_roundtrip_sparse_moe():
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=8,
        use_swiglu=True,
        num_experts=4,
        top_k_experts=2,
    )
    model = ParallaxLM(config)
    idx = np.array([[2, 5, 7, 11]])
    logits_before = model(idx).data

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "moe_ckpt.npz")
        save_checkpoint(ckpt_path, model, step=10)

        restored_model, step = load_checkpoint(ckpt_path)
        assert step == 10
        assert restored_model.config.num_experts == 4
        assert restored_model.config.top_k_experts == 2
        logits_after = restored_model(idx).data
        assert np.allclose(logits_before, logits_after, atol=1e-7)


def test_checkpoint_shape_and_dtype_validation():
    config = ParallaxLMConfig(vocab_size=16, embed_dim=8, num_heads=2, num_layers=1, max_seq_len=8)
    model = ParallaxLM(config)
    state_dict = get_model_state_dict(model)

    bad_shape_dict = dict(state_dict, **{"final_norm.weight": np.ones((16,))})
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_model_state_dict(model, bad_shape_dict)

    bad_dtype_dict = dict(state_dict, **{"final_norm.weight": np.array(["a"] * 8)})
    with pytest.raises(TypeError, match="must have numeric dtype"):
        load_model_state_dict(model, bad_dtype_dict)

    bad_type_dict = dict(state_dict, **{"final_norm.weight": [1.0] * 8})
    with pytest.raises(TypeError, match="must be a numpy ndarray"):
        load_model_state_dict(model, bad_type_dict)
