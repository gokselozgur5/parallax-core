import os
import tempfile
import numpy as np
import pytest
from parallax.checkpoint import load_checkpoint, save_checkpoint
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig, TransformerBlock
from parallax.tensor import Tensor


def test_transformer_block_forward_backward():
    np.random.seed(42)
    B, T, C = 2, 4, 16
    block = TransformerBlock(embed_dim=C, num_heads=2, use_rope=False)
    x = Tensor(np.random.randn(B, T, C))

    out = block(x)
    assert out.shape == (B, T, C)

    loss = out.sum()
    loss.backward()

    # Residual gradient must flow directly through identity connection
    assert x.grad is not None
    assert x.grad.shape == (B, T, C)
    # Norm and MLP parameters must receive gradients
    assert block.attn_norm.weight.grad is not None
    assert block.mlp.fc1.weight.grad is not None


def test_parallax_lm_end_to_end_forward():
    np.random.seed(1337)
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        use_rope=True,
    )
    model = ParallaxLM(config)

    # Batch of 2 sequences, each of length 5
    tokens = np.array([
        [1, 5, 12, 8, 3],
        [4, 9, 2, 15, 0],
    ])

    logits = model(tokens)
    assert logits.shape == (2, 5, 32)

    # Test backward pass from logits
    loss = logits.sum()
    loss.backward()

    # Embedding and LM head parameters should accumulate gradients
    assert model.token_embeddings.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert np.any(model.token_embeddings.weight.grad != 0.0)


def test_parallax_lm_huggingface_parity_config_and_checkpoint():
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
        use_swiglu=True,
        intermediate_size=28,
        rms_norm_eps=1e-5,
        qkv_bias=True,
        rope_theta=100000.0,
        rope_interleaved=False,
    )
    model = ParallaxLM(config)
    idx = np.array([[1, 3, 5, 7]])
    out1 = model(idx)
    loss = out1.sum()
    loss.backward()

    assert model.blocks[0].attn.q_proj.bias.grad is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = os.path.join(tmpdir, "hf_parity.npz")
        save_checkpoint(ckpt_file, model, step=5)
        restored_model, step = load_checkpoint(ckpt_file)
        assert step == 5
        out2 = restored_model(idx)
        assert np.allclose(out1.data, out2.data, atol=1e-7)
