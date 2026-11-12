"""Unit and analytical gradient verification tests for Qwen3 architecture."""

import numpy as np

from parallax.gradcheck import check_tensor_gradient
from parallax.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from parallax.tensor import Tensor


def test_qwen3_32b_preset_and_decoupled_head_dim_shapes() -> None:
    cfg_32b = Qwen3Config.qwen3_32b()
    assert cfg_32b.embed_dim == 5120 and cfg_32b.num_heads == 64 and cfg_32b.head_dim == 128
    assert cfg_32b.qk_norm is True and cfg_32b.qkv_bias is False
    tiny_cfg = Qwen3Config(
        vocab_size=64, embed_dim=20, num_heads=4, num_kv_heads=2,
        head_dim=8, num_layers=2, intermediate_size=32, max_seq_len=32,
    )
    model = Qwen3ForCausalLM(tiny_cfg)
    attn = model.blocks[0].attn
    assert attn.q_proj.weight.shape == (32, 20)
    assert attn.k_proj.weight.shape == (16, 20)
    assert attn.out_proj.weight.shape == (20, 32)
    tokens = np.array([[1, 5, 9, 2]], dtype=np.int64)
    _, caches = model(tokens, kv_caches=[None, None])  # type: ignore[list-item]
    step_logits, _ = model(np.array([[7]], dtype=np.int64), kv_caches=caches)
    full_logits = model(np.array([[1, 5, 9, 2, 7]], dtype=np.int64))
    np.testing.assert_allclose(step_logits.data[:, 0, :], full_logits.data[:, -1, :], atol=1e-6)


def test_qwen3_qk_norm_and_custom_head_dim_gradcheck() -> None:
    np.random.seed(42)
    cfg = Qwen3Config(
        vocab_size=16, embed_dim=12, num_heads=4, num_kv_heads=2,
        head_dim=4, num_layers=1, intermediate_size=24, max_seq_len=16,
    )
    model = Qwen3ForCausalLM(cfg)
    attn = model.blocks[0].attn
    assert attn.q_norm is not None and attn.k_norm is not None
    attn.q_norm.weight.data = np.random.uniform(0.7, 1.3, size=attn.q_norm.weight.shape)
    attn.k_norm.weight.data = np.random.uniform(0.7, 1.3, size=attn.k_norm.weight.shape)
    x = Tensor(np.random.randn(2, 3, 12) * 0.2, requires_grad=True)

    def loss_fn(inp: Tensor, qw: Tensor, kw: Tensor) -> Tensor:
        assert attn.q_norm is not None and attn.k_norm is not None
        attn.q_norm.weight = qw  # type: ignore[assignment]
        attn.k_norm.weight = kw  # type: ignore[assignment]
        out = attn(inp)
        assert isinstance(out, Tensor)
        return (out * out).sum()

    assert check_tensor_gradient(
        loss_fn, [x, attn.q_norm.weight, attn.k_norm.weight], eps=1e-5, tol=1e-4
    )
