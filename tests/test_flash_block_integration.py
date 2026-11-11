import copy
import numpy as np

from parallax.generator import generate
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def _clone_model(src_model: ParallaxLM, use_flash_attn: bool) -> ParallaxLM:
    cfg = copy.deepcopy(src_model.config)
    cfg.use_flash_attn = use_flash_attn
    tgt_model = ParallaxLM(cfg)

    # Copy weights
    tgt_model.token_embeddings.weight.data = src_model.token_embeddings.weight.data.copy()
    tgt_model.final_norm.weight.data = src_model.final_norm.weight.data.copy()
    if not cfg.tie_weights:
        tgt_model.lm_head.weight.data = src_model.lm_head.weight.data.copy()

    for i in range(cfg.num_layers):
        src_blk = src_model.blocks[i]
        tgt_blk = tgt_model.blocks[i]

        tgt_blk.attn_norm.weight.data = src_blk.attn_norm.weight.data.copy()
        tgt_blk.mlp_norm.weight.data = src_blk.mlp_norm.weight.data.copy()

        tgt_blk.attn.q_proj.weight.data = src_blk.attn.q_proj.weight.data.copy()
        tgt_blk.attn.k_proj.weight.data = src_blk.attn.k_proj.weight.data.copy()
        tgt_blk.attn.v_proj.weight.data = src_blk.attn.v_proj.weight.data.copy()
        tgt_blk.attn.out_proj.weight.data = src_blk.attn.out_proj.weight.data.copy()

        if hasattr(tgt_blk.mlp, "w_gate"):
            tgt_blk.mlp.w_gate.weight.data = src_blk.mlp.w_gate.weight.data.copy()
            tgt_blk.mlp.w_up.weight.data = src_blk.mlp.w_up.weight.data.copy()
            tgt_blk.mlp.w_down.weight.data = src_blk.mlp.w_down.weight.data.copy()
        elif hasattr(tgt_blk.mlp, "fc1"):
            tgt_blk.mlp.fc1.weight.data = src_blk.mlp.fc1.weight.data.copy()
            tgt_blk.mlp.fc2.weight.data = src_blk.mlp.fc2.weight.data.copy()

    return tgt_model


def test_flash_attention_block_prefill_equivalence_mha():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
        use_swiglu=True,
        rope_interleaved=False,
        use_flash_attn=False,
    )
    model_std = ParallaxLM(cfg)
    model_flash = _clone_model(model_std, use_flash_attn=True)

    idx = np.array([[1, 5, 8, 12, 20]], dtype=np.int64)

    logits_std = model_std(idx).data
    logits_flash = model_flash(idx).data

    np.testing.assert_allclose(logits_flash, logits_std, atol=1e-5, rtol=1e-5)


def test_flash_attention_block_prefill_equivalence_gqa():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,  # GQA: 4 query heads, 2 kv heads
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
        use_swiglu=True,
        rope_interleaved=False,
        use_flash_attn=False,
    )
    model_std = ParallaxLM(cfg)
    model_flash = _clone_model(model_std, use_flash_attn=True)

    idx = np.array([[2, 4, 6, 8, 10, 12]], dtype=np.int64)

    logits_std = model_std(idx).data
    logits_flash = model_flash(idx).data

    np.testing.assert_allclose(logits_flash, logits_std, atol=1e-5, rtol=1e-5)


def test_flash_attention_generation_with_kv_cache():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=32,
        use_swiglu=True,
        use_flash_attn=False,
    )
    model_std = ParallaxLM(cfg)
    model_flash = _clone_model(model_std, use_flash_attn=True)

    prompt = [1, 2, 3]
    tokens_std = generate(model_std, prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
    tokens_flash = generate(model_flash, prompt, max_new_tokens=4, temperature=0.0, use_cache=True)

    assert tokens_flash == tokens_std


def test_flash_attention_loss_backward_equivalence_mha():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
        use_swiglu=True,
        rope_interleaved=False,
        use_flash_attn=False,
    )
    model_std = ParallaxLM(cfg)
    model_flash = _clone_model(model_std, use_flash_attn=True)

    idx = np.array([[1, 5, 8, 12, 20]], dtype=np.int64)

    out_std = model_std(idx)
    loss_std = (out_std * out_std).sum()
    loss_std.backward()

    out_flash = model_flash(idx)
    loss_flash = (out_flash * out_flash).sum()
    loss_flash.backward()

    for i in range(cfg.num_layers):
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            g_std = getattr(model_std.blocks[i].attn, name).weight.grad
            g_flash = getattr(model_flash.blocks[i].attn, name).weight.grad
            np.testing.assert_allclose(g_flash, g_std, atol=1e-5, rtol=1e-5)


def test_flash_attention_loss_backward_equivalence_gqa():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
        use_swiglu=True,
        rope_interleaved=False,
        use_flash_attn=False,
    )
    model_std = ParallaxLM(cfg)
    model_flash = _clone_model(model_std, use_flash_attn=True)

    idx = np.array([[2, 4, 6, 8, 10, 12]], dtype=np.int64)

    out_std = model_std(idx)
    loss_std = (out_std * out_std).sum()
    loss_std.backward()

    out_flash = model_flash(idx)
    loss_flash = (out_flash * out_flash).sum()
    loss_flash.backward()

    for i in range(cfg.num_layers):
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            g_std = getattr(model_std.blocks[i].attn, name).weight.grad
            g_flash = getattr(model_flash.blocks[i].attn, name).weight.grad
            np.testing.assert_allclose(g_flash, g_std, atol=1e-5, rtol=1e-5)

