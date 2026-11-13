"""Tests for Qwen3CUDAModel QK-Norm, custom head_dim, and streaming GGUF K-Quants."""

from pathlib import Path
import numpy as np
import pytest

from parallax.gguf import GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, save_gguf
from parallax.gguf_loader import load_gguf_pretrained
from parallax.gguf_quant import BLOCK_BYTES_Q4_K, BLOCK_BYTES_Q6_K, dequantize_q4_k, dequantize_q6_k


def test_torch_q4k_and_q6k_kernels_match_numpy() -> None:
    torch = pytest.importorskip("torch")
    from parallax.cuda_engine import _dequant_q4_k_torch, _dequant_q6_k_torch

    rng = np.random.default_rng(55)
    b4 = rng.integers(0, 256, size=(2, BLOCK_BYTES_Q4_K), dtype=np.uint8)
    b4[:, 0:4] = np.array([0x00, 0x3C, 0x00, 0x38], dtype=np.uint8)
    t4 = _dequant_q4_k_torch(torch.from_numpy(b4.reshape(-1)), (2, 256), torch.float32)
    np.testing.assert_allclose(t4.numpy(), dequantize_q4_k(b4.tobytes(), (2, 256)), atol=1e-6)

    b6 = rng.integers(0, 256, size=(2, BLOCK_BYTES_Q6_K), dtype=np.uint8)
    b6[:, 208:210] = np.array([0x00, 0x3C], dtype=np.uint8)
    t6 = _dequant_q6_k_torch(torch.from_numpy(b6.reshape(-1)), (2, 256), torch.float32)
    np.testing.assert_allclose(t6.numpy(), dequantize_q6_k(b6.tobytes(), (2, 256)), atol=1e-6)


def _write_tiny_qwen3_gguf(gguf_path: str) -> None:
    rng = np.random.default_rng(99)
    raw4 = rng.integers(0, 256, size=(16, BLOCK_BYTES_Q4_K), dtype=np.uint8)
    raw4[:, 0:4] = np.array([0x00, 0x30, 0x00, 0x28], dtype=np.uint8)
    raw6 = rng.integers(0, 256, size=(16, BLOCK_BYTES_Q6_K), dtype=np.uint8)
    raw6[:, 208:210] = np.array([0x00, 0x30], dtype=np.uint8)
    gg = {
        "token_embd.weight": raw4[:4].tobytes(), "output_norm.weight": np.ones(64, dtype=np.float32),
        "output.weight": raw6[:4].tobytes(), "blk.0.attn_norm.weight": np.ones(64, dtype=np.float32),
        "blk.0.ffn_norm.weight": np.ones(64, dtype=np.float32), "blk.0.attn_q.weight": raw4[:8].tobytes(),
        "blk.0.attn_k.weight": raw4[:4].tobytes(), "blk.0.attn_v.weight": raw6[:4].tobytes(),
        "blk.0.attn_output.weight": raw4[:8].tobytes(), "blk.0.attn_q_norm.weight": np.ones(16, dtype=np.float32) * 1.1,
        "blk.0.attn_k_norm.weight": np.ones(16, dtype=np.float32) * 0.9, "blk.0.ffn_gate.weight": raw4[:8].tobytes(),
        "blk.0.ffn_up.weight": raw4[:8].tobytes(), "blk.0.ffn_down.weight": raw6[:8].tobytes(),
    }
    ttypes = {k: (GGML_TYPE_Q6_K if k in ("output.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight") else GGML_TYPE_Q4_K) for k, v in gg.items() if isinstance(v, bytes)}
    tshapes = {
        "token_embd.weight": (16, 64), "output.weight": (16, 64), "blk.0.attn_q.weight": (32, 64),
        "blk.0.attn_k.weight": (16, 64), "blk.0.attn_v.weight": (16, 64), "blk.0.attn_output.weight": (64, 32),
        "blk.0.ffn_gate.weight": (32, 64), "blk.0.ffn_up.weight": (32, 64), "blk.0.ffn_down.weight": (64, 32),
    }
    save_gguf(gguf_path, {
        "general.architecture": "qwen3", "qwen3.embedding_length": 64, "qwen3.block_count": 1,
        "qwen3.attention.head_count": 2, "qwen3.attention.head_count_kv": 1, "qwen3.attention.key_length": 16,
        "qwen3.feed_forward_length": 32, "general.vocab_size": 16,
    }, gg, tensor_types=ttypes, tensor_shapes=tshapes)


def test_qwen3_cuda_model_gguf_streaming_matches_parallax_lm(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from parallax.cuda_engine import Qwen3CUDAModel

    gguf_path = str(tmp_path / "tiny_qwen3_kquant.gguf")
    _write_tiny_qwen3_gguf(gguf_path)
    ref_logits = load_gguf_pretrained(gguf_path)(np.array([[1, 3, 5]], dtype=np.int64)).data[0, -1, :]
    eng = Qwen3CUDAModel(gguf_path, device="cpu", dtype=torch.float32)
    cuda_logits, _ = eng.forward_step(torch.tensor([[1, 3, 5]], dtype=torch.long), start_pos=0)
    np.testing.assert_allclose(cuda_logits[0].numpy(), ref_logits, rtol=1e-5, atol=1e-4)


def test_chat_qwen_cuda_build_chatml_prompt_qwen3_modes() -> None:
    from scripts.chat_qwen_cuda import build_chatml_prompt

    hist = [{"role": "system", "content": "sys"}, {"role": "user", "content": "selam"}]
    p_fast = build_chatml_prompt(hist, is_qwen3=True, enable_thinking=False)
    assert p_fast.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    p_think = build_chatml_prompt(hist, is_qwen3=True, enable_thinking=True)
    assert p_think.endswith("<|im_start|>assistant\n")
