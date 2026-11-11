from parallax.gguf import GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, load_gguf
from parallax.gguf_loader import save_parallax_to_gguf
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from scripts.quantize_model import quantize_model


def test_quantize_model_cli_q8_0(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    f32_path = str(tmp_path / "model_f32.gguf")
    q8_path = str(tmp_path / "model_q8.gguf")

    save_parallax_to_gguf(model, f32_path)
    quantize_model(f32_path, q8_path, "q8_0")

    loaded_q8 = load_gguf(q8_path)
    assert "blk.0.attn_q.weight" in loaded_q8.tensors
    # Check that tensor ggml_type is Q8_0
    q_info = loaded_q8.tensor_infos["blk.0.attn_q.weight"]
    assert q_info.ggml_type == GGML_TYPE_Q8_0


def test_quantize_model_cli_q4_0(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    f32_path = str(tmp_path / "model_f32.gguf")
    q4_path = str(tmp_path / "model_q4.gguf")

    save_parallax_to_gguf(model, f32_path)
    quantize_model(f32_path, q4_path, "q4_0")

    loaded_q4 = load_gguf(q4_path)
    q_info = loaded_q4.tensor_infos["blk.0.attn_q.weight"]
    assert q_info.ggml_type == GGML_TYPE_Q4_0

