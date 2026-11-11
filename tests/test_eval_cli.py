from parallax.gguf_loader import save_parallax_to_gguf
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from scripts.eval_model import eval_model_on_text


def test_eval_model_cli_gguf(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=128,
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    model_path = str(tmp_path / "model.gguf")
    text_path = str(tmp_path / "sample.txt")

    save_parallax_to_gguf(model, model_path)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Hello world! This is a test evaluation sequence for parallax.")

    ppl = eval_model_on_text(model_path, text_path, context_len=16, stride=8)
    assert ppl > 1.0
    assert isinstance(ppl, float)
