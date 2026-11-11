import math
import numpy as np
import pytest

from parallax.attention import CausalSelfAttention, apply_rope
from parallax.attention_sink import RollingKVCache
from parallax.engine import Value
from parallax.eval import compute_cross_entropy_loss, compute_perplexity
from parallax.generator import generate, generate_stream
from parallax.linear_attention import LinearAttention
from parallax.loader_utils import _assign_bias, _assign_param
from parallax.loss import cross_entropy_loss, perplexity_from_loss
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear, Parameter
from parallax.optim import AdamW, CosineAnnealingWithWarmup
from parallax.prune import prune_unstructured_magnitude
from parallax.tensor import Tensor
from parallax.tokenizer import BPETokenizer
from parallax.viz import format_attention_heatmap
from parallax.yarn import apply_yarn_rope


def _create_mini_model():
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=32,
        intermediate_size=32,
    )
    np.random.seed(42)
    return ParallaxLM(cfg)


def test_shared_loader_utils_assign_param_and_bias():
    param = Parameter(np.zeros((4, 8)), label="weight")
    tensors = {"w": np.ones((4, 8)), "mismatch": np.ones((4, 4))}
    _assign_param(param, tensors, "w")
    assert np.allclose(param.data, 1.0)

    # Missing key is no-op
    _assign_param(param, tensors, "missing")
    assert np.allclose(param.data, 1.0)

    # Shape mismatch raises ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        _assign_param(param, tensors, "mismatch")

    # Linear bias assignment
    proj = Linear(8, 4, bias=False)
    bias_tensors = {"b": np.full((4,), 2.5), "b_bad": np.zeros((8,))}
    _assign_bias(proj, bias_tensors, "b")
    assert proj.bias is not None
    assert np.allclose(proj.bias.data, 2.5)

    with pytest.raises(ValueError, match="Shape mismatch"):
        _assign_bias(proj, bias_tensors, "b_bad")


def test_generator_deduplication_equivalence():
    model = _create_mini_model()
    prompt = [1, 2, 3]

    np.random.seed(123)
    out_gen = generate(model, prompt, max_new_tokens=5, temperature=0.0, use_cache=True)

    np.random.seed(123)
    out_stream = list(generate_stream(model, prompt, max_new_tokens=5, temperature=0.0, use_cache=True))

    assert out_gen == prompt + out_stream


def test_perplexity_functions_naming_and_consistency():
    # loss.perplexity_from_loss
    loss_val = math.log(8.0)
    ppl1 = perplexity_from_loss(loss_val)
    assert math.isclose(ppl1, 8.0, rel_tol=1e-5)

    # eval.compute_perplexity on model and tokens
    model = _create_mini_model()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    ppl2 = compute_perplexity(model, tokens, max_context_len=8, stride=4)
    assert ppl2 > 0.0
    assert not math.isinf(ppl2)


def test_validation_asserts_replaced_with_exceptions():
    # attention RoPE dim must be even
    x = np.random.randn(2, 4, 7)
    with pytest.raises(ValueError, match="RoPE requires even embedding dimension"):
        apply_rope(x)

    # attention MHA embed_dim divisible by num_heads
    with pytest.raises(ValueError, match="embed_dim must be divisible by num_heads"):
        CausalSelfAttention(embed_dim=15, num_heads=2)

    # attention sink window_size > 0
    with pytest.raises(ValueError, match="window_size must be positive"):
        RollingKVCache(sink_tokens=4, window_size=0)
    with pytest.raises(ValueError, match="sink_tokens must be non-negative"):
        RollingKVCache(sink_tokens=-1, window_size=32)

    # engine Value power and log
    v = Value(2.0)
    with pytest.raises(TypeError, match="Power must be int or float"):
        _ = v ** "bad"  # type: ignore
    v_neg = Value(-1.0)
    with pytest.raises(ValueError, match="Logarithm undefined"):
        v_neg.log()

    # eval compute_cross_entropy_loss shape validation
    with pytest.raises(ValueError, match="logits must be 2D"):
        compute_cross_entropy_loss(np.ones((2, 2, 2)), np.ones(2, dtype=np.int64))
    with pytest.raises(ValueError, match="targets must be 1D"):
        compute_cross_entropy_loss(np.ones((2, 4)), np.ones((2, 1), dtype=np.int64))
    with pytest.raises(ValueError, match="length mismatch"):
        compute_cross_entropy_loss(np.ones((2, 4)), np.ones(3, dtype=np.int64))

    # linear attention embed_dim
    with pytest.raises(ValueError, match="embed_dim must be divisible by num_heads"):
        LinearAttention(embed_dim=15, num_heads=2)

    # loss cross_entropy_loss label smoothing range
    logits = Tensor(np.ones((2, 4)))
    targets = np.array([0, 1])
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        cross_entropy_loss(logits, targets, label_smoothing=1.5)

    # optim total_steps >= warmup_steps
    opt = AdamW([Parameter(np.ones(2))])
    with pytest.raises(ValueError, match="total_steps must be >= warmup_steps"):
        CosineAnnealingWithWarmup(opt, warmup_steps=10, total_steps=5)

    # prune sparsity range
    with pytest.raises(ValueError, match="sparsity must be between 0.0 and 1.0"):
        prune_unstructured_magnitude(np.ones((4, 4)), sparsity=1.5)

    # tensor matmul type and backward scalar shape
    t1 = Tensor(np.ones((2, 2)))
    with pytest.raises(TypeError, match="Matmul requires another Tensor"):
        _ = t1 @ np.ones((2, 2))  # type: ignore
    with pytest.raises(ValueError, match="Default backward only supported for scalar outputs"):
        t1.backward()

    # tokenizer vocab_size
    tok = BPETokenizer()
    with pytest.raises(ValueError, match="vocab_size must be at least 260"):
        tok.train("hello", vocab_size=200)

    # viz format_attention_heatmap 2D check and tokens len
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        format_attention_heatmap(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="tokens length must match"):
        format_attention_heatmap(np.ones((2, 2)), tokens=["a", "b", "c"])

    # yarn head dim must be even
    with pytest.raises(ValueError, match="YaRN requires even head dimension"):
        apply_yarn_rope(np.ones((1, 4, 7)))


def test_license_metadata_consistency():
    import pathlib
    import tomllib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    cargo_path = repo_root / "crates" / "parallax-rs" / "Cargo.toml"
    license_path = repo_root / "LICENSE"

    assert license_path.is_file()
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    with open(cargo_path, "rb") as f:
        cargo_data = tomllib.load(f)

    pyproject_license = pyproject_data.get("project", {}).get("license", {}).get("text")
    assert pyproject_license == "Proprietary"

    # Cargo.toml points to the proprietary repo LICENSE file
    license_file_rel = cargo_data.get("package", {}).get("license-file")
    assert license_file_rel is not None
    cargo_license_path = (cargo_path.parent / license_file_rel).resolve()
    assert cargo_license_path == license_path.resolve()
