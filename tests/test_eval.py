import numpy as np
import pytest
from parallax.eval import compute_cross_entropy_loss, compute_perplexity
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_compute_cross_entropy_loss_uniform():
    # If all 4 logits are equal (uniform distribution), loss = ln(4)
    logits = np.zeros((10, 4))
    targets = np.random.randint(0, 4, size=10)
    loss = compute_cross_entropy_loss(logits, targets)
    assert np.isclose(loss, np.log(4.0))


def test_compute_cross_entropy_loss_perfect():
    # Large logit for target -> loss near 0
    logits = np.zeros((5, 10))
    targets = np.array([0, 1, 2, 3, 4])
    for i, t in enumerate(targets):
        logits[i, t] = 100.0
    loss = compute_cross_entropy_loss(logits, targets)
    assert loss < 1e-4


def test_compute_perplexity_sliding_window():
    config = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=8,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    tokens = list(np.random.randint(0, 16, size=24))

    ppl = compute_perplexity(model, tokens, max_context_len=8, stride=4)
    assert ppl > 1.0
    assert np.isfinite(ppl)
