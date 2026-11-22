import numpy as np
from parallax.eval import compute_cross_entropy_loss, compute_perplexity


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


def test_compute_perplexity_sliding_window(monkeypatch):
    class UniformStubModel:
        def __call__(self, inputs):
            return np.zeros((*inputs.shape, 16))

    tokens = [i % 16 for i in range(24)]
    scored_targets = []
    orig_loss = compute_cross_entropy_loss

    def tracking_loss(logits, targets):
        scored_targets.extend(targets.tolist())
        return orig_loss(logits, targets)

    monkeypatch.setattr("parallax.eval.compute_cross_entropy_loss", tracking_loss)
    ppl = compute_perplexity(UniformStubModel(), tokens, max_context_len=8, stride=4)

    assert ppl > 1.0
    assert np.isfinite(ppl)
    assert np.isclose(ppl, 16.0)
    assert scored_targets == tokens[1:]
