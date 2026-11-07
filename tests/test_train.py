import numpy as np
import pytest
from parallax.generator import CharTokenizer, generate, sample_next_token
from parallax.loss import cross_entropy_loss
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.tensor import Tensor


def test_cross_entropy_loss_and_backward():
    # 2 samples, 3 classes
    logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 1.2]]))
    targets = np.array([0, 1])

    loss = cross_entropy_loss(logits, targets)
    assert loss.shape == ()
    assert loss.data > 0.0

    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == (2, 3)
    # Sum of gradients per row must be zero for softmax cross-entropy
    row_sums = np.sum(logits.grad, axis=-1)
    assert np.allclose(row_sums, [0.0, 0.0], atol=1e-5)


def test_adamw_step_decreases_loss():
    np.random.seed(42)
    # Simple linear regression task: learn y = 2*x
    from parallax.nn import Linear

    layer = Linear(1, 1, bias=False)
    layer.weight.data.fill(0.1)  # Far from target 2.0
    optimizer = AdamW(layer.parameters(), lr=0.1, weight_decay=0.0)

    x = Tensor(np.array([[1.0], [2.0], [3.0]]))
    target_y = np.array([[2.0], [4.0], [6.0]])

    initial_diff = layer(x) + (-target_y)
    initial_loss = (initial_diff * initial_diff).mean().data

    for _ in range(15):
        optimizer.zero_grad()
        pred = layer(x)
        diff = pred + (-target_y)
        loss = (diff * diff).mean()
        loss.backward()
        optimizer.step()

    final_diff = layer(x) + (-target_y)
    final_loss = (final_diff * final_diff).mean().data
    assert final_loss < initial_loss * 0.1


def test_tokenizer_and_autoregressive_generation():
    text = "hello parallax"
    tokenizer = CharTokenizer(text)
    assert tokenizer.vocab_size == len(set(text))

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello"

    config = ParallaxLMConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
        use_rope=False,
    )
    model = ParallaxLM(config)

    prompt = tokenizer.encode("h")
    generated_indices = generate(model, prompt, max_new_tokens=5, temperature=0.7)
    assert len(generated_indices) == 6
    output_text = tokenizer.decode(generated_indices)
    assert len(output_text) == 6
