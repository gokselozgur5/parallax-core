import math
import numpy as np
from parallax.generator import CharTokenizer, generate
from parallax.gradcheck import check_tensor_gradient
from parallax.loss import cross_entropy_loss
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.tensor import Tensor


def _reference_ce(rows: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    losses, grads = [], []
    n = len(targets)
    for row, target in zip(rows.tolist(), targets.tolist()):
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        z = sum(exps)
        probs = [e / z for e in exps]
        losses.append(-(row[target] - m - math.log(z)))
        grads.append([(p - (1.0 if j == target else 0.0)) / n for j, p in enumerate(probs)])
    return sum(losses) / n, np.array(grads, dtype=np.float64)


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

    expected_loss, expected_grad = _reference_ce(logits.data, targets)
    assert math.isclose(float(loss.data), expected_loss, rel_tol=1e-12, abs_tol=1e-12)
    assert np.allclose(logits.grad, expected_grad, rtol=1e-12, atol=1e-12)
    assert (
        check_tensor_gradient(lambda x: cross_entropy_loss(x, targets), [logits], eps=1e-5, tol=1e-5)
        is True
    )


def test_cross_entropy_loss_3d_shift_and_upstream_scale():
    raw = np.array(
        [
            [[1.5, -0.5, 0.2, 2.1], [-1.0, 0.0, 0.8, -0.3]],
            [[0.4, 1.2, -0.7, 0.9], [2.5, 0.1, -1.5, 0.3]],
        ]
    )
    targets = np.array([[3, 2], [1, 0]])
    expected_loss, expected_grad_flat = _reference_ce(raw.reshape(-1, 4), targets.reshape(-1))
    expected_grad = expected_grad_flat.reshape(raw.shape)

    shifted_loss = cross_entropy_loss(Tensor(raw + 500.0), targets)
    assert math.isclose(float(shifted_loss.data), expected_loss, rel_tol=1e-12, abs_tol=1e-12)

    logits = Tensor(raw.copy())
    scaled = cross_entropy_loss(logits, targets) * 2.5
    scaled.backward()
    assert logits.grad is not None
    assert np.allclose(logits.grad, expected_grad * 2.5, rtol=1e-12, atol=1e-12)
    assert (
        check_tensor_gradient(lambda x: cross_entropy_loss(x, targets), [logits], eps=1e-5, tol=1e-5)
        is True
    )


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
