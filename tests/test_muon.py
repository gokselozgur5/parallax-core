import numpy as np
from parallax.muon import newton_schulz5


def test_newton_schulz5_produces_semi_orthogonal_matrix():
    np.random.seed(42)
    G = np.random.randn(16, 32).astype(np.float32)
    O = newton_schulz5(G, steps=5)

    assert O.shape == G.shape
    # O @ O.T should be approximately identity (since M < N)
    gram = np.dot(O, O.T)
    # Check that diagonal is dominated and non-zero
    assert np.all(np.diag(gram) > 0.5)


from parallax.muon import Muon


def test_muon_step_decreases_linear_loss():
    np.random.seed(42)
    # Simple linear objective: minimize ||W @ x - y||^2
    d_in, d_out = 8, 8
    W = np.random.randn(d_out, d_in).astype(np.float32)
    x = np.random.randn(d_in).astype(np.float32)
    y = np.random.randn(d_out).astype(np.float32)

    initial_loss = float(np.sum((np.dot(W, x) - y) ** 2))

    optimizer = Muon([W], lr=0.05, momentum=0.9, nesterov=True)

    for _ in range(5):
        pred = np.dot(W, x)
        diff = pred - y
        grad = 2.0 * np.outer(diff, x)
        optimizer.step([grad])

    final_loss = float(np.sum((np.dot(W, x) - y) ** 2))
    assert final_loss < initial_loss


from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.muon import MuonOptimizer
from parallax.qgalore import QGaLoreAdamW
from parallax.trainer import Trainer, TrainerConfig
from parallax.tensor import Tensor


def test_muon_optimizer_with_trainer():
    cfg = ParallaxLMConfig(vocab_size=32, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=8)
    model = ParallaxLM(cfg)
    optimizer = MuonOptimizer(model.parameters(), lr=0.01)

    def synthetic_data():
        while True:
            x = np.random.randint(0, 32, (2, 8))
            y = np.random.randint(0, 32, (2, 8))
            yield x, y

    trainer_cfg = TrainerConfig(max_steps=2, grad_accum_steps=1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,  # type: ignore[arg-type]
        config=trainer_cfg,
        train_loader=synthetic_data(),
    )
    history = trainer.train()
    assert len(history) == 2


def test_qgalore_adamw_with_trainer():
    cfg = ParallaxLMConfig(vocab_size=32, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=8)
    model = ParallaxLM(cfg)
    optimizer = QGaLoreAdamW(model.parameters(), lr=0.01, rank=4, update_interval=2)

    def synthetic_data():
        while True:
            x = np.random.randint(0, 32, (2, 8))
            y = np.random.randint(0, 32, (2, 8))
            yield x, y

    trainer_cfg = TrainerConfig(max_steps=2, grad_accum_steps=1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,  # type: ignore[arg-type]
        config=trainer_cfg,
        train_loader=synthetic_data(),
    )
    history = trainer.train()
    assert len(history) == 2
