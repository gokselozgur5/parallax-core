"""
Train a character-level or BPE ParallaxLM on the Shakespeare corpus.
"""

import os
import sys
import numpy as np

# Ensure parallax is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parallax.checkpoint import save_checkpoint
from parallax.generator import CharTokenizer, generate
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW, CosineAnnealingWithWarmup
from parallax.trainer import Trainer, TrainerConfig


def train_shakespeare(
    data_path: str = "data/shakespeare.txt",
    save_path: str = "data/shakespeare_model.npz",
    steps: int = 50,
):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    tokens = np.array(tokenizer.encode(text), dtype=np.int64)

    config = ParallaxLMConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=48,
        num_heads=4,
        num_kv_heads=2,  # GQA
        num_layers=3,
        max_seq_len=64,
        use_rope=True,
        use_swiglu=True,  # SwiGLU FFN
        tie_weights=True,
    )
    model = ParallaxLM(config)
    optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps=5, total_steps=steps, min_lr=0.005)

    batch_size = 4
    seq_len = 32

    def data_generator():
        while True:
            starts = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
            x = np.stack([tokens[i : i + seq_len] for i in starts])
            y = np.stack([tokens[i + 1 : i + seq_len + 1] for i in starts])
            yield x, y

    trainer_cfg = TrainerConfig(
        max_steps=steps,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        eval_interval=10,
        eval_steps=2,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=trainer_cfg,
        train_loader=data_generator(),
        scheduler=scheduler,
    )

    history = trainer.train()
    save_checkpoint(save_path, model, optimizer=optimizer, step=steps)
    return model, tokenizer, history


if __name__ == "__main__":
    train_shakespeare()
