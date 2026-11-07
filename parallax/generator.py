"""
Character-level tokenizer and autoregressive text generation utilities.
"""

from typing import Dict, List, Optional
import numpy as np
from parallax.models.gpt import ParallaxLM


class CharTokenizer:
    """Simple, transparent character-level vocabulary tokenizer."""

    def __init__(self, text: Optional[str] = None):
        if text is None:
            # Standard ASCII printable vocabulary
            chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
        else:
            chars = sorted(list(set(text)))

        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.itos.get(i, "?") for i in indices])


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> int:
    """Samples next token from logits with temperature and top-k filtering.

    Args:
        logits: 1D array of shape [vocab_size]
        temperature: Temperature scaling factor (lower = more deterministic)
        top_k: If set, restricts sampling to the top-k highest probability tokens
    """
    if temperature <= 0.0:
        return int(np.argmax(logits))

    scaled_logits = logits / temperature

    if top_k is not None and top_k > 0 and top_k < len(scaled_logits):
        # Zero out logits below top_k threshold
        indices_to_remove = np.argsort(scaled_logits)[:-top_k]
        scaled_logits[indices_to_remove] = -1e9

    # Softmax probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)

    return int(np.random.choice(len(probs), p=probs))


def generate(
    model: ParallaxLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 20,
    temperature: float = 0.8,
    top_k: Optional[int] = 10,
) -> List[int]:
    """Autoregressively generates new tokens conditioned on a prompt."""
    tokens = list(prompt_tokens)
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len
        context = tokens[-max_seq_len:]
        idx = np.array([context], dtype=np.int64)

        # Forward pass to get logits for the last token
        logits = model(idx)  # [1, len(context), vocab_size]
        next_token_logits = logits.data[0, -1, :]

        next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)
        tokens.append(next_token)

    return tokens
