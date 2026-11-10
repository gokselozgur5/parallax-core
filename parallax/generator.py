"""
Character-level tokenizer and autoregressive text generation utilities.
"""

from typing import Dict, Iterator, List, Optional, Tuple
import numpy as np
from parallax.attention_sink import RollingKVCache
from parallax.models.gpt import ParallaxLM
from parallax.tensor import no_grad


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
    top_p: Optional[float] = None,
) -> int:
    """Samples next token from logits with temperature, top-k, and top-p (nucleus) filtering.

    Args:
        logits: 1D array of shape [vocab_size]
        temperature: Temperature scaling factor (lower = more deterministic)
        top_k: If set, restricts sampling to the top-k highest probability tokens
        top_p: If set (in (0, 1]), restricts sampling to the smallest set of tokens
               whose cumulative probability mass meets or exceeds top_p (Nucleus sampling)
    """
    if temperature is not None:
        if np.isnan(temperature) or np.isinf(temperature) or temperature < 0:
            raise ValueError(f"Invalid temperature: {temperature}. Must be finite and non-negative.")
        if temperature < 1e-7 or temperature == 0:
            return int(np.argmax(logits))
    else:
        return int(np.argmax(logits))

    scaled_logits = logits.copy() / temperature

    # Top-K filtering
    if top_k is not None and 0 < top_k < len(scaled_logits):
        indices_to_remove = np.argsort(scaled_logits)[:-top_k]
        scaled_logits[indices_to_remove] = -np.inf

    # Softmax probabilities
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)

    # Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort probabilities descending
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Remove tokens with cumulative probability above top_p (keep at least the first token)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False  # Keep at least top 1 token

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        # Renormalize
        sum_p = np.sum(probs)
        if sum_p > 0:
            probs = probs / sum_p
        else:
            probs[sorted_indices[0]] = 1.0

    return int(np.random.choice(len(probs), p=probs))


def generate_stream(
    model: ParallaxLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 20,
    temperature: float = 0.8,
    top_k: Optional[int] = 10,
    top_p: Optional[float] = None,
    use_cache: bool = True,
    eos_token_id: Optional[int] = None,
    rolling_cache: Optional[RollingKVCache] = None,
) -> Iterator[int]:
    """Yields newly generated token IDs one by one."""
    tokens = list(prompt_tokens)
    max_seq_len = model.config.max_seq_len

    if not use_cache:
        for _ in range(max_new_tokens):
            context = tokens[-max_seq_len:]
            idx = np.array([context], dtype=np.int64)
            with no_grad():
                logits = model(idx)
            next_token_logits = logits.data[0, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens.append(next_token)
            yield next_token
            if eos_token_id is not None and next_token == eos_token_id:
                break
        return

    # Fast KV cached stream
    context = tokens[-max_seq_len:]
    idx = np.array([context], dtype=np.int64)

    B = 1
    num_heads = model.config.num_heads
    num_kv_heads = model.config.num_kv_heads if model.config.num_kv_heads is not None else num_heads
    head_dim = model.config.embed_dim // num_heads
    kv_caches: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64),
         np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64))
        for _ in range(model.config.num_layers)
    ]

    with no_grad():
        logits, kv_caches = model(idx, kv_caches=kv_caches)
    if rolling_cache is not None:
        kv_caches = rolling_cache.evict_layers(kv_caches)
    next_token_logits = logits.data[0, -1, :]
    next_token = sample_next_token(
        next_token_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    tokens.append(next_token)
    yield next_token
    if eos_token_id is not None and next_token == eos_token_id:
        return

    for _ in range(max_new_tokens - 1):
        if rolling_cache is None and len(tokens) >= max_seq_len:
            break

        idx_next = np.array([[next_token]], dtype=np.int64)
        rope_offset = rolling_cache.total_seen if rolling_cache is not None else None
        with no_grad():
            logits, kv_caches = model(idx_next, kv_caches=kv_caches, rope_offset=rope_offset)
        if rolling_cache is not None:
            kv_caches = rolling_cache.evict_layers(kv_caches)
        next_token_logits = logits.data[0, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tokens.append(next_token)
        yield next_token
        if eos_token_id is not None and next_token == eos_token_id:
            break


def generate(
    model: ParallaxLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 20,
    temperature: float = 0.8,
    top_k: Optional[int] = 10,
    top_p: Optional[float] = None,
    use_cache: bool = True,
    eos_token_id: Optional[int] = None,
    rolling_cache: Optional[RollingKVCache] = None,
) -> List[int]:
    """Autoregressively generates new tokens conditioned on a prompt.

    Args:
        model: ParallaxLM model.
        prompt_tokens: List of initial prompt token IDs.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k filtering limit.
        top_p: Top-p nucleus filtering threshold.
        use_cache: If True, uses incremental O(1) step KV caching instead of full O(N^2) recomputation.
        eos_token_id: Optional end-of-sequence token ID to terminate generation early.
    """
    tokens = list(prompt_tokens)
    max_seq_len = model.config.max_seq_len

    with no_grad():
        if not use_cache:
            for _ in range(max_new_tokens):
                context = tokens[-max_seq_len:]
                idx = np.array([context], dtype=np.int64)
                logits = model(idx)
                next_token_logits = logits.data[0, -1, :]
                next_token = sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                tokens.append(next_token)
                if eos_token_id is not None and next_token == eos_token_id:
                    break
            return tokens

        # Fast KV cached generation
        # 1. Warm up prompt context into KV caches
        context = tokens[-max_seq_len:]
        idx = np.array([context], dtype=np.int64)

        # Initialize empty caches per layer (respecting GQA num_kv_heads if configured)
        B = 1
        num_heads = model.config.num_heads
        num_kv_heads = model.config.num_kv_heads if model.config.num_kv_heads is not None else num_heads
        head_dim = model.config.embed_dim // num_heads
        kv_caches: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64),
             np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64))
            for _ in range(model.config.num_layers)
        ]

        logits, kv_caches = model(idx, kv_caches=kv_caches)
        if rolling_cache is not None:
            kv_caches = rolling_cache.evict_layers(kv_caches)
        next_token_logits = logits.data[0, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tokens.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            return tokens

        # 2. Incrementally generate one token at a time using cached history
        for _ in range(max_new_tokens - 1):
            if rolling_cache is None and len(tokens) >= max_seq_len:
                # Context exceeded max length for pure increment
                break

            idx_next = np.array([[next_token]], dtype=np.int64)
            rope_offset = rolling_cache.total_seen if rolling_cache is not None else None
            logits, kv_caches = model(idx_next, kv_caches=kv_caches, rope_offset=rope_offset)
            if rolling_cache is not None:
                kv_caches = rolling_cache.evict_layers(kv_caches)
            next_token_logits = logits.data[0, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

    return tokens
