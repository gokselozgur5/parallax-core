"""
KV-Cache Pruning and Dynamic Eviction (StreamingLLM, Scissorhands, H2O Heavy-Hitter Oracle).

Provides attention-sink preservation, sliding window retention, and cumulative
attention-score based Heavy Hitter KV-cache eviction.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class PruningMetrics:
    total_tokens_seen: int = 0
    total_evicted: int = 0
    current_cache_size: int = 0
    eviction_steps: int = 0


class KVCachePruner:
    """Dynamic KV-Cache Pruner combining Attention Sinks, Local Window, and H2O.

    Architecture:
    - Sinks (Attention Sinks, e.g. first 4 tokens): Always retained. Initial tokens
      absorb outsized attention mass in autoregressive LLMs (StreamingLLM).
    - Local Context Window (e.g. last L tokens): Always retained to preserve recent
      fluent local context and recency bias.
    - Heavy Hitters (H2O / Scissorhands): Selected from the middle tokens between
      the sink and local window, based on accumulated attention scores.
    - Total cache budget = num_sinks + window_size + heavy_hitters_budget.

    Shapes supported:
    - 4D KV tensors: [batch_size, num_heads, seq_len, head_dim] (standard Parallax format).
    - 3D attention scores: [batch_size, num_heads, seq_len] or [batch_size, seq_len]
      or 4D full attention matrix: [batch_size, num_heads, q_len, kv_len].
    """

    def __init__(
        self,
        num_sinks: int = 4,
        window_size: int = 16,
        heavy_hitters: int = 16,
        max_budget: Optional[int] = None,
    ):
        if num_sinks < 0:
            raise ValueError("num_sinks must be non-negative")
        if window_size < 0:
            raise ValueError("window_size must be non-negative")
        if heavy_hitters < 0:
            raise ValueError("heavy_hitters must be non-negative")

        self.num_sinks = num_sinks
        self.window_size = window_size
        self.heavy_hitters = heavy_hitters

        # If max_budget is provided, adjust heavy_hitters to fit
        if max_budget is not None:
            if max_budget < (num_sinks + window_size):
                raise ValueError(
                    f"max_budget ({max_budget}) cannot be smaller than num_sinks + window_size "
                    f"({num_sinks} + {window_size} = {num_sinks + window_size})"
                )
            self.heavy_hitters = max_budget - (num_sinks + window_size)

        self.budget = self.num_sinks + self.window_size + self.heavy_hitters

        # Cumulative attention scores per layer: list of np.ndarray [B, H, current_seq_len] or None
        self._accumulated_scores: List[Optional[np.ndarray]] = []
        self.metrics = PruningMetrics()

    def reset(self) -> None:
        """Reset internal accumulator and metrics."""
        self._accumulated_scores = []
        self.metrics = PruningMetrics()

    def update_attention_scores(
        self,
        layer_idx: int,
        attn_weights: np.ndarray,
    ) -> None:
        """Accumulate attention scores for the given layer.

        Args:
            layer_idx: Index of transformer layer.
            attn_weights: Attention weights of shape:
                - [B, H, q_len, kv_len] (full attention map)
                - [B, H, kv_len] (head-level weights to KV)
                - [B, kv_len] (pooled weights)
        """
        while len(self._accumulated_scores) <= layer_idx:
            self._accumulated_scores.append(None)

        # Reduce query dimension if present
        if attn_weights.ndim == 4:
            # Sum or mean over query dimension: [B, H, kv_len]
            step_scores = np.sum(attn_weights, axis=2)
        elif attn_weights.ndim == 3:
            step_scores = attn_weights
        elif attn_weights.ndim == 2:
            # [B, 1, kv_len]
            step_scores = attn_weights[:, None, :]
        else:
            raise ValueError(f"Unsupported attention weights shape: {attn_weights.shape}")

        current = self._accumulated_scores[layer_idx]
        if current is None:
            self._accumulated_scores[layer_idx] = step_scores.copy()
        else:
            # Handle sequence length difference (e.g. if cache grew before accumulation)
            curr_len = current.shape[-1]
            new_len = step_scores.shape[-1]
            if curr_len == new_len:
                self._accumulated_scores[layer_idx] = current + step_scores
            elif new_len > curr_len:
                # Pad current with zeros for the new tokens then add
                pad_width = [(0, 0)] * (current.ndim - 1) + [(0, new_len - curr_len)]
                padded_curr = np.pad(current, pad_width, mode="constant", constant_values=0.0)
                self._accumulated_scores[layer_idx] = padded_curr + step_scores
            else:
                # new_len < curr_len
                self._accumulated_scores[layer_idx] = current[..., :new_len] + step_scores

    def compute_keep_indices(
        self,
        seq_len: int,
        layer_idx: int = 0,
        attn_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the 1D sorted array of indices to retain in the KV cache.

        Args:
            seq_len: Current length of the KV cache along sequence dimension.
            layer_idx: Layer index to retrieve cumulative attention scores from.
            attn_scores: Optional explicit attention scores to use / accumulate.

        Returns:
            1D np.ndarray of integer indices sorted in ascending order.
        """
        if seq_len <= self.budget:
            return np.arange(seq_len, dtype=np.int64)

        # 1. Sinks: [0, ..., num_sinks - 1]
        sink_indices = set(range(min(self.num_sinks, seq_len)))

        # 2. Local context window: [seq_len - window_size, ..., seq_len - 1]
        window_start = max(self.num_sinks, seq_len - self.window_size)
        window_indices = set(range(window_start, seq_len))

        # Combined protected indices
        protected_indices = sink_indices.union(window_indices)

        # Middle candidates for heavy-hitters
        candidate_indices = [i for i in range(seq_len) if i not in protected_indices]

        needed_hh = self.heavy_hitters
        if len(candidate_indices) <= needed_hh or needed_hh == 0:
            selected_middle = candidate_indices[:needed_hh]
        else:
            # Determine attention scores for candidates
            scores = None
            if attn_scores is not None:
                scores = attn_scores
            elif (
                len(self._accumulated_scores) > layer_idx
                and self._accumulated_scores[layer_idx] is not None
            ):
                scores = self._accumulated_scores[layer_idx]

            if scores is not None:
                # Aggregate score across batch and heads: shape [seq_len]
                while scores.ndim > 1:
                    scores = np.sum(scores, axis=0)

                # Ensure candidate scores match indices
                if len(scores) >= seq_len:
                    cand_scores = scores[candidate_indices]
                else:
                    # Pad missing scores with 0
                    full_scores = np.zeros(seq_len, dtype=scores.dtype)
                    full_scores[: len(scores)] = scores
                    cand_scores = full_scores[candidate_indices]

                # Top-K candidate selection (H2O Heavy Hitters)
                topk_rel_idx = np.argsort(cand_scores)[-needed_hh:]
                selected_middle = [candidate_indices[i] for i in topk_rel_idx]
            else:
                # Fallback: if no scores provided, retain the most recent candidates
                selected_middle = candidate_indices[-needed_hh:]

        kept = sorted(list(protected_indices) + selected_middle)
        return np.array(kept, dtype=np.int64)

    def prune(
        self,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
        layer_idx: int = 0,
        attn_scores: Optional[np.ndarray] = None,
        update_metrics: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prunes a single layer's (k, v) cache according to sink + window + H2O.

        Args:
            k_cache: [batch, heads, seq_len, head_dim]
            v_cache: [batch, heads, seq_len, head_dim]
            layer_idx: Layer index for attention accumulator.
            attn_scores: Optional attention scores.
            update_metrics: Whether to update pruner statistics.

        Returns:
            Tuple of pruned (k_cache, v_cache).
        """
        seq_len = k_cache.shape[2]
        if update_metrics:
            self.metrics.total_tokens_seen = max(self.metrics.total_tokens_seen, seq_len)

        if seq_len <= self.budget:
            if update_metrics:
                self.metrics.current_cache_size = seq_len
            return k_cache, v_cache

        keep_indices = self.compute_keep_indices(
            seq_len=seq_len,
            layer_idx=layer_idx,
            attn_scores=attn_scores,
        )

        # Slice along sequence dimension (axis=2)
        pruned_k = np.take(k_cache, keep_indices, axis=2)
        pruned_v = np.take(v_cache, keep_indices, axis=2)

        # Update accumulated attention scores for this layer to match new indices
        if len(self._accumulated_scores) > layer_idx:
            acc = self._accumulated_scores[layer_idx]
            if acc is not None:
                valid_indices = keep_indices[keep_indices < acc.shape[-1]]
                self._accumulated_scores[layer_idx] = np.take(acc, valid_indices, axis=-1)

        if update_metrics:
            evicted_count = seq_len - len(keep_indices)
            self.metrics.total_evicted += evicted_count
            self.metrics.current_cache_size = len(keep_indices)
            self.metrics.eviction_steps += 1

        return pruned_k, pruned_v

    def prune_layers(
        self,
        kv_caches: List[Tuple[np.ndarray, np.ndarray]],
        attn_scores: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply pruning across all layer KV caches."""
        if not kv_caches:
            return kv_caches

        seq_len = kv_caches[0][0].shape[2]
        self.metrics.total_tokens_seen = max(self.metrics.total_tokens_seen, seq_len)

        pruned = []
        for i, (k, v) in enumerate(kv_caches):
            scores_i = None
            if isinstance(attn_scores, list) and i < len(attn_scores):
                scores_i = attn_scores[i]
            elif isinstance(attn_scores, np.ndarray):
                scores_i = attn_scores

            # Only update global metrics on the first layer to avoid double counting
            pk, pv = self.prune(
                k,
                v,
                layer_idx=i,
                attn_scores=scores_i,
                update_metrics=(i == 0),
            )
            pruned.append((pk, pv))
        return pruned

