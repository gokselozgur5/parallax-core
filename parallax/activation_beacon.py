"""Activation Beacon long-context memory condensation kernel."""

from typing import Tuple
import numpy as np


class ActivationBeaconCompressor:
    """Condenses past sequence activation windows into compact beacon representations."""

    def __init__(self, interval: int = 16, beacon_tokens: int = 1) -> None:
        self.interval = int(interval)
        self.beacon_tokens = int(beacon_tokens)
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")
        if self.beacon_tokens <= 0:
            raise ValueError(f"beacon_tokens must be positive, got {self.beacon_tokens}")

    def condense_activations(self, activations: np.ndarray) -> np.ndarray:
        """Condense long sequence activations into interval beacon vectors and keep remainder."""
        if activations.ndim < 2:
            raise ValueError(
                f"activations must have at least 2 dimensions, got shape {activations.shape}"
            )
        seq_len = activations.shape[-2]
        dim = activations.shape[-1]
        lead_shape = activations.shape[:-2]
        if seq_len == 0:
            return activations.copy()

        num_intervals = seq_len // self.interval
        parts = []
        if num_intervals > 0:
            full_len = num_intervals * self.interval
            full_acts = activations[..., :full_len, :]
            if self.interval % self.beacon_tokens == 0:
                step = self.interval // self.beacon_tokens
                chunked = full_acts.reshape(*lead_shape, num_intervals, self.beacon_tokens, step, dim)
                beacons = np.mean(chunked, axis=-2).reshape(
                    *lead_shape, num_intervals * self.beacon_tokens, dim
                )
            else:
                chunks = np.array_split(full_acts, num_intervals * self.beacon_tokens, axis=-2)
                beacons = np.concatenate(
                    [np.mean(c, axis=-2, keepdims=True) for c in chunks], axis=-2
                )
            parts.append(beacons)

        rem_len = seq_len - num_intervals * self.interval
        if rem_len > 0:
            rem_acts = activations[..., num_intervals * self.interval :, :]
            parts.append(rem_acts)

        if parts:
            return np.concatenate(parts, axis=-2)
        return activations.copy()

    def compress_kv_cache(
        self,
        past_k: np.ndarray,
        past_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compress old KV cache into condensed beacon KV representations."""
        beacon_k = self.condense_activations(past_k)
        beacon_v = self.condense_activations(past_v)
        return beacon_k, beacon_v
