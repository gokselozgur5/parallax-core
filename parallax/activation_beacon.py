"""Activation Beacon long-context memory condensation kernel."""

from typing import List, Tuple
import numpy as np


class ActivationBeaconCompressor:
    """Condenses past sequence activation windows into compact beacon representations."""

    def __init__(self, interval: int = 16, beacon_tokens: int = 1) -> None:
        self.interval = int(interval)
        self.beacon_tokens = int(beacon_tokens)

    def condense_activations(self, activations: np.ndarray) -> np.ndarray:
        """Condense long sequence activations into interval mean beacon vectors."""
        seq_len, dim = activations.shape
        num_beacons = seq_len // self.interval
        if num_beacons == 0:
            return np.zeros((0, dim))
        trimmed = activations[:num_beacons * self.interval]
        windows = trimmed.reshape(num_beacons, self.interval, dim)
        return np.mean(windows, axis=1)
