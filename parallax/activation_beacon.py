"""Activation Beacon long-context memory condensation kernel."""

from typing import List, Tuple
import numpy as np


class ActivationBeaconCompressor:
    """Condenses past sequence activation windows into compact beacon representations."""

    def __init__(self, interval: int = 16, beacon_tokens: int = 1) -> None:
        self.interval = int(interval)
        self.beacon_tokens = int(beacon_tokens)
