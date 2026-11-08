"""Multi-stream speculative decoding with dynamic branch verification.

Generates multiple draft token continuation streams in parallel and selects the
longest accepted branch against the target model distribution.
"""

from typing import List, Tuple, Callable
import numpy as np


class SpeculativeStreamBranch:
    """Represents a speculative candidate rollout stream."""

    def __init__(self, branch_id: int, tokens: List[int], cumulative_score: float = 0.0):
        self.branch_id = branch_id
        self.tokens = tokens
        self.cumulative_score = cumulative_score
