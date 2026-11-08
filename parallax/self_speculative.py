"""Self-Speculative Decoding via layer-skipping and full verification."""

from typing import Callable, List, Tuple
import numpy as np


class SelfSpeculativeEngine:
    """Uses internal layer-skipping early exit to draft tokens without a separate model."""

    def __init__(self, exit_layer: int, total_layers: int) -> None:
        self.exit_layer = int(exit_layer)
        self.total_layers = int(total_layers)

    def draft_tokens(
        self,
        prompt_tokens: List[int],
        draft_fn: Callable[[List[int]], int],
        num_draft: int = 3,
    ) -> List[int]:
        """Generate speculative draft tokens using early-exit draft function."""
        drafts = []
        curr = list(prompt_tokens)
        for _ in range(num_draft):
            next_tok = draft_fn(curr)
            drafts.append(next_tok)
            curr.append(next_tok)
        return drafts
