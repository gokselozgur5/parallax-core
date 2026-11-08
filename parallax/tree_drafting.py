"""Speculative Tree-Drafting (Eagle style) candidate verification kernel."""

from typing import List, Tuple
import numpy as np


class TreeDraftNode:
    """Represents a candidate token branch node in the speculative tree."""

    def __init__(self, token_id: int, parent_idx: int = -1) -> None:
        self.token_id = int(token_id)
        self.parent_idx = int(parent_idx)
