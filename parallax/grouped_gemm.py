"""Grouped GEMM kernel for contiguous expert dispatch."""

from typing import List, Tuple
import numpy as np


class GroupedGEMMDispatcher:
    """Manages contiguous activation grouping and GEMM dispatch across MoE experts."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = int(num_experts)
