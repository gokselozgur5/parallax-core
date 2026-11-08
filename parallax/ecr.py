"""Expert Choice Routing (ECR) for load-balanced Mixture of Experts."""

from typing import List, Tuple
import numpy as np


class ExpertChoiceRouter:
    """Routes tokens where experts select top tokens ensuring perfect balance."""

    def __init__(self, num_experts: int, capacity_factor: float = 1.0) -> None:
        self.num_experts = int(num_experts)
        self.capacity_factor = float(capacity_factor)
