"""
Kahneman-Tversky Optimization (KTO: Ethayarajh et al., 2024) from first principles.
"""

from typing import Optional, Tuple
import numpy as np


def kto_value_function(
    z: np.ndarray,
    is_desirable: bool = True,
    lambda_d: float = 1.0,
    lambda_u: float = 1.33,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prospect Theory value function v(z) and its analytical derivative dv/dz."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

    if is_desirable:
        # v(z) = 1 - sigma(z) = sigma(-z)
        loss = lambda_d * (1.0 - sig)
        # d(loss)/dz = - lambda_d * sig * (1 - sig)
        dloss_dz = - lambda_d * sig * (1.0 - sig)
    else:
        # v(z) = 1 - sigma(-z) = sigma(z)
        loss = lambda_u * sig
        # d(loss)/dz = lambda_u * sig * (1 - sig)
        dloss_dz = lambda_u * sig * (1.0 - sig)

    return loss, dloss_dz
