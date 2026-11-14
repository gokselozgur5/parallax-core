from typing import List, Optional, Tuple

import torch
from torch.nn.modules.linear import Linear as BaseLinear

class DispatchedLinear(BaseLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dispatch the linear operation to a specific backend
        return dispatched_matmul(x, self.weight.T, self.bias)

def dispatched_matmul(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Perform matrix multiplication using dispatching.

    Args:
    - x (torch.Tensor): Input tensor of shape (batch_size, in_features).
    - weight (torch.Tensor): Weight tensor of shape (out_features, in_features).
    - bias (Optional[torch.Tensor]): Bias tensor of shape (out_features) or None.

    Returns:
    - torch.Tensor: Output tensor of shape (batch_size, out_features).
    """
    # Implement the actual matrix multiplication logic here
    # For example, using PyTorch's matmul function
    return torch.matmul(x, weight) + bias if bias is not None else torch.matmul(x, weight)

def test_dispatched_linear():
    # Test cases to verify the correctness of DispatchedLinear
    x = torch.randn(10, 5)
    weight = torch.randn(3, 5)
    bias = torch.randn(3)

    dispatched_output = dispatched_matmul(x, weight, bias)
    nn_output = BaseLinear.apply(x, weight, bias)

    assert torch.allclose(dispatched_output, nn_output), "The dispatched linear operation did not match the nn.Linear output."

if __name__ == "__main__":
    test_dispatched_linear()
