from parallax.grouped_gemm import GroupedGEMMDispatcher


def test_grouped_gemm_dispatcher_init() -> None:
    dispatcher = GroupedGEMMDispatcher(num_experts=8)
    assert dispatcher.num_experts == 8

import numpy as np


def test_group_tokens_partitioning() -> None:
    dispatcher = GroupedGEMMDispatcher(num_experts=3)
    tokens = np.array([[1.0], [2.0], [3.0], [4.0]])
    assignments = [0, 1, 0, 2]
    batches, indices = dispatcher.group_tokens(tokens, assignments)
    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert indices[0] == [0, 2]
    assert indices[1] == [1]
    assert indices[2] == [3]

def test_grouped_gemm_forward_computation() -> None:
    dispatcher = GroupedGEMMDispatcher(num_experts=2)
    tokens = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    assignments = [0, 1, 0]
    w0 = np.array([[2.0], [2.0]])
    w1 = np.array([[5.0], [5.0]])
    out = dispatcher.grouped_gemm_forward(tokens, assignments, [w0, w1])
    assert out.shape == (3, 1)
    # Token 0 -> exp 0: [1, 0] @ [2, 2] = 2
    assert out[0, 0] == 2.0
    # Token 1 -> exp 1: [0, 2] @ [5, 5] = 10
    assert out[1, 0] == 10.0
    # Token 2 -> exp 0: [3, 1] @ [2, 2] = 8
    assert out[2, 0] == 8.0
