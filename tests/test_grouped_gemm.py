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
