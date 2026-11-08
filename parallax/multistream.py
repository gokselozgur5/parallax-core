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



def verify_stream_branches(
    branches: List[SpeculativeStreamBranch],
    oracle_fn: Callable[[List[int]], int],
    prefix: List[int],
) -> Tuple[int, List[int], int]:
    """Verify multiple speculative draft branches and select the longest accepted prefix.

    Args:
        branches: Candidate draft branches.
        oracle_fn: Target model next-token prediction callable.
        prefix: Current accepted sequence prefix.

    Returns:
        Tuple of:
            - best_branch_id: ID of the selected branch.
            - accepted_tokens: Tokens accepted from that branch.
            - next_oracle_token: The target model's correction or follow-up token.
    """
    best_branch_id = -1
    best_accepted: List[int] = []
    best_next_tok = -1

    for branch in branches:
        current_seq = list(prefix)
        accepted = []
        next_tok = -1

        for tok in branch.tokens:
            expected = oracle_fn(current_seq)
            if tok == expected:
                accepted.append(tok)
                current_seq.append(tok)
            else:
                next_tok = expected
                break

        if next_tok == -1:
            # All tokens in this branch accepted; sample bonus token
            next_tok = oracle_fn(current_seq)

        if len(accepted) > len(best_accepted) or best_branch_id == -1:
            best_branch_id = branch.branch_id
            best_accepted = accepted
            best_next_tok = next_tok

    return best_branch_id, best_accepted, best_next_tok
