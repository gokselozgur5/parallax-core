from parallax.multistream import SpeculativeStreamBranch


def test_speculative_stream_branch_init():
    b = SpeculativeStreamBranch(branch_id=1, tokens=[10, 20, 30], cumulative_score=-1.5)
    assert b.branch_id == 1
    assert b.tokens == [10, 20, 30]
    assert b.cumulative_score == -1.5


from parallax.multistream import verify_stream_branches


def test_verify_stream_branches_selects_longest_match():
    # Sequence oracle where next token is len(seq)
    def oracle(seq):
        return len(seq)

    prefix = [0, 1]  # Next token must be 2, then 3, then 4

    b1 = SpeculativeStreamBranch(branch_id=1, tokens=[2, 99])  # Accepts 1 token [2]
    b2 = SpeculativeStreamBranch(branch_id=2, tokens=[2, 3, 4])  # Accepts all 3 tokens [2, 3, 4]
    b3 = SpeculativeStreamBranch(branch_id=3, tokens=[99, 99])  # Accepts 0 tokens

    best_id, accepted, next_tok = verify_stream_branches([b1, b2, b3], oracle, prefix)

    assert best_id == 2
    assert accepted == [2, 3, 4]
    assert next_tok == 5  # Bonus token after [0, 1, 2, 3, 4] is len(seq) = 5
