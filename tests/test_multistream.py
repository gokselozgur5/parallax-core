from parallax.multistream import SpeculativeStreamBranch


def test_speculative_stream_branch_init():
    b = SpeculativeStreamBranch(branch_id=1, tokens=[10, 20, 30], cumulative_score=-1.5)
    assert b.branch_id == 1
    assert b.tokens == [10, 20, 30]
    assert b.cumulative_score == -1.5
