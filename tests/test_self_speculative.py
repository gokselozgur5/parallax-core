from parallax.self_speculative import SelfSpeculativeEngine


def test_self_speculative_engine_init() -> None:
    engine = SelfSpeculativeEngine(exit_layer=2, total_layers=6)
    assert engine.exit_layer == 2
    assert engine.total_layers == 6

def test_draft_tokens_generation() -> None:
    engine = SelfSpeculativeEngine(exit_layer=2, total_layers=4)
    # Mock early-exit draft function that appends token + 1
    mock_draft = lambda seq: seq[-1] + 1
    drafts = engine.draft_tokens([10], mock_draft, num_draft=3)
    assert drafts == [11, 12, 13]
