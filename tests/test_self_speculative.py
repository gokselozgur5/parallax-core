from parallax.self_speculative import SelfSpeculativeEngine


def test_self_speculative_engine_init() -> None:
    engine = SelfSpeculativeEngine(exit_layer=2, total_layers=6)
    assert engine.exit_layer == 2
    assert engine.total_layers == 6
