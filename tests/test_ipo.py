from parallax.ipo import IPOConfig


def test_ipo_config_target_margin() -> None:
    config = IPOConfig(tau=0.2)
    assert config.tau == 0.2
    assert config.target_margin == 2.5
