from parallax.soft_capping import LogitSoftCapper


def test_logit_soft_capper_init() -> None:
    capper = LogitSoftCapper(soft_cap=50.0)
    assert capper.soft_cap == 50.0
