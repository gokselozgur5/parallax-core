from parallax.activation_beacon import ActivationBeaconCompressor


def test_activation_beacon_init() -> None:
    comp = ActivationBeaconCompressor(interval=32, beacon_tokens=2)
    assert comp.interval == 32
    assert comp.beacon_tokens == 2
