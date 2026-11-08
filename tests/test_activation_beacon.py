from parallax.activation_beacon import ActivationBeaconCompressor


def test_activation_beacon_init() -> None:
    comp = ActivationBeaconCompressor(interval=32, beacon_tokens=2)
    assert comp.interval == 32
    assert comp.beacon_tokens == 2

import numpy as np


def test_condense_activations_shape_and_mean() -> None:
    comp = ActivationBeaconCompressor(interval=4)
    # 8 tokens of dim 2
    acts = np.ones((8, 2))
    acts[:4] *= 2.0
    acts[4:] *= 4.0
    beacons = comp.condense_activations(acts)
    assert beacons.shape == (2, 2)
    assert np.allclose(beacons[0], [2.0, 2.0])
    assert np.allclose(beacons[1], [4.0, 4.0])
