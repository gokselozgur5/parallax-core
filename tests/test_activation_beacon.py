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

def test_compress_kv_cache_shapes() -> None:
    comp = ActivationBeaconCompressor(interval=8)
    past_k = np.random.randn(24, 16)
    past_v = np.random.randn(24, 16)
    beacon_k, beacon_v = comp.compress_kv_cache(past_k, past_v)
    # 24 tokens / 8 interval = 3 beacon tokens
    assert beacon_k.shape == (3, 16)
    assert beacon_v.shape == (3, 16)


def test_condense_preserves_remainder_tokens() -> None:
    comp = ActivationBeaconCompressor(interval=16, beacon_tokens=1)
    # 20 tokens: 1 interval + 4 remainder
    acts20 = np.random.randn(20, 8)
    condensed20 = comp.condense_activations(acts20)
    assert condensed20.shape == (5, 8)
    np.testing.assert_array_equal(condensed20[1:], acts20[16:])

    # 10 tokens (< interval): all 10 kept
    acts10 = np.random.randn(10, 8)
    condensed10 = comp.condense_activations(acts10)
    assert condensed10.shape == (10, 8)
    np.testing.assert_array_equal(condensed10, acts10)


def test_condense_multiple_beacon_tokens_and_batch_dims() -> None:
    comp = ActivationBeaconCompressor(interval=16, beacon_tokens=2)
    # Shape [2, 3, 35, 8]: 2 full intervals (2*2 = 4 beacons) + 3 remainder tokens -> 7 tokens
    acts = np.random.randn(2, 3, 35, 8)
    condensed = comp.condense_activations(acts)
    assert condensed.shape == (2, 3, 7, 8)
    np.testing.assert_array_equal(condensed[:, :, 4:, :], acts[:, :, 32:, :])

