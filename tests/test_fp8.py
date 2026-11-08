from parallax.fp8 import FP8Format


def test_fp8_format_constants() -> None:
    assert FP8Format.E4M3_MAX == 448.0
    assert FP8Format.E5M2_MAX == 57344.0
