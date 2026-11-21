import numpy as np
from parallax.core.parity import ParityChecker


class _NoGemm:
    name = "broken"


def test_missing_op_is_reported_in_error_message():
    a = np.ones((2, 2), dtype=np.float32)
    res = ParityChecker().check_op(_NoGemm(), "gemm", a @ a, a, a)
    assert res.status == "error"
    assert res.error_message == "missing op gemm"
    assert isinstance(res.max_rel_error, float)
