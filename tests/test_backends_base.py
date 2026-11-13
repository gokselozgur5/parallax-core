import numpy as np
import pytest

from parallax.backends.base import Backend, check_backend_contract


class StubBackend(Backend):
    name = "stub"

    def available(self) -> bool:
        return True

    def supports(self, op, dtype, shape) -> bool:
        return op == "gemm" and np.dtype(dtype) == np.float32 and len(shape) == 2


def test_backend_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Backend()  # type: ignore[abstract]


def test_incomplete_subclass_raises():
    class OnlyAvailable(Backend):
        name = "partial"

        def available(self) -> bool:
            return True

    with pytest.raises(TypeError):
        OnlyAvailable()  # type: ignore[abstract]
