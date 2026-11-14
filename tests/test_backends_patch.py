import pytest
from parallax.backends import registry
from parallax.backends.linear import DispatchedLinear
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.patch import disable_native, enable_native
from parallax.nn import Linear, Module
