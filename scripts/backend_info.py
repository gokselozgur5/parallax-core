"""Command-line utility to display Parallax backend availability and capabilities."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from parallax.backends import cpuinfo, registry
from parallax.backends.base import Backend
from parallax.backends.c_backend import CBackend
from parallax.backends.metal_backend import MetalBackend
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.report import generate_markdown_report
from parallax.backends.rust_backend import RustBackend

