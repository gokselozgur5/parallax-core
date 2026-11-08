"""FP8 (E4M3 and E5M2) quantization and micro-scaling emulation."""

from typing import Tuple
import numpy as np


class FP8Format:
    """Defines floating-point bit layout and dynamic range limits."""

    E4M3_MAX = 448.0
    E5M2_MAX = 57344.0
