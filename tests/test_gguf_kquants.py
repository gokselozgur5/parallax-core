"""Bit-exact reference parity tests for GGUF Q4_K, Q6_K, and streaming header."""

from pathlib import Path
import numpy as np

from parallax.gguf import GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, load_gguf, read_gguf_header, save_gguf
from parallax.gguf_quant import BLOCK_BYTES_Q4_K, BLOCK_BYTES_Q6_K, dequantize_q4_k, dequantize_q6_k


def _ref_q4_k(raw: bytes) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLOCK_BYTES_Q4_K)
    out = []
    for blk in buf:
        d = float(blk[0:2].copy().view("<f2")[0])
        dmin = float(blk[2:4].copy().view("<f2")[0])
        q = blk[4:16]
        qs = blk[16:144]
        for g in range(4):
            for k, shift in enumerate((0, 4)):
                j = 2 * g + k
                sc = int(q[j] & 63) if j < 4 else int((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4))
                mn = int(q[j + 4] & 63) if j < 4 else int((q[j + 4] >> 4) | ((q[j] >> 6) << 4))
                vals = ((qs[g * 32 : (g + 1) * 32] >> shift) & 0xF).astype(np.float32)
                out.append(d * sc * vals - dmin * mn)
    return np.concatenate(out)
