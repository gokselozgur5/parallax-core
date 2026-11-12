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


def test_q4_k_and_q6_k_parity_and_streaming_header(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    raw4 = rng.integers(0, 256, size=2 * BLOCK_BYTES_Q4_K, dtype=np.uint8)
    raw4.reshape(2, BLOCK_BYTES_Q4_K)[:, 0:4] = np.array([0x00, 0x3C, 0x00, 0x38], dtype=np.uint8)
    np.testing.assert_allclose(dequantize_q4_k(raw4.tobytes(), (2, 256)), _ref_q4_k(raw4.tobytes()).reshape(2, 256), atol=1e-6)

    raw6 = rng.integers(0, 256, size=2 * BLOCK_BYTES_Q6_K, dtype=np.uint8)
    raw6.reshape(2, BLOCK_BYTES_Q6_K)[:, 208:210] = np.array([0x00, 0x3C], dtype=np.uint8)
    q6 = dequantize_q6_k(raw6.tobytes(), (2, 256))
    assert q6.shape == (2, 256) and np.isfinite(q6).all()

    gguf_path = str(tmp_path / "kquant.gguf")
    save_gguf(
        gguf_path,
        {"general.architecture": "qwen3"},
        {"w4": raw4.tobytes(), "w6": raw6.tobytes()},
        tensor_types={"w4": GGML_TYPE_Q4_K, "w6": GGML_TYPE_Q6_K},
        tensor_shapes={"w4": (2, 256), "w6": (2, 256)},
    )
    hdr, base_off = read_gguf_header(gguf_path)
    assert hdr.tensors == {} and base_off > 0 and set(hdr.tensor_infos) == {"w4", "w6"}
    loaded = load_gguf(gguf_path, cast_float64=False)
    np.testing.assert_allclose(loaded.tensors["w4"], dequantize_q4_k(raw4.tobytes(), (2, 256)))
    np.testing.assert_allclose(loaded.tensors["w6"], q6)


def test_q6_k_matches_ggml_scalar_reference() -> None:
    rng = np.random.default_rng(77)
    blk = rng.integers(0, 256, size=BLOCK_BYTES_Q6_K, dtype=np.uint8)
    blk[208:210] = np.array([0x00, 0x38], dtype=np.uint8)
    d = float(blk[208:210].copy().view("<f2")[0])
    sc = blk[192:208].view(np.int8)
    ref = np.zeros(256, dtype=np.float32)
    for half in range(2):
        ql = blk[half * 64 : (half + 1) * 64]
        qh = blk[128 + half * 32 : 128 + (half + 1) * 32]
        s8 = sc[half * 8 : (half + 1) * 8]
        for l_idx in range(32):
            is_idx = l_idx // 16
            q1 = int(((ql[l_idx] & 0xF) | (((qh[l_idx] >> 0) & 3) << 4)).astype(np.int8)) - 32
            q2 = int(((ql[l_idx + 32] & 0xF) | (((qh[l_idx] >> 2) & 3) << 4)).astype(np.int8)) - 32
            q3 = int(((ql[l_idx] >> 4) | (((qh[l_idx] >> 4) & 3) << 4)).astype(np.int8)) - 32
            q4 = int(((ql[l_idx + 32] >> 4) | (((qh[l_idx] >> 6) & 3) << 4)).astype(np.int8)) - 32
            base = half * 128 + l_idx
            ref[base], ref[base + 32] = d * s8[is_idx] * q1, d * s8[is_idx + 2] * q2
            ref[base + 64], ref[base + 96] = d * s8[is_idx + 4] * q3, d * s8[is_idx + 6] * q4
    np.testing.assert_allclose(dequantize_q6_k(blk.tobytes(), (256,)), ref, atol=1e-6)
