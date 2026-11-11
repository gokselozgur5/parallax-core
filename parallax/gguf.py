"""
Zero-dependency GGUF (v2/v3) Binary Reader and Writer.

Implements the official GGUF binary format specification (used by llama.cpp and Ollama)
using only the Python standard library (`struct`) and NumPy:
- Magic bytes verification (`GGUF` / 0x46554747) and version header (v2/v3)
- Full key-value metadata parsing supporting all GGUF primitive and array types
- Tensor descriptor extraction with 32-byte alignment
- Pure NumPy tensor data slicing and dequantization for F32, F16, and BF16
"""

from dataclasses import dataclass, field
import math
import os
import struct
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import numpy as np

from parallax.gguf_quant import (
    BLOCK_BYTES_Q4_0,
    BLOCK_BYTES_Q8_0,
    QK4_0,
    QK8_0,
    dequantize_q4_0,
    dequantize_q8_0,
)


GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
MAX_GGUF_HEADER_SIZE: int = 100 * 1024 * 1024  # 100MB limit for header/metadata to prevent DoS
MAX_GGUF_STRING_LEN: int = 64 * 1024 * 1024  # 64MB limit for any single string
MAX_GGUF_ARRAY_LEN: int = 10_000_000  # 10M elements limit for array metadata
MAX_GGUF_ARRAY_DEPTH: int = 16  # Maximum recursion depth for nested arrays


def _read_exact(f: BinaryIO, n: int, desc: str = "payload") -> bytes:
    raw = f.read(n)
    if len(raw) < n:
        raise ValueError(f"Unexpected EOF reading GGUF {desc} (expected {n} bytes, got {len(raw)})")
    return raw

# GGUF Metadata Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_BF16 = 26

# LLAMA / GGUF General File Types (llama_ftype)
LLAMA_FTYPE_ALL_F32 = 0
LLAMA_FTYPE_MOSTLY_F16 = 1
LLAMA_FTYPE_MOSTLY_Q4_0 = 2
LLAMA_FTYPE_MOSTLY_Q4_1 = 3
LLAMA_FTYPE_MOSTLY_Q8_0 = 7


@dataclass
class GGUFTensorInfo:
    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass
class GGUFFile:
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: Dict[str, np.ndarray] = field(default_factory=dict)
    tensor_infos: Dict[str, GGUFTensorInfo] = field(default_factory=dict)


def _read_string(f: BinaryIO) -> str:
    len_bytes = _read_exact(f, 8, "string length")
    str_len = struct.unpack("<Q", len_bytes)[0]
    if str_len > MAX_GGUF_STRING_LEN:
        raise ValueError(f"GGUF string length {str_len} exceeds maximum allowed limit {MAX_GGUF_STRING_LEN}")
    raw = _read_exact(f, str_len, "string payload")
    return raw.decode("utf-8", errors="replace")


def _write_string(f: BinaryIO, s: str) -> None:
    raw = s.encode("utf-8")
    f.write(struct.pack("<Q", len(raw)))
    f.write(raw)


def _read_metadata_value(
    f: BinaryIO,
    val_type: int,
    depth: int = 0,
    max_header_size: int = MAX_GGUF_HEADER_SIZE,
) -> Any:
    if f.tell() > max_header_size:
        raise ValueError("GGUF header/metadata exceeded maximum allowed size limit (DoS protection)")
    if val_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", _read_exact(f, 1, "uint8"))[0]
    elif val_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", _read_exact(f, 1, "int8"))[0]
    elif val_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", _read_exact(f, 2, "uint16"))[0]
    elif val_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", _read_exact(f, 2, "int16"))[0]
    elif val_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", _read_exact(f, 4, "uint32"))[0]
    elif val_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", _read_exact(f, 4, "int32"))[0]
    elif val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", _read_exact(f, 4, "float32"))[0]
    elif val_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", _read_exact(f, 1, "bool"))[0])
    elif val_type == GGUF_TYPE_STRING:
        return _read_string(f)
    elif val_type == GGUF_TYPE_ARRAY:
        if depth >= MAX_GGUF_ARRAY_DEPTH:
            raise ValueError(f"GGUF array nesting depth exceeds maximum limit {MAX_GGUF_ARRAY_DEPTH}")
        hdr_bytes = _read_exact(f, 12, "array metadata header")
        elem_type, elem_count = struct.unpack("<IQ", hdr_bytes)
        if elem_count > MAX_GGUF_ARRAY_LEN:
            raise ValueError(f"GGUF array element count {elem_count} exceeds maximum limit {MAX_GGUF_ARRAY_LEN}")
        return [
            _read_metadata_value(f, elem_type, depth=depth + 1, max_header_size=max_header_size)
            for _ in range(elem_count)
        ]
    elif val_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", _read_exact(f, 8, "uint64"))[0]
    elif val_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", _read_exact(f, 8, "int64"))[0]
    elif val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", _read_exact(f, 8, "float64"))[0]
    else:
        raise ValueError(f"Unknown GGUF metadata value type: {val_type}")


def _write_metadata_value(f: BinaryIO, val: Any) -> None:
    if isinstance(val, bool):
        f.write(struct.pack("<IB", GGUF_TYPE_BOOL, 1 if val else 0))
    elif isinstance(val, int):
        if -2147483648 <= val <= 2147483647:
            f.write(struct.pack("<Ii", GGUF_TYPE_INT32, val))
        else:
            f.write(struct.pack("<Iq", GGUF_TYPE_INT64, val))
    elif isinstance(val, float):
        f.write(struct.pack("<Id", GGUF_TYPE_FLOAT64, val))
    elif isinstance(val, str):
        f.write(struct.pack("<I", GGUF_TYPE_STRING))
        _write_string(f, val)
    elif isinstance(val, list):
        f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
        if not val:
            f.write(struct.pack("<IQ", GGUF_TYPE_STRING, 0))
        elif isinstance(val[0], str):
            f.write(struct.pack("<IQ", GGUF_TYPE_STRING, len(val)))
            for s in val:
                _write_string(f, s)
        elif isinstance(val[0], int):
            f.write(struct.pack("<IQ", GGUF_TYPE_INT32, len(val)))
            for item in val:
                f.write(struct.pack("<i", item))
        elif isinstance(val[0], float):
            f.write(struct.pack("<IQ", GGUF_TYPE_FLOAT32, len(val)))
            for item in val:
                f.write(struct.pack("<f", item))
        else:
            raise TypeError(f"Unsupported array element type: {type(val[0])}")
    else:
        raise TypeError(f"Unsupported metadata value type: {type(val)}")


def load_gguf(file_path: str, cast_float64: bool = True) -> GGUFFile:
    """Parses a GGUF binary file into a GGUFFile container."""
    with open(file_path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic header: expected {GGUF_MAGIC}, got {magic}")

        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version} (expected v2 or v3)")

        hdr_info = f.read(16)
        if len(hdr_info) < 16:
            raise ValueError("Truncated GGUF file header: unable to read tensor and metadata counts")
        tensor_count, metadata_kv_count = struct.unpack("<QQ", hdr_info)
        if tensor_count > 1_000_000 or metadata_kv_count > 1_000_000:
            raise ValueError(f"Excessive GGUF counts (tensor_count={tensor_count}, metadata_kv_count={metadata_kv_count}) exceeds DoS threshold")

        f.seek(0, os.SEEK_END)
        total_file_size = f.tell()
        f.seek(24)  # 4 magic + 4 version + 8 tensor_count + 8 metadata_kv_count

        # Parse key-value metadata
        metadata: Dict[str, Any] = {}
        for _ in range(metadata_kv_count):
            if f.tell() > MAX_GGUF_HEADER_SIZE or f.tell() > total_file_size:
                raise ValueError("GGUF header/metadata exceeded maximum allowed size limit (DoS protection)")
            key = _read_string(f)
            type_bytes = f.read(4)
            if len(type_bytes) < 4:
                raise ValueError("Unexpected EOF reading GGUF metadata value type")
            val_type = struct.unpack("<I", type_bytes)[0]
            val = _read_metadata_value(f, val_type)
            metadata[key] = val

        alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError(f"Invalid GGUF alignment: {alignment} must be a positive power of 2")

        # Parse tensor descriptors
        tensor_infos: Dict[str, GGUFTensorInfo] = {}
        for _ in range(tensor_count):
            if f.tell() > MAX_GGUF_HEADER_SIZE or f.tell() > total_file_size:
                raise ValueError("GGUF header/descriptors exceeded maximum allowed size limit (DoS protection)")
            name = _read_string(f)
            ndims_bytes = f.read(4)
            if len(ndims_bytes) < 4:
                raise ValueError(f"Unexpected EOF reading dimension count for tensor '{name}'")
            n_dims = struct.unpack("<I", ndims_bytes)[0]
            if n_dims > 16:
                raise ValueError(f"Unreasonable tensor rank {n_dims} for tensor '{name}' (DoS protection)")

            dims = []
            for _ in range(n_dims):
                dim_bytes = f.read(8)
                if len(dim_bytes) < 8:
                    raise ValueError(f"Unexpected EOF reading dimensions for tensor '{name}'")
                dim = struct.unpack("<Q", dim_bytes)[0]
                dims.append(dim)

            # In GGUF, dimensions are stored inner-to-outer; reverse for standard NumPy row-major shape
            shape = tuple(reversed(dims))
            desc_tail = f.read(12)
            if len(desc_tail) < 12:
                raise ValueError(f"Unexpected EOF reading tensor type and offset for '{name}'")
            ggml_type, offset = struct.unpack("<IQ", desc_tail)
            tensor_infos[name] = GGUFTensorInfo(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                offset=offset,
            )

        # Calculate tensor data payload start offset with alignment
        header_end = f.tell()
        padding = (alignment - (header_end % alignment)) % alignment
        data_base_offset = header_end + padding

        # Extract tensors
        tensors: Dict[str, np.ndarray] = {}
        for name, info in tensor_infos.items():
            if any(d < 0 for d in info.shape):
                raise ValueError(f"Negative dimension in shape {info.shape} for tensor '{name}'")
            try:
                num_elements = math.prod(info.shape) if info.shape else 1
            except OverflowError as e:
                raise ValueError(f"Integer overflow calculating element count for tensor '{name}': {e}") from e

            if info.ggml_type == GGML_TYPE_F32:
                byte_size = num_elements * 4
            elif info.ggml_type == GGML_TYPE_F16 or info.ggml_type == GGML_TYPE_BF16:
                byte_size = num_elements * 2
            elif info.ggml_type == GGML_TYPE_Q8_0:
                n_blocks = (num_elements + QK8_0 - 1) // QK8_0
                byte_size = n_blocks * BLOCK_BYTES_Q8_0
            elif info.ggml_type == GGML_TYPE_Q4_0:
                n_blocks = (num_elements + QK4_0 - 1) // QK4_0
                byte_size = n_blocks * BLOCK_BYTES_Q4_0
            else:
                raise NotImplementedError(f"Unsupported GGML tensor type: {info.ggml_type}")

            tensor_start_offset = data_base_offset + info.offset
            if tensor_start_offset < 0 or byte_size < 0 or tensor_start_offset + byte_size > total_file_size:
                raise ValueError("Invalid tensor offset/length: exceeds file bounds")

            f.seek(tensor_start_offset)
            raw_bytes = f.read(byte_size)
            if len(raw_bytes) < byte_size:
                raise ValueError("Invalid tensor offset/length: exceeds file bounds")

            if info.ggml_type == GGML_TYPE_F32:
                arr = np.frombuffer(raw_bytes, dtype="<f4").reshape(info.shape).copy()
            elif info.ggml_type == GGML_TYPE_F16:
                arr = np.frombuffer(raw_bytes, dtype="<f2").reshape(info.shape).copy()
            elif info.ggml_type == GGML_TYPE_BF16:
                u16 = np.frombuffer(raw_bytes, dtype="<u2")
                u32 = u16.astype(np.uint32) << 16
                arr = u32.view(np.float32).reshape(info.shape).copy()
            elif info.ggml_type == GGML_TYPE_Q8_0:
                arr = dequantize_q8_0(raw_bytes, info.shape)
            elif info.ggml_type == GGML_TYPE_Q4_0:
                arr = dequantize_q4_0(raw_bytes, info.shape)

            if cast_float64 and np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64)

            tensors[name] = arr

        return GGUFFile(
            version=version,
            metadata=metadata,
            tensors=tensors,
            tensor_infos=tensor_infos,
        )


def save_gguf(
    file_path: str,
    metadata: Dict[str, Any],
    tensors: Dict[str, Union[np.ndarray, bytes]],
    version: int = 3,
    alignment: int = GGUF_DEFAULT_ALIGNMENT,
    tensor_types: Optional[Dict[str, int]] = None,
    tensor_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> None:
    """Serializes metadata and tensors into an official GGUF v3 binary file."""
    with open(file_path, "wb") as f:
        # Magic & Header
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", version, len(tensors), len(metadata)))

        # Metadata
        for k, v in metadata.items():
            _write_string(f, k)
            _write_metadata_value(f, v)

        # Prepare tensor binary chunks and descriptors
        tensor_chunks: List[bytes] = []
        tensor_descriptors: List[Tuple[str, Tuple[int, ...], int, int]] = []
        current_offset = 0

        for name, arr in tensors.items():
            if isinstance(arr, (bytes, bytearray)):
                raw_bytes = bytes(arr)
                ggml_type = tensor_types.get(name, GGML_TYPE_Q8_0) if tensor_types else GGML_TYPE_Q8_0
                shape = tensor_shapes.get(name, (len(raw_bytes),)) if tensor_shapes else (len(raw_bytes),)
            else:
                arr_contig = np.ascontiguousarray(arr)
                dtype_name = arr_contig.dtype.name

                if dtype_name == "float32":
                    ggml_type = GGML_TYPE_F32
                    raw_bytes = arr_contig.astype("<f4").tobytes()
                    shape = arr_contig.shape
                elif dtype_name == "float16":
                    ggml_type = GGML_TYPE_F16
                    raw_bytes = arr_contig.astype("<f2").tobytes()
                    shape = arr_contig.shape
                elif dtype_name in ("float64",):
                    # Standardize float64 to float32 for GGUF serialization
                    ggml_type = GGML_TYPE_F32
                    raw_bytes = arr_contig.astype("<f4").tobytes()
                    shape = arr_contig.shape
                else:
                    raise TypeError(f"Unsupported dtype for GGUF serialization: {dtype_name} on tensor '{name}'")



            # Offset must be aligned to alignment boundary
            pad = (alignment - (current_offset % alignment)) % alignment
            current_offset += pad
            if pad > 0:
                tensor_chunks.append(b"\x00" * pad)

            tensor_descriptors.append((name, shape, ggml_type, current_offset))
            tensor_chunks.append(raw_bytes)
            current_offset += len(raw_bytes)


        # Write tensor descriptors
        for name, shape, ggml_type, offset in tensor_descriptors:
            _write_string(f, name)
            # In GGUF, dimensions are written in reverse order (inner to outer)
            f.write(struct.pack("<I", len(shape)))
            for dim in reversed(shape):
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<IQ", ggml_type, offset))

        # Pad header to alignment boundary
        header_len = f.tell()
        pad = (alignment - (header_len % alignment)) % alignment
        if pad > 0:
            f.write(b"\x00" * pad)

        # Write tensor data
        for chunk in tensor_chunks:
            f.write(chunk)
