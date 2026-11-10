"""
Zero-dependency SafeTensors reader and writer.

Implements the HuggingFace safetensors binary serialization specification
using only the Python standard library (`struct`, `json`) and NumPy:
- 8-byte unsigned 64-bit little-endian integer for header byte size
- UTF-8 encoded JSON header with tensor metadata and byte offsets
- Contiguous little-endian binary tensor payload
- Native support for F64, F32, F16, BF16 (bfloat16 bit-shift decoding),
  I64, I32, I16, I8, U8, and BOOL
"""

import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


NUMPY_TO_SAFETENSORS_DTYPE: Dict[str, str] = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}

SAFETENSORS_TO_NUMPY_DTYPE: Dict[str, str] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "?",
}


def save_safetensors(
    file_path: str,
    tensors: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves a dictionary of NumPy arrays to a SafeTensors binary file.

    Args:
        file_path: Output file path.
        tensors: Dictionary mapping tensor names to NumPy arrays.
        metadata: Optional metadata string dictionary stored in __metadata__.
    """
    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    data_chunks: List[bytes] = []
    current_offset = 0

    for name, arr in tensors.items():
        arr_contiguous = np.ascontiguousarray(arr)
        dtype_name = arr_contiguous.dtype.name

        if dtype_name in NUMPY_TO_SAFETENSORS_DTYPE:
            sf_dtype = NUMPY_TO_SAFETENSORS_DTYPE[dtype_name]
        else:
            raise TypeError(f"Unsupported NumPy dtype for safetensors: {dtype_name} on tensor '{name}'")

        raw_bytes = arr_contiguous.tobytes()
        num_bytes = len(raw_bytes)
        end_offset = current_offset + num_bytes

        header[name] = {
            "dtype": sf_dtype,
            "shape": list(arr_contiguous.shape),
            "data_offsets": [current_offset, end_offset],
        }

        data_chunks.append(raw_bytes)
        current_offset = end_offset

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)

    with open(file_path, "wb") as f:
        # 8-byte unsigned 64-bit integer (little endian)
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for chunk in data_chunks:
            f.write(chunk)


def load_safetensors(
    file_path: str,
    cast_float64: bool = True,
) -> Dict[str, np.ndarray]:
    """Loads tensors from a SafeTensors binary file.

    Args:
        file_path: Path to the .safetensors file.
        cast_float64: If True, automatically casts float16, bfloat16, and float32
                      tensors to float64 for compatibility with Parallax autograd.

    Returns:
        Dictionary mapping tensor names to NumPy arrays.
    """
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Invalid safetensors file: {file_path} is smaller than 8 bytes")

        header_len = struct.unpack("<Q", header_len_bytes)[0]
        f.seek(0, os.SEEK_END)
        total_file_size = f.tell()
        f.seek(8)

        if header_len > total_file_size - 8:
            raise ValueError(f"Truncated header in safetensors file: {file_path}")

        header_json_bytes = f.read(header_len)
        if len(header_json_bytes) < header_len:
            raise ValueError(f"Truncated header in safetensors file: {file_path}")

        header = json.loads(header_json_bytes.decode("utf-8"))
        data_base_offset = 8 + header_len

        tensors: Dict[str, np.ndarray] = {}

        for key, info in header.items():
            if key == "__metadata__":
                continue

            dtype_str = info["dtype"]
            shape = tuple(info["shape"])
            start_off, end_off = info["data_offsets"]
            byte_len = end_off - start_off

            tensor_start_offset = data_base_offset + start_off
            if start_off < 0 or byte_len < 0 or tensor_start_offset + byte_len > total_file_size:
                raise ValueError("Invalid tensor offset/length: exceeds file bounds")

            f.seek(tensor_start_offset)
            raw_data = f.read(byte_len)
            if len(raw_data) < byte_len:
                raise ValueError(f"Truncated tensor payload for '{key}' in {file_path}")

            if dtype_str == "BF16":
                # Convert bfloat16 to float32 via uint16 -> uint32 bit shift
                u16 = np.frombuffer(raw_data, dtype="<u2")
                u32 = u16.astype(np.uint32) << 16
                arr = u32.view(np.float32).reshape(shape).copy()
            elif dtype_str in SAFETENSORS_TO_NUMPY_DTYPE:
                np_dtype = SAFETENSORS_TO_NUMPY_DTYPE[dtype_str]
                arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(shape).copy()
            else:
                raise ValueError(f"Unsupported safetensors dtype '{dtype_str}' for tensor '{key}'")

            if cast_float64 and np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float64)

            tensors[key] = arr

        return tensors
