#!/usr/bin/env python3
"""
CLI tool for quantizing ParallaxLM GGUF models.

Supports quantizing uncompressed F32 model tensors to Q8_0 or Q4_0 formats.
"""

import argparse
import os
import sys
from typing import Dict
import numpy as np
from parallax.gguf import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUFFile,
    GGUFTensorInfo,
    load_gguf,
    save_gguf,
)
from parallax.gguf_quant import quantize_q4_0, quantize_q8_0


def quantize_model(input_path: str, output_path: str, quant_type: str) -> None:
    """Quantizes an existing GGUF model file into Q8_0 or Q4_0 format."""
    print(f"Loading {input_path}...")
    gguf_in = load_gguf(input_path)

    quant_type_upper = quant_type.upper()
    if quant_type_upper == "Q8_0":
        target_type = GGML_TYPE_Q8_0
        quant_fn = quantize_q8_0
    elif quant_type_upper == "Q4_0":
        target_type = GGML_TYPE_Q4_0
        quant_fn = quantize_q4_0
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    quantized_tensors: Dict[str, np.ndarray] = {}
    new_tensor_infos = []

    for info in gguf_in.tensor_infos:
        arr = gguf_in.tensors[info.name]
        # Only quantize 2D weight matrices whose trailing dimension is divisible by block size 32
        if arr.ndim >= 2 and arr.shape[-1] % 32 == 0 and "norm" not in info.name:
            q_arr = quant_fn(arr)
            quantized_tensors[info.name] = q_arr
            new_tensor_infos.append(
                GGUFTensorInfo(
                    name=info.name,
                    shape=info.shape,
                    dtype=target_type,
                    offset=0,
                )
            )
        else:
            quantized_tensors[info.name] = arr
            new_tensor_infos.append(
                GGUFTensorInfo(
                    name=info.name,
                    shape=info.shape,
                    dtype=info.dtype,
                    offset=0,
                )
            )

    metadata = dict(gguf_in.metadata)
    metadata["general.quantization_version"] = 2
    metadata["general.file_type"] = target_type

    print(f"Writing quantized model to {output_path} ({quant_type_upper})...")
    save_gguf(output_path, metadata, quantized_tensors, new_tensor_infos)
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize GGUF models to Q8_0 or Q4_0.")
    parser.add_argument("input", help="Path to input GGUF file")
    parser.add_argument("output", help="Path to output GGUF file")
    parser.add_argument("--type", choices=["q8_0", "q4_0"], default="q8_0", help="Quantization format")

    args = parser.parse_args()
    quantize_model(args.input, args.output, args.type)


if __name__ == "__main__":
    main()
