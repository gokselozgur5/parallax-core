"""
Zero-dependency GGUF Tokenizer Extractor and Decoder.

Extracts BPE/SPM token dictionaries, byte fallbacks, and merge tables directly
from GGUF metadata headers without requiring sentencepiece, protobuf, or huggingface_hub.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from parallax.gguf import GGUFFile, load_gguf
from parallax.tokenizer import BPETokenizer


_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9a-fA-F]{2})>$")


def _parse_gguf_token(token_str: str) -> bytes:
    """Converts a GGUF string token representation into raw bytes."""
    m = _BYTE_FALLBACK_RE.match(token_str)
    if m:
        byte_val = int(m.group(1), 16)
        return bytes([byte_val])
    # llama.cpp BPE uses   (U+2581 Lower One Eighth Block) for space
    normalized = token_str.replace("\u2581", " ")
    return normalized.encode("utf-8")


class GGUFTokenizer:
    """Tokenizer reconstructed directly from GGUF binary metadata."""

    def __init__(
        self,
        tokens: List[str],
        merges: Optional[List[str]] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        unk_token_id: Optional[int] = None,
    ):
        self.raw_tokens = tokens
        self.vocab_size = len(tokens)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

        # Mapping: token_id -> raw bytes
        self.id_to_bytes: Dict[int, bytes] = {}
        # Mapping: token_str -> token_id
        self.token_to_id: Dict[str, int] = {}
        # Mapping: raw bytes -> token_id
        self.bytes_to_id: Dict[bytes, int] = {}

        for idx, tok in enumerate(tokens):
            self.token_to_id[tok] = idx
            parsed_bytes = _parse_gguf_token(tok)
            self.id_to_bytes[idx] = parsed_bytes
            if parsed_bytes not in self.bytes_to_id:
                self.bytes_to_id[parsed_bytes] = idx

        # Parse BPE merges if present
        self.merges: Dict[Tuple[int, int], int] = {}
        if merges:
            for rank, merge_str in enumerate(merges):
                parts = merge_str.split(" ")
                if len(parts) == 2:
                    p0, p1 = parts[0], parts[1]
                    id0 = self.token_to_id.get(p0)
                    id1 = self.token_to_id.get(p1)
                    # The merged token
                    merged_str = p0 + p1
                    merged_id = self.token_to_id.get(merged_str)
                    if id0 is not None and id1 is not None and merged_id is not None:
                        self.merges[(id0, id1)] = merged_id

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs into a reconstructed string."""
        byte_chunks: List[bytes] = []
        for idx in token_ids:
            if idx in self.id_to_bytes:
                byte_chunks.append(self.id_to_bytes[idx])
            else:
                byte_chunks.append(b"?")
        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> List[int]:
        """Encodes text to token IDs using byte-fallback and vocabulary matching."""
        if not text:
            return []

        # If full text exists directly in vocabulary, return it
        if text in self.token_to_id:
            return [self.token_to_id[text]]

        # Byte-level fallback encoding
        raw_bytes = text.encode("utf-8")
        token_ids: List[int] = []

        i = 0
        n = len(raw_bytes)
        while i < n:
            # Greedy longest prefix matching against byte vocabulary
            matched = False
            for length in range(min(32, n - i), 0, -1):
                chunk = raw_bytes[i : i + length]
                if chunk in self.bytes_to_id:
                    token_ids.append(self.bytes_to_id[chunk])
                    i += length
                    matched = True
                    break

            if not matched:
                # Single byte fallback (<0xHH> or unk)
                single_byte = raw_bytes[i : i + 1]
                hex_tok = f"<0x{single_byte[0]:02X}>"
                if hex_tok in self.token_to_id:
                    token_ids.append(self.token_to_id[hex_tok])
                elif self.unk_token_id is not None:
                    token_ids.append(self.unk_token_id)
                i += 1

        # Apply BPE merges iteratively if merge table exists
        if self.merges and len(token_ids) >= 2:
            changed = True
            while changed and len(token_ids) >= 2:
                changed = False
                new_ids: List[int] = []
                idx = 0
                while idx < len(token_ids):
                    if idx < len(token_ids) - 1:
                        pair = (token_ids[idx], token_ids[idx + 1])
                        if pair in self.merges:
                            new_ids.append(self.merges[pair])
                            idx += 2
                            changed = True
                            continue
                    new_ids.append(token_ids[idx])
                    idx += 1
                token_ids = new_ids

        return token_ids


def load_gguf_tokenizer(source: Union[str, GGUFFile]) -> GGUFTokenizer:
    """Extracts and instantiates a GGUFTokenizer from a GGUF file or loaded GGUFFile."""
    if isinstance(source, str):
        gguf_file = load_gguf(source, cast_float64=False)
    else:
        gguf_file = source

    metadata = gguf_file.metadata
    tokens = metadata.get("tokenizer.ggml.tokens", [])
    merges = metadata.get("tokenizer.ggml.merges")

    bos_token_id = metadata.get("tokenizer.ggml.bos_token_id")
    if bos_token_id is not None:
        bos_token_id = int(bos_token_id)

    eos_token_id = metadata.get("tokenizer.ggml.eos_token_id")
    if eos_token_id is not None:
        eos_token_id = int(eos_token_id)

    pad_token_id = metadata.get("tokenizer.ggml.padding_token_id")
    if pad_token_id is not None:
        pad_token_id = int(pad_token_id)

    unk_token_id = metadata.get("tokenizer.ggml.unknown_token_id")
    if unk_token_id is not None:
        unk_token_id = int(unk_token_id)

    return GGUFTokenizer(
        tokens=tokens,
        merges=merges,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        unk_token_id=unk_token_id,
    )
