"""
Zero-dependency GGUF Tokenizer Extractor and Decoder.

Extracts BPE/SPM token dictionaries, byte fallbacks, and merge tables directly
from GGUF metadata headers without requiring sentencepiece, protobuf, or huggingface_hub.
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

from parallax.gguf import GGUFFile, load_gguf
from parallax.hf_tokenizer import _BYTE_TO_UNICODE, _UNICODE_TO_BYTE, _translate_hf_regex


_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9a-fA-F]{2})>$")
_GPT2_SPLIT_RE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
_QWEN2_SPLIT_RE = re.compile(
    _translate_hf_regex(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
)
_LLAMA3_SPLIT_RE = re.compile(
    _translate_hf_regex(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
)



def _parse_gguf_token(token_str: str, model_type: Optional[str] = None, is_special: bool = False) -> bytes:
    """Converts a GGUF string token representation into raw bytes."""
    if is_special:
        return token_str.encode("utf-8")
    m = _BYTE_FALLBACK_RE.match(token_str)
    if m:
        byte_val = int(m.group(1), 16)
        return bytes([byte_val])
    if model_type == "gpt2":
        raw = bytes([_UNICODE_TO_BYTE[c] for c in token_str if c in _UNICODE_TO_BYTE])
        if raw:
            return raw
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
        model_type: Optional[str] = None,
        token_types: Optional[List[int]] = None,
        scores: Optional[List[float]] = None,
        pre_type: Optional[str] = None,
    ):
        self.raw_tokens = tokens
        self.vocab_size = len(tokens)
        self.scores = [float(s) for s in scores] if scores is not None else None
        self.bos_token_id = bos_token_id if (bos_token_id is not None and 0 <= bos_token_id < self.vocab_size) else None
        self.eos_token_id = eos_token_id if (eos_token_id is not None and 0 <= eos_token_id < self.vocab_size) else None
        self.pad_token_id = pad_token_id if (pad_token_id is not None and 0 <= pad_token_id < self.vocab_size) else None
        self.unk_token_id = unk_token_id if (unk_token_id is not None and 0 <= unk_token_id < self.vocab_size) else None
        self.model_type = model_type
        self.pre_type = pre_type
        self.split_re = _GPT2_SPLIT_RE
        if pre_type in ("qwen2", "qwen"):
            self.split_re = _QWEN2_SPLIT_RE
        elif pre_type in ("llama-bpe", "llama3", "llama-3"):
            self.split_re = _LLAMA3_SPLIT_RE
        elif pre_type in ("gpt-2", "gpt2", None):
            self.split_re = _GPT2_SPLIT_RE
        else:
            warnings.warn(
                f"Unknown tokenizer.ggml.pre={pre_type!r}: falling back to default regex.",
                UserWarning,
                stacklevel=2,
            )
            self.split_re = _GPT2_SPLIT_RE


        # Mapping: token_id -> raw bytes
        self.id_to_bytes: Dict[int, bytes] = {}
        # Mapping: token_str -> token_id
        self.token_to_id: Dict[str, int] = {}
        # Mapping: raw bytes -> token_id
        self.bytes_to_id: Dict[bytes, int] = {}

        special_ids = {i for i, t in enumerate(token_types or []) if i < len(tokens) and int(t) in (3, 4)}
        specials = sorted((tokens[i] for i in special_ids if tokens[i]), key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")") if specials else None

        for idx, tok in enumerate(tokens):
            self.token_to_id[tok] = idx
            parsed_bytes = _parse_gguf_token(tok, model_type=model_type, is_special=(idx in special_ids))
            self.id_to_bytes[idx] = parsed_bytes
            if parsed_bytes not in self.bytes_to_id:
                self.bytes_to_id[parsed_bytes] = idx

        # Parse BPE merges if present
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        if merges:
            for rank, merge_str in enumerate(merges):
                parts = merge_str.split(" ")
                if len(parts) == 2:
                    p0, p1 = parts[0], parts[1]
                    self.merge_ranks[(p0, p1)] = rank
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

    def _bpe_merge_symbols(self, symbols: List[str]) -> List[int]:
        while len(symbols) >= 2:
            best_pair = min(
                ((symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1) if (symbols[i], symbols[i + 1]) in self.merge_ranks),
                key=lambda p: self.merge_ranks[p],
                default=None,
            )
            if best_pair is None:
                break
            merged, new_syms, idx = best_pair[0] + best_pair[1], [], 0
            while idx < len(symbols):
                if idx < len(symbols) - 1 and (symbols[idx], symbols[idx + 1]) == best_pair:
                    new_syms.append(merged)
                    idx += 2
                else:
                    new_syms.append(symbols[idx])
                    idx += 1
            symbols = new_syms
        out: List[int] = []
        for s in symbols:
            if s in self.token_to_id:
                out.append(self.token_to_id[s])
            elif self.unk_token_id is not None:
                out.append(self.unk_token_id)
            else:
                raise KeyError(
                    f"Token '{s}' not found in vocabulary and no unk_token defined."
                )
        return out

    def _spm_score_merge(self, symbols: List[str]) -> List[int]:
        while len(symbols) >= 2:
            best_idx = None
            best_score = float("-inf")
            for idx in range(len(symbols) - 1):
                pair_str = symbols[idx] + symbols[idx + 1]
                if pair_str in self.token_to_id:
                    tid = self.token_to_id[pair_str]
                    score = self.scores[tid] if self.scores and tid < len(self.scores) else 0.0
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            if best_idx is None:
                break
            symbols = symbols[:best_idx] + [symbols[best_idx] + symbols[best_idx + 1]] + symbols[best_idx + 2:]
        return [self.token_to_id[s] for s in symbols if s in self.token_to_id]

    def encode(self, text: str) -> List[int]:
        """Encodes text to token IDs using byte-fallback and vocabulary matching."""
        if not text:
            return []
        if self._special_re is not None:
            parts = [p for p in self._special_re.split(text) if p]
            if len(parts) > 1 or (len(parts) == 1 and parts[0] in self.token_to_id and self._special_re.fullmatch(parts[0])):
                out: List[int] = []
                for p in parts:
                    out.extend([self.token_to_id[p]] if self._special_re.fullmatch(p) else self._encode_plain(p))
                return out
        return self._encode_plain(text)

    def _encode_plain(self, text: str) -> List[int]:
        if self.model_type == "llama" and not text.startswith((" ", "\u2581")):
            text = " " + text

        if self.scores is not None and self.model_type == "llama" and not self.merge_ranks:
            text = text.replace(" ", "\u2581")
            spm_symbols: List[str] = [
                ch if ch in self.token_to_id else (
                    f"<0x{ch.encode('utf-8')[0]:02X}>" if f"<0x{ch.encode('utf-8')[0]:02X}>" in self.token_to_id
                    else (self.raw_tokens[self.unk_token_id] if self.unk_token_id is not None else "")
                )
                for ch in text
            ]
            return self._spm_score_merge([s for s in spm_symbols if s])

        if self.merge_ranks and self.model_type == "gpt2":
            ids: List[int] = []
            for piece in self.split_re.findall(text):
                ids.extend(self._bpe_merge_symbols([_BYTE_TO_UNICODE[b] for b in piece.encode("utf-8")]))
            return ids

        # If merge table exists, perform rank-ordered BPE over initial symbols
        if self.merge_ranks:
            symbols: List[str] = []
            i = 0
            n = len(text)
            while i < n:
                ch = text[i]
                if ch in self.token_to_id:
                    symbols.append(ch)
                    i += 1
                else:
                    raw = ch.encode("utf-8")
                    for b in raw:
                        hex_tok = f"<0x{b:02X}>"
                        if hex_tok in self.token_to_id:
                            symbols.append(hex_tok)
                        elif self.unk_token_id is not None:
                            symbols.append(self.raw_tokens[self.unk_token_id])
                    i += 1

            return self._bpe_merge_symbols(symbols)

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

        return token_ids


def load_gguf_tokenizer(source: Union[str, GGUFFile]) -> GGUFTokenizer:
    """Extracts and instantiates a GGUFTokenizer from a GGUF file or loaded GGUFFile."""
    if isinstance(source, str):
        gguf_file = load_gguf(source, cast_float64=False, header_only=True)
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

    model_type = metadata.get("tokenizer.ggml.model")
    if model_type is not None:
        model_type = str(model_type)

    pre_type = metadata.get("tokenizer.ggml.pre")
    if pre_type is not None:
        pre_type = str(pre_type)

    return GGUFTokenizer(
        tokens=tokens,
        merges=merges,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        unk_token_id=unk_token_id,
        model_type=model_type,
        token_types=metadata.get("tokenizer.ggml.token_type"),
        scores=metadata.get("tokenizer.ggml.scores"),
        pre_type=pre_type,
    )
