"""
Zero-dependency HuggingFace tokenizer.json Parser and BPE Tokenizer.

Implements byte-level BPE decoding and encoding directly from standard HuggingFace
`tokenizer.json` configuration files without requiring `tokenizers` or `transformers`.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union


def _bytes_to_unicode() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Constructs standard GPT-2 / HuggingFace byte-to-unicode bijection maps."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    return b2u, u2b


_BYTE_TO_UNICODE, _UNICODE_TO_BYTE = _bytes_to_unicode()


class HFTokenizer:
    """Byte-level BPE tokenizer loaded from a HuggingFace tokenizer.json file."""

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: Optional[List[str]] = None,
        added_tokens: Optional[Dict[str, int]] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        unk_token: Optional[str] = None,
    ):
        self.vocab = dict(vocab)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self.added_tokens = added_tokens or {}
        for tok, tok_id in self.added_tokens.items():
            self.vocab[tok] = tok_id
            self.id_to_token[tok_id] = tok

        # Precompute byte representation for every token in vocab
        self.id_to_bytes: Dict[int, bytes] = {}
        for tok_id, tok_str in self.id_to_token.items():
            if tok_str in self.added_tokens:
                self.id_to_bytes[tok_id] = tok_str.encode("utf-8")
            else:
                # Map unicode characters back to original raw bytes
                raw = bytes([_UNICODE_TO_BYTE[c] for c in tok_str if c in _UNICODE_TO_BYTE])
                if not raw:
                    raw = tok_str.encode("utf-8")
                self.id_to_bytes[tok_id] = raw

        # Parse BPE merges
        self.merges: Dict[Tuple[str, str], int] = {}
        if merges:
            for rank, line in enumerate(merges):
                parts = line.strip().split(" ")
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = rank

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.bos_token_id = self.vocab.get(bos_token) if bos_token else None
        self.eos_token_id = self.vocab.get(eos_token) if eos_token else None
        self.pad_token_id = self.vocab.get(pad_token) if pad_token else None
        self.unk_token_id = self.vocab.get(unk_token) if unk_token else None

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs into a UTF-8 string."""
        byte_chunks: List[bytes] = []
        for idx in token_ids:
            if idx in self.id_to_bytes:
                byte_chunks.append(self.id_to_bytes[idx])
            else:
                byte_chunks.append(b"?")
        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def encode(self, text: str) -> List[int]:
        """Encodes input string to BPE token IDs."""
        if not text:
            return []

        # If full string matches special/added token directly
        if text in self.vocab:
            return [self.vocab[text]]

        # Escape added special tokens for regex split
        if self.added_tokens:
            escaped = [re.escape(k) for k in sorted(self.added_tokens.keys(), key=len, reverse=True)]
            pattern = f"({'|'.join(escaped)})"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        token_ids: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.added_tokens:
                token_ids.append(self.added_tokens[part])
                continue

            # Convert raw bytes of part to GPT-2 unicode characters
            raw_bytes = part.encode("utf-8")
            unicode_chars = [_BYTE_TO_UNICODE[b] for b in raw_bytes]

            # Iterative BPE merging over unicode character symbols
            symbols = list(unicode_chars)
            while len(symbols) >= 2:
                # Find pair with lowest merge rank
                best_pair = None
                best_rank = float("inf")
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break

                # Replace occurrences of best_pair
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                        new_symbols.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            # Map symbols to vocab IDs
            for sym in symbols:
                if sym in self.vocab:
                    token_ids.append(self.vocab[sym])
                elif self.unk_token_id is not None:
                    token_ids.append(self.unk_token_id)

        return token_ids


def load_hf_tokenizer(tokenizer_file_or_dir: str) -> HFTokenizer:
    """Loads an HFTokenizer from a `tokenizer.json` file or model directory."""
    if os.path.isdir(tokenizer_file_or_dir):
        path = os.path.join(tokenizer_file_or_dir, "tokenizer.json")
    else:
        path = tokenizer_file_or_dir

    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenizer.json not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_data = data.get("model", {})
    vocab = model_data.get("vocab", {})
    merges = model_data.get("merges", [])

    added_tokens_dict: Dict[str, int] = {}
    for item in data.get("added_tokens", []):
        content = item.get("content")
        token_id = item.get("id")
        if content is not None and token_id is not None:
            added_tokens_dict[content] = int(token_id)

    # Inspect tokenizer_config.json if present in same directory for special tokens
    model_dir = os.path.dirname(os.path.abspath(path))
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    bos_token = None
    eos_token = None
    pad_token = None
    unk_token = None

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                bos_token = cfg.get("bos_token")
                eos_token = cfg.get("eos_token")
                pad_token = cfg.get("pad_token")
                unk_token = cfg.get("unk_token")
                if isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                if isinstance(pad_token, dict):
                    pad_token = pad_token.get("content")
                if isinstance(unk_token, dict):
                    unk_token = unk_token.get("content")
        except Exception:
            pass

    return HFTokenizer(
        vocab=vocab,
        merges=merges,
        added_tokens=added_tokens_dict,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
    )
