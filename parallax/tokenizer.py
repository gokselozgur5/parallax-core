"""
Byte-Pair Encoding (BPE) Subword Tokenizer.

Implements byte-level BPE tokenization from first principles:
- Byte-level base vocabulary (256 raw bytes + special tokens)
- Iterative frequency-based pair statistics collection
- Pair merge rules training and serialization/deserialization
- Deterministic subword token encoding and decoding
"""

import codecs
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def get_pair_stats(ids: List[int], counts: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[Tuple[int, int], int]:
    """Computes pair frequencies of consecutive token IDs."""
    if counts is None:
        counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_tokens(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replaces all occurrences of `pair` in `ids` with `new_id`."""
    new_ids: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer."""

    def __init__(self, special_tokens: Optional[Dict[str, int]] = None):
        # Base byte vocabulary: 0-255 map directly to single-byte bytes
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens: Dict[str, int] = special_tokens or {
            "<pad>": 256,
            "<bos>": 257,
            "<eos>": 258,
            "<unk>": 259,
        }
        # Add special tokens to vocab
        for tok, tok_id in self.special_tokens.items():
            self.vocab[tok_id] = tok.encode("utf-8")

        self.inverse_special: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Trains BPE merge rules on raw text up to the desired target vocab size."""
        assert vocab_size >= 260, "vocab_size must be at least 260 to accommodate byte tokens + specials"
        num_merges = vocab_size - len(self.vocab)
        if num_merges <= 0:
            return

        raw_bytes = text.encode("utf-8")
        tokens = list(raw_bytes)

        next_id = max(self.vocab.keys()) + 1

        for step in range(num_merges):
            stats = get_pair_stats(tokens)
            if not stats:
                break
            # Find most frequent pair
            top_pair = max(stats, key=stats.get)  # type: ignore[arg-type]
            if stats[top_pair] < 2:
                # Pair occurred only once, no point in merging
                break

            self.merges[top_pair] = next_id
            self.vocab[next_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            tokens = merge_tokens(tokens, top_pair, next_id)

            if verbose and (step + 1) % 10 == 0:
                print(f"BPE Merge {step + 1}/{num_merges}: pair {top_pair} -> id {next_id}")

            next_id += 1

    def _encode_bytes(self, raw_bytes: bytes) -> List[int]:
        tokens = list(raw_bytes)
        while len(tokens) >= 2:
            stats = get_pair_stats(tokens)
            pair_candidates = {pair: self.merges[pair] for pair in stats if pair in self.merges}
            if not pair_candidates:
                break
            best_pair = min(pair_candidates, key=pair_candidates.get)  # type: ignore[arg-type]
            new_id = self.merges[best_pair]
            tokens = merge_tokens(tokens, best_pair, new_id)
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encodes text to a sequence of BPE token IDs."""
        if not text:
            return []

        if not self.special_tokens:
            return self._encode_bytes(text.encode("utf-8"))

        escaped = [re.escape(tok) for tok in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        pattern = f"({'|'.join(escaped)})"
        parts = re.split(pattern, text)
        tokens: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self._encode_bytes(part.encode("utf-8")))

        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decodes a sequence of BPE token IDs back to a UTF-8 string."""
        byte_chunks: List[bytes] = []
        for idx in ids:
            if idx in self.vocab:
                byte_chunks.append(self.vocab[idx])
            else:
                byte_chunks.append(b"?")

        full_bytes = b"".join(byte_chunks)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, file_path: str) -> None:
        """Serializes merges and special tokens into a JSON file."""
        data = {
            "special_tokens": self.special_tokens,
            "merges": [
                {"pair": list(pair), "id": new_id}
                for pair, new_id in self.merges.items()
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "BPETokenizer":
        """Loads a trained BPE tokenizer from a serialized JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        special_tokens = data.get("special_tokens", {})
        tokenizer = cls(special_tokens=special_tokens)

        for item in data.get("merges", []):
            pair = (int(item["pair"][0]), int(item["pair"][1]))
            new_id = int(item["id"])
            tokenizer.merges[pair] = new_id
            tokenizer.vocab[new_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]

        return tokenizer


class IncrementalTokenDecoder:
    """Incremental UTF-8 token decoder for streaming generation.

    Decodes tokens progressively, buffering incomplete multi-byte UTF-8 sequences
    so that multi-byte characters split across token boundaries are not corrupted
    into replacement characters (U+FFFD).
    """

    def __init__(self, tokenizer: Optional[Any] = None, errors: str = "replace"):
        self.tokenizer = tokenizer
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)

    def decode_token(self, token_id: int) -> str:
        """Decodes a single token into newly completed text characters."""
        if self.tokenizer is None:
            return f"[{token_id}] "

        raw_bytes: Optional[bytes] = None
        if hasattr(self.tokenizer, "id_to_bytes") and isinstance(self.tokenizer.id_to_bytes, dict):
            raw_bytes = self.tokenizer.id_to_bytes.get(token_id)
        elif hasattr(self.tokenizer, "vocab") and isinstance(self.tokenizer.vocab, dict):
            val = self.tokenizer.vocab.get(token_id)
            if isinstance(val, bytes):
                raw_bytes = val

        if raw_bytes is None:
            raw_bytes = self.tokenizer.decode([token_id]).encode("utf-8", errors="replace")

        return self.decoder.decode(raw_bytes, final=False)

    def finish(self) -> str:
        """Flushes any remaining buffered bytes."""
        return self.decoder.decode(b"", final=True)
