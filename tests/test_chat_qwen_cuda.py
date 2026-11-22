"""Tests for scripts/chat_qwen_cuda.py."""

from scripts.chat_qwen_cuda import detect_thinking_support


class DummyHFTokenizer:
    def __init__(self, vocab: dict) -> None:
        self.vocab = vocab


class DummyGGUFTokenizer:
    def __init__(self, token_to_id: dict) -> None:
        self.token_to_id = token_to_id


def test_detect_thinking_support_hf_tokenizer() -> None:
    hf_with_think = DummyHFTokenizer({"<think>": 151668, "<|im_start|>": 151644})
    assert detect_thinking_support(hf_with_think) is True

    hf_without_think = DummyHFTokenizer({"<|im_start|>": 151644, "<|im_end|>": 151645})
    assert detect_thinking_support(hf_without_think) is False


def test_detect_thinking_support_gguf_tokenizer() -> None:
    gguf_with_think = DummyGGUFTokenizer({"<think>": 151668, "<|im_start|>": 151644})
    assert detect_thinking_support(gguf_with_think) is True

    gguf_without_think = DummyGGUFTokenizer({"<|im_start|>": 151644, "<|im_end|>": 151645})
    assert detect_thinking_support(gguf_without_think) is False
