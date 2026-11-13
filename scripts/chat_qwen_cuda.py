#!/usr/bin/env python3
"""
Interactive Chat with Qwen2.5 on NVIDIA RTX 4090 GPU using Parallax CUDA Engine.

Features:
- Native CUDA tensor forward pass with KV Cache & bfloat16 precision
- Blazing-fast inference (>30-60 tokens/sec on 3B/0.5B)
- Top-P (Nucleus) & Top-K sampling (no repetitive templates)
- Natural conversation flow with Parallax identity
"""

import os
import sys
import time

# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from typing import Dict, List
from parallax.cuda_engine import Qwen3CUDAModel, is_cuda_available
from parallax.hf_tokenizer import load_hf_tokenizer


def print_banner(model_name: str, device_name: str, vram_gb: float) -> None:
    print("=" * 68)
    print("  PARALLAX-CORE: Interactive GPU Inference")
    print(f"  Model  : {model_name}")
    print(f"  Device : {device_name} ({vram_gb:.1f} GB Memory, bfloat16)")
    print("  Commands: /think, /nothink, /clear, exit")
    print("=" * 68 + "\n")


def build_chatml_prompt(history: List[Dict[str, str]], is_qwen3: bool, enable_thinking: bool = False) -> str:
    kept = [history[0]] + history[1:][-8:] if len(history) > 9 else history
    body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in kept)
    suffix = "" if (not is_qwen3 or enable_thinking) else "<think>\n\n</think>\n\n"
    return body + "<|im_start|>assistant\n" + suffix


def main():
    if not is_cuda_available():
        print("ERROR: CUDA GPU is not available.")
        sys.exit(1)

    import torch
    dev_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    default_gguf = os.path.expanduser("~/.parallax/models/Qwen3-32B-Q4_K_M.gguf")
    candidates = [sys.argv[1]] if len(sys.argv) > 1 else [default_gguf, "models/qwen2.5-3b", "models/qwen2.5-0.5b"]
    model_dir = next((p for p in candidates if os.path.exists(p)), "")
    if not model_dir:
        print("ERROR: Model path not found. Pass a .gguf file or HF directory.")
        sys.exit(1)

    model_name = os.path.basename(model_dir)
    print(f"[1/2] Loading weights onto {dev_name} ({model_name})...", flush=True)
    t0 = time.time()
    model = Qwen3CUDAModel(model_dir=model_dir, device="cuda")
    print(f"[OK] Loaded in {time.time() - t0:.2f}s.\n", flush=True)
    tokenizer = model.tokenizer if model.tokenizer is not None else load_hf_tokenizer(model_dir)
    is_qwen3 = "<think>" in getattr(tokenizer, "token_to_id", {})
    print_banner(model_name, dev_name, vram)
    
    system_prompt = "You are Parallax, a concise and direct AI partner. Mirror the user's language naturally."
    conversation_history = [{"role": "system", "content": system_prompt}]
    enable_thinking = False

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("q", "exit", "quit"):
                break
            if user_input in ("/think", "/nothink", "/clear"):
                if user_input == "/clear":
                    conversation_history = [{"role": "system", "content": system_prompt}]
                else:
                    enable_thinking = (user_input == "/think")
                print(f"[mode] thinking={enable_thinking}, history={len(conversation_history) - 1}\n")
                continue
                
            conversation_history.append({"role": "user", "content": user_input})
            
            prompt = build_chatml_prompt(conversation_history, is_qwen3, enable_thinking)
            prompt_tokens = tokenizer.encode(prompt)
            print("\nParallax: ", end="", flush=True)
            t_gen_start = time.time()
            gen_tokens: List[int] = []
            printed_len = 0
            
            for token_id in model.generate_stream(
                prompt_tokens=prompt_tokens,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.12,
            ):
                gen_tokens.append(token_id)
                token_text = tokenizer.decode([token_id])
                print(token_text, end="", flush=True)
                
            gen_time = time.time() - t_gen_start
            speed = len(gen_tokens) / gen_time if gen_time > 0 else 0
            
            print(f"\n\n⚡ [{speed:.1f} tokens/s | {len(gen_tokens)} tokens in {gen_time:.2f}s]\n")
            
            assistant_reply = tokenizer.decode(gen_tokens).strip()
            conversation_history.append({"role": "assistant", "content": assistant_reply})
            
        except (KeyboardInterrupt, EOFError):
            print("\nÇıkış yapıldı.")
            break


if __name__ == "__main__":
    main()
