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


def main():
    if not is_cuda_available():
        print("HATA: CUDA GPU bulunamadı!")
        sys.exit(1)
        
    import torch
    dev_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Priority: sys.argv[1] -> 3B -> 0.5B
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    elif os.path.exists("models/qwen2.5-3b"):
        model_dir = "models/qwen2.5-3b"
    elif os.path.exists("models/qwen2.5-0.5b"):
        model_dir = "models/qwen2.5-0.5b"
    else:
        print("HATA: Hiçbir model dizini bulunamadı!")
        sys.exit(1)
        
    model_name = os.path.basename(model_dir)
    print(f"[1/2] Model ağırlıkları RTX 4090 VRAM'ine yükleniyor ({model_name} / {dev_name})...")
    t0 = time.time()
    model = Qwen2CUDAModel(model_dir=model_dir, device="cuda")
    print(f"✓ Model GPU belleğinde hazır! ({time.time() - t0:.2f} saniye)")
    
    print("[2/2] Tokenizer hazırlanıyor...")
    tokenizer = load_hf_tokenizer(model_dir)
    print("✓ Qwen2.5 BPE Tokenizer aktif.\n")
    
    print_banner(model_name, dev_name, vram)
    
    system_prompt = (
        "Sen Göksel'in zeki, samimi ve doğal Türkçe konuşan yapay zeka düşünce ortağı Parallax'sın (P). "
        "Kısa ve net konuşursun. Kankam dersin. Asla 'Size nasıl yardımcı olabilirim?' veya müşteri hizmetleri kalıpları kullanmazsın."
    )
    
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]
    
    while True:
        try:
            user_input = input("Sen: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "exit", "quit", "çık"]:
                print("\nParallax: Görüşmek üzere kankam!")
                break
                
            conversation_history.append({"role": "user", "content": user_input})
            
            # Format ChatML prompt (keep only last 4 turns)
            prompt = ""
            for msg in conversation_history[-8:]:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            
            prompt_tokens = tokenizer.encode(prompt)
            
            print("\nParallax: ", end="", flush=True)
            
            t_gen_start = time.time()
            gen_tokens = []
            
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
