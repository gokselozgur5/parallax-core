"""
Interactive Terminal Chat with real HuggingFace models using parallax-core engine.
Zero external ML dependencies (No PyTorch, No Transformers).
"""

import sys
import os
import time

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from parallax.hf_loader import load_hf_pretrained
from parallax.hf_tokenizer import load_hf_tokenizer

def main():
    model_path = "/Users/gok/repos/parallax-core/models/qwen2.5-0.5b"
    if not os.path.exists(model_path):
        print(f"Hata: Model yolu bulunamadı: {model_path}")
        return

    print("=" * 65)
    print("🚀 PARALLAX-CORE: Standalone Qwen2.5-0.5B-Instruct Canlı Çıkarım")
    print("Sıfır PyTorch • Saf Matematik & C/Metal Donanım Çekirdeği")
    print("=" * 65)

    print("\n[1/2] Model ağırlıkları SafeTensors'tan doğrudan belleğe yükleniyor...")
    t0 = time.time()
    model = load_hf_pretrained(model_path)
    print(f"✓ Model başarıyla yüklendi! ({time.time() - t0:.2f} saniye)")
    print(f"  - Katman Sayısı: {len(model.blocks)}")
    print(f"  - Kelime Havuzu (Vocab): {model.config.vocab_size}")
    print(f"  - Gizli Boyut (Embed Dim): {model.config.embed_dim}")

    print("\n[2/2] Tokenizer hazırlanıyor...")
    tokenizer_path = os.path.join(model_path, "tokenizer.json")
    tokenizer = load_hf_tokenizer(tokenizer_path)
    print("✓ Qwen2.5 BPE Tokenizer aktif.")

    print("\n" + "-" * 65)
    print("Sohbete başlayabilirsiniz! Çıkmak için 'q' veya 'exit' yazın.")
    print("-" * 65)

    while True:
        try:
            prompt = input("\nSen: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["q", "exit", "quit"]:
                print("\nParallax: Görüşmek üzere kankam!")
                break

            # Qwen2.5 Chat format with System Prompt
            formatted_prompt = (
                "<|im_start|>system\nSen yardımsever, zeki ve samimi bir Türkçe yapay zeka asistanısın.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            token_ids = tokenizer.encode(formatted_prompt)
            all_tokens = np.array([token_ids], dtype=np.int64)

            print("\nParallax (Qwen2.5): ", end="", flush=True)

            # Greedy decoding with repetition penalty
            for step in range(128):
                output = model(all_tokens)
                logits_data = output.data if hasattr(output, "data") else output
                current_logits = logits_data[0, -1, :].copy()
                
                # Apply repetition penalty (1.2) to already generated tokens
                generated_tokens = all_tokens[0, len(token_ids):]
                if len(generated_tokens) > 0:
                    for t in set(generated_tokens):
                        if current_logits[t] > 0:
                            current_logits[t] /= 1.2
                        else:
                            current_logits[t] *= 1.2

                next_token = int(np.argmax(current_logits))
                
                # Stop tokens (<|im_end|>, <|endoftext|>)
                if next_token in [151645, 151643]:
                    break
                    
                piece = tokenizer.decode([next_token])
                print(piece, end="", flush=True)

                all_tokens = np.concatenate([all_tokens, np.array([[next_token]], dtype=np.int64)], axis=1)

            print()

        except (KeyboardInterrupt, EOFError):
            print("\nÇıkış yapıldı.")
            break

if __name__ == "__main__":
    main()
