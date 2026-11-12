"""
Zero-dependency GGUF Model and Weights Loader for ParallaxLM.

Extracts architecture hyper-parameters and maps standard GGML/GGUF tensors
directly into a runnable ParallaxLM model without requiring `llama.cpp` or PyTorch.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.gguf import load_gguf, save_gguf
from parallax.gguf_tokenizer import GGUFTokenizer, load_gguf_tokenizer
from parallax.loader_utils import _assign_bias, _assign_param


def gguf_metadata_to_parallax(
    metadata: Dict[str, Any],
    tensors: Optional[Dict[str, np.ndarray]] = None,
) -> ParallaxLMConfig:
    """Extracts ParallaxLMConfig from GGUF metadata dictionary."""
    arch = str(metadata.get("general.architecture", "llama")).lower()
    if arch not in ("llama", "mistral", "qwen2", "qwen3", "smollm"):
        raise ValueError(
            f"Unsupported GGUF architecture: '{arch}'. Supported: llama, mistral, qwen2, qwen3, smollm."
        )

    # Architecture hyper-parameters
    required_keys = (
        f"{arch}.embedding_length",
        f"{arch}.block_count",
        f"{arch}.attention.head_count",
    )
    for req_key in required_keys:
        if req_key not in metadata:
            raise ValueError(f"Missing required GGUF metadata key: '{req_key}'")

    # Embedding and sequence dimensions
    embed_dim = int(metadata[f"{arch}.embedding_length"])
    num_layers = int(metadata[f"{arch}.block_count"])
    num_heads = int(metadata[f"{arch}.attention.head_count"])

    num_kv_heads = metadata.get(f"{arch}.attention.head_count_kv")
    if num_kv_heads is not None:
        num_kv_heads = int(num_kv_heads)
    else:
        num_kv_heads = num_heads

    intermediate_size = metadata.get(f"{arch}.feed_forward_length")
    if intermediate_size is not None:
        intermediate_size = int(intermediate_size)

    max_seq_len = int(metadata.get(f"{arch}.context_length", 2048))
    rms_norm_eps = float(metadata.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5))
    rope_theta = float(metadata.get(f"{arch}.rope.freq_base", 10000.0))

    # Vocab size calculation
    tokens = metadata.get("tokenizer.ggml.tokens")
    if tokens is not None:
        vocab_size = len(tokens)
    elif tensors and "token_embd.weight" in tensors:
        vocab_size = int(tensors["token_embd.weight"].shape[0])
    else:
        vocab_size = int(metadata.get("general.vocab_size", 32000))

    # Determine whether weights are tied (output.weight present vs omitted)
    if tensors is not None and "output.weight" in tensors:
        tie_weights = False
    else:
        tie_weights = True

    # Detect QKV bias presence
    qkv_bias = False
    if tensors is not None:
        qkv_bias = any(
            k.startswith("blk.0.attn_q.bias") or k.startswith("blk.0.attn_k.bias")
            for k in tensors
        )
    elif arch.startswith("qwen2"):
        qkv_bias = True

    # Detect SwiGLU vs standard MLP
    if f"{arch}.use_swiglu" in metadata:
        use_swiglu = bool(metadata[f"{arch}.use_swiglu"])
    elif tensors is not None:
        use_swiglu = "blk.0.ffn_gate.weight" in tensors
    else:
        use_swiglu = True

    # RoPE layout convention
    # When omitted in metadata:
    # - llama, mistral: llama.cpp permutes weights to interleaved layout (default: True)
    # - qwen2, phi, gemma, etc.: standard contiguous layout (default: False)
    if f"{arch}.rope.interleaved" in metadata:
        rope_interleaved = bool(metadata[f"{arch}.rope.interleaved"])
    else:
        rope_interleaved = arch in ("llama", "mistral")

    return ParallaxLMConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_rope=True,
        rope_theta=rope_theta,
        rope_interleaved=rope_interleaved,
        use_swiglu=use_swiglu,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        tie_weights=tie_weights,
        qkv_bias=qkv_bias,
        head_dim=int(metadata[f"{arch}.attention.key_length"]) if f"{arch}.attention.key_length" in metadata else None,
        qk_norm=bool(arch == "qwen3" or (tensors is not None and "blk.0.attn_q_norm.weight" in tensors)),
    )


def load_gguf_weights_into_model(
    model: ParallaxLM,
    gguf_tensors: Dict[str, np.ndarray],
    strict: bool = True,
) -> ParallaxLM:
    """Maps standard GGUF tensor dictionary into ParallaxLM parameters."""
    expected_keys = [
        "token_embd.weight",
        "output_norm.weight",
    ]
    if not model.config.tie_weights:
        expected_keys.append("output.weight")

    for i in range(model.config.num_layers):
        block = model.blocks[i]
        expected_keys.extend([
            f"blk.{i}.attn_norm.weight",
            f"blk.{i}.ffn_norm.weight",
            f"blk.{i}.attn_q.weight",
            f"blk.{i}.attn_k.weight",
            f"blk.{i}.attn_v.weight",
            f"blk.{i}.attn_output.weight",
        ])
        if model.config.qkv_bias:
            expected_keys.extend([
                f"blk.{i}.attn_q.bias",
                f"blk.{i}.attn_k.bias",
                f"blk.{i}.attn_v.bias",
            ])
            if getattr(block.attn, "out_proj", None) is not None and block.attn.out_proj.bias is not None:
                expected_keys.append(f"blk.{i}.attn_output.bias")
        if model.config.use_swiglu:
            expected_keys.extend([
                f"blk.{i}.ffn_gate.weight",
                f"blk.{i}.ffn_up.weight",
                f"blk.{i}.ffn_down.weight",
            ])
        else:
            expected_keys.extend([
                f"blk.{i}.ffn_up.weight",
                f"blk.{i}.ffn_down.weight",
            ])
            block = model.blocks[i]
            if hasattr(block.mlp, "fc1") and block.mlp.fc1.bias is not None:
                expected_keys.append(f"blk.{i}.ffn_up.bias")
            if hasattr(block.mlp, "fc2") and block.mlp.fc2.bias is not None:
                expected_keys.append(f"blk.{i}.ffn_down.bias")

    missing_keys = [k for k in expected_keys if k not in gguf_tensors]
    unexpected_keys = [k for k in gguf_tensors if k not in expected_keys]

    object.__setattr__(model, "_missing_keys", missing_keys)
    object.__setattr__(model, "_unexpected_keys", unexpected_keys)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Strict GGUF weight loading failed. Missing: {missing_keys}, Unexpected: {unexpected_keys}"
        )

    # 1. Token Embeddings
    _assign_param(model.token_embeddings.weight, gguf_tensors, "token_embd.weight")

    # 2. Transformer Blocks
    for i in range(model.config.num_layers):
        block = model.blocks[i]

        # Normalization layers
        _assign_param(block.attn_norm.weight, gguf_tensors, f"blk.{i}.attn_norm.weight")
        _assign_param(block.mlp_norm.weight, gguf_tensors, f"blk.{i}.ffn_norm.weight")

        # Attention projection weights
        _assign_param(block.attn.q_proj.weight, gguf_tensors, f"blk.{i}.attn_q.weight")
        _assign_param(block.attn.k_proj.weight, gguf_tensors, f"blk.{i}.attn_k.weight")
        _assign_param(block.attn.v_proj.weight, gguf_tensors, f"blk.{i}.attn_v.weight")
        _assign_param(block.attn.out_proj.weight, gguf_tensors, f"blk.{i}.attn_output.weight")

        # Attention projection biases (e.g. Qwen2)
        _assign_bias(block.attn.q_proj, gguf_tensors, f"blk.{i}.attn_q.bias")
        _assign_bias(block.attn.k_proj, gguf_tensors, f"blk.{i}.attn_k.bias")
        _assign_bias(block.attn.v_proj, gguf_tensors, f"blk.{i}.attn_v.bias")
        _assign_bias(block.attn.out_proj, gguf_tensors, f"blk.{i}.attn_output.bias")

        # SwiGLU or standard MLP weights
        mlp: Any = block.mlp
        if hasattr(mlp, "w_gate") and f"blk.{i}.ffn_gate.weight" in gguf_tensors:
            _assign_param(mlp.w_gate.weight, gguf_tensors, f"blk.{i}.ffn_gate.weight")
            _assign_param(mlp.w_up.weight, gguf_tensors, f"blk.{i}.ffn_up.weight")
            _assign_param(mlp.w_down.weight, gguf_tensors, f"blk.{i}.ffn_down.weight")
        elif hasattr(mlp, "fc1"):
            _assign_param(mlp.fc1.weight, gguf_tensors, f"blk.{i}.ffn_up.weight")
            _assign_param(mlp.fc2.weight, gguf_tensors, f"blk.{i}.ffn_down.weight")
            _assign_bias(mlp.fc1, gguf_tensors, f"blk.{i}.ffn_up.bias")
            _assign_bias(mlp.fc2, gguf_tensors, f"blk.{i}.ffn_down.bias")

    # 3. Final Norm
    _assign_param(model.final_norm.weight, gguf_tensors, "output_norm.weight")

    # 4. LM Head
    if model.config.tie_weights:
        model.lm_head.weight = model.token_embeddings.weight
    else:
        _assign_param(model.lm_head.weight, gguf_tensors, "output.weight")

    return model


def load_gguf_pretrained(
    file_path: str,
    return_tokenizer: bool = False,
    strict: bool = True,
) -> Union[ParallaxLM, Tuple[ParallaxLM, GGUFTokenizer]]:
    """Parses a GGUF file and instantiates a ready-to-run ParallaxLM model, optionally returning its tokenizer."""
    gguf_file = load_gguf(file_path, cast_float64=True)
    p_config = gguf_metadata_to_parallax(gguf_file.metadata, gguf_file.tensors)
    model = ParallaxLM(p_config)
    load_gguf_weights_into_model(model, gguf_file.tensors, strict=strict)

    if return_tokenizer:
        tokenizer = load_gguf_tokenizer(gguf_file)
        return model, tokenizer

    return model


def save_parallax_to_gguf(
    model: ParallaxLM,
    file_path: str,
    tokenizer: Optional[Any] = None,
    architecture: str = "llama",
) -> None:
    """Serializes a ParallaxLM model and optional tokenizer into a standard GGUF v3 file."""
    cfg = model.config

    if (
        hasattr(cfg, "num_experts")
        and cfg.num_experts is not None
        and cfg.num_experts > 1
    ) or any(
        hasattr(block, "moe")
        or (hasattr(block, "mlp") and (hasattr(block.mlp, "router") or hasattr(block.mlp, "experts")))
        for block in model.blocks
    ):
        raise NotImplementedError("MoE model export is not supported.")

    arch = architecture.lower()

    # Construct GGUF metadata
    if cfg.intermediate_size is not None:
        intermediate_size = cfg.intermediate_size
    elif cfg.use_swiglu:
        intermediate_size = int(2 * (4 * cfg.embed_dim) / 3)
        intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
    else:
        intermediate_size = 4 * cfg.embed_dim

    num_kv_heads = cfg.num_kv_heads if cfg.num_kv_heads is not None else cfg.num_heads


    metadata: Dict[str, Any] = {
        "general.architecture": arch,
        "general.name": "ParallaxLM",
        f"{arch}.embedding_length": int(cfg.embed_dim),
        f"{arch}.block_count": int(cfg.num_layers),
        f"{arch}.feed_forward_length": int(intermediate_size),
        f"{arch}.attention.head_count": int(cfg.num_heads),
        f"{arch}.attention.head_count_kv": int(num_kv_heads),
        f"{arch}.context_length": int(cfg.max_seq_len),
        f"{arch}.attention.layer_norm_rms_epsilon": float(cfg.rms_norm_eps),
        f"{arch}.rope.freq_base": float(cfg.rope_theta),
        f"{arch}.rope.interleaved": bool(cfg.rope_interleaved),
        f"{arch}.use_swiglu": bool(cfg.use_swiglu),
    }

    # Populate tokenizer metadata if provided
    if tokenizer is not None:
        if hasattr(tokenizer, "raw_tokens"):
            metadata["tokenizer.ggml.tokens"] = list(tokenizer.raw_tokens)
        elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
            # Sort tokens by ID
            sorted_tokens = [tok for tok, _ in sorted(tokenizer.vocab.items(), key=lambda x: x[1])]
            metadata["tokenizer.ggml.tokens"] = sorted_tokens
        if getattr(tokenizer, "bos_token_id", None) is not None:
            metadata["tokenizer.ggml.bos_token_id"] = int(tokenizer.bos_token_id)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            metadata["tokenizer.ggml.eos_token_id"] = int(tokenizer.eos_token_id)
    else:
        # Default ASCII printable chars + hex byte fallbacks for full coverage
        default_tokens = []
        for i in range(cfg.vocab_size):
            if i < 256:
                default_tokens.append(f"<0x{i:02X}>")
            else:
                default_tokens.append(f"<tok_{i}>")
        metadata["tokenizer.ggml.tokens"] = default_tokens


    # Collect tensor state dictionary
    tensors: Dict[str, Union[np.ndarray, bytes]] = {
        "token_embd.weight": model.token_embeddings.weight.data.astype(np.float32),
        "output_norm.weight": model.final_norm.weight.data.astype(np.float32),
    }

    if not cfg.tie_weights:
        tensors["output.weight"] = model.lm_head.weight.data.astype(np.float32)

    for i in range(cfg.num_layers):
        block = model.blocks[i]
        tensors[f"blk.{i}.attn_norm.weight"] = block.attn_norm.weight.data.astype(np.float32)
        tensors[f"blk.{i}.ffn_norm.weight"] = block.mlp_norm.weight.data.astype(np.float32)

        tensors[f"blk.{i}.attn_q.weight"] = block.attn.q_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.attn_k.weight"] = block.attn.k_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.attn_v.weight"] = block.attn.v_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.attn_output.weight"] = block.attn.out_proj.weight.data.astype(np.float32)

        if block.attn.q_proj.bias is not None:
            tensors[f"blk.{i}.attn_q.bias"] = block.attn.q_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"blk.{i}.attn_q.bias"] = np.zeros(block.attn.q_proj.out_features, dtype=np.float32)

        if block.attn.k_proj.bias is not None:
            tensors[f"blk.{i}.attn_k.bias"] = block.attn.k_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"blk.{i}.attn_k.bias"] = np.zeros(block.attn.k_proj.out_features, dtype=np.float32)

        if block.attn.v_proj.bias is not None:
            tensors[f"blk.{i}.attn_v.bias"] = block.attn.v_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"blk.{i}.attn_v.bias"] = np.zeros(block.attn.v_proj.out_features, dtype=np.float32)

        if block.attn.out_proj.bias is not None:
            tensors[f"blk.{i}.attn_output.bias"] = block.attn.out_proj.bias.data.astype(np.float32)

        mlp: Any = block.mlp
        if hasattr(mlp, "w_gate"):
            tensors[f"blk.{i}.ffn_gate.weight"] = mlp.w_gate.weight.data.astype(np.float32)
            tensors[f"blk.{i}.ffn_up.weight"] = mlp.w_up.weight.data.astype(np.float32)
            tensors[f"blk.{i}.ffn_down.weight"] = mlp.w_down.weight.data.astype(np.float32)
        elif hasattr(mlp, "fc1"):
            tensors[f"blk.{i}.ffn_up.weight"] = mlp.fc1.weight.data.astype(np.float32)
            tensors[f"blk.{i}.ffn_down.weight"] = mlp.fc2.weight.data.astype(np.float32)
            if mlp.fc1.bias is not None:
                tensors[f"blk.{i}.ffn_up.bias"] = mlp.fc1.bias.data.astype(np.float32)
            if mlp.fc2.bias is not None:
                tensors[f"blk.{i}.ffn_down.bias"] = mlp.fc2.bias.data.astype(np.float32)

    save_gguf(file_path, metadata=metadata, tensors=tensors, version=3)
