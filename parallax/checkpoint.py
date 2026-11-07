"""
Model Checkpointing and Parameter Serialization.

Implements lightweight NumPy NPZ-based state dict export and import
for saving and resuming ParallaxLM model weights and optimizer moments.
"""

from dataclasses import asdict
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW


def get_model_state_dict(model: ParallaxLM) -> Dict[str, np.ndarray]:
    """Extracts a named state dictionary mapping parameter paths to ndarrays."""
    state_dict: Dict[str, np.ndarray] = {
        "token_embeddings.weight": model.token_embeddings.weight.data,
        "final_norm.weight": model.final_norm.weight.data,
    }

    if not model.config.tie_weights:
        state_dict["lm_head.weight"] = model.lm_head.weight.data
        if model.lm_head.bias is not None:
            state_dict["lm_head.bias"] = model.lm_head.bias.data

    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}"
        state_dict[f"{prefix}.attn_norm.weight"] = block.attn_norm.weight.data
        state_dict[f"{prefix}.attn.q_proj.weight"] = block.attn.q_proj.weight.data
        state_dict[f"{prefix}.attn.k_proj.weight"] = block.attn.k_proj.weight.data
        state_dict[f"{prefix}.attn.v_proj.weight"] = block.attn.v_proj.weight.data
        state_dict[f"{prefix}.attn.out_proj.weight"] = block.attn.out_proj.weight.data

        state_dict[f"{prefix}.mlp_norm.weight"] = block.mlp_norm.weight.data
        state_dict[f"{prefix}.mlp.fc1.weight"] = block.mlp.fc1.weight.data
        if block.mlp.fc1.bias is not None:
            state_dict[f"{prefix}.mlp.fc1.bias"] = block.mlp.fc1.bias.data
        state_dict[f"{prefix}.mlp.fc2.weight"] = block.mlp.fc2.weight.data
        if block.mlp.fc2.bias is not None:
            state_dict[f"{prefix}.mlp.fc2.bias"] = block.mlp.fc2.bias.data

    return state_dict


def load_model_state_dict(model: ParallaxLM, state_dict: Dict[str, np.ndarray]) -> None:
    """Loads a named state dictionary into a ParallaxLM instance in-place."""
    model.token_embeddings.weight.data = state_dict["token_embeddings.weight"].copy()
    model.final_norm.weight.data = state_dict["final_norm.weight"].copy()

    if model.config.tie_weights:
        model.lm_head.weight = model.token_embeddings.weight
    else:
        model.lm_head.weight.data = state_dict["lm_head.weight"].copy()
        if model.lm_head.bias is not None and "lm_head.bias" in state_dict:
            model.lm_head.bias.data = state_dict["lm_head.bias"].copy()

    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}"
        block.attn_norm.weight.data = state_dict[f"{prefix}.attn_norm.weight"].copy()
        block.attn.q_proj.weight.data = state_dict[f"{prefix}.attn.q_proj.weight"].copy()
        block.attn.k_proj.weight.data = state_dict[f"{prefix}.attn.k_proj.weight"].copy()
        block.attn.v_proj.weight.data = state_dict[f"{prefix}.attn.v_proj.weight"].copy()
        block.attn.out_proj.weight.data = state_dict[f"{prefix}.attn.out_proj.weight"].copy()

        block.mlp_norm.weight.data = state_dict[f"{prefix}.mlp_norm.weight"].copy()
        block.mlp.fc1.weight.data = state_dict[f"{prefix}.mlp.fc1.weight"].copy()
        if block.mlp.fc1.bias is not None and f"{prefix}.mlp.fc1.bias" in state_dict:
            block.mlp.fc1.bias.data = state_dict[f"{prefix}.mlp.fc1.bias"].copy()
        block.mlp.fc2.weight.data = state_dict[f"{prefix}.mlp.fc2.weight"].copy()
        if block.mlp.fc2.bias is not None and f"{prefix}.mlp.fc2.bias" in state_dict:
            block.mlp.fc2.bias.data = state_dict[f"{prefix}.mlp.fc2.bias"].copy()


def save_checkpoint(
    file_path: str,
    model: ParallaxLM,
    optimizer: Optional[AdamW] = None,
    step: int = 0,
) -> None:
    """Saves model weights, configuration, and optimizer state to an NPZ archive."""
    state_dict = get_model_state_dict(model)
    config_dict = asdict(model.config)
    config_json = json.dumps(config_dict)

    save_arrays: Dict[str, Any] = {
        "__config_json__": np.array(config_json),
        "__step__": np.array(step),
    }
    save_arrays.update(state_dict)

    if optimizer is not None:
        save_arrays["__opt_step_count__"] = np.array(optimizer.step_count)
        save_arrays["__opt_lr__"] = np.array(optimizer.lr)
        for i, m in enumerate(optimizer.m):
            save_arrays[f"__opt_m_{i}__"] = m
        for i, v in enumerate(optimizer.v):
            save_arrays[f"__opt_v_{i}__"] = v

    np.savez(file_path, **save_arrays)


def load_checkpoint(
    file_path: str,
    optimizer: Optional[AdamW] = None,
) -> Tuple[ParallaxLM, int]:
    """Loads a checkpoint archive, instantiating the model with stored config."""
    archive = np.load(file_path, allow_pickle=False)

    config_json = str(archive["__config_json__"])
    step = int(archive["__step__"])
    config_dict = json.loads(config_json)
    config = ParallaxLMConfig(**config_dict)

    model = ParallaxLM(config)
    state_dict = {
        k: archive[k]
        for k in archive.files
        if not k.startswith("__")
    }
    load_model_state_dict(model, state_dict)

    if optimizer is not None and "__opt_step_count__" in archive:
        optimizer.step_count = int(archive["__opt_step_count__"])
        optimizer.lr = float(archive["__opt_lr__"])
        for i in range(len(optimizer.m)):
            m_key = f"__opt_m_{i}__"
            v_key = f"__opt_v_{i}__"
            if m_key in archive:
                optimizer.m[i] = archive[m_key].copy()
            if v_key in archive:
                optimizer.v[i] = archive[v_key].copy()

    return model, step
