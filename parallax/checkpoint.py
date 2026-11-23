"""
Model Checkpointing and Parameter Serialization.

Implements lightweight NumPy NPZ-based state dict export and import
for saving and resuming ParallaxLM model weights and optimizer moments.
"""

from dataclasses import asdict
import json
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def _get_layer_weight_data(layer: Any) -> np.ndarray:
    """Extracts weight array from linear or LoRA layer, keeping unmerged adapters."""
    if hasattr(layer, "effective_weight_data"):
        return layer.effective_weight_data
    return layer.weight.data


def get_model_state_dict(model: ParallaxLM) -> Dict[str, np.ndarray]:
    """Extracts a named state dictionary mapping parameter paths to ndarrays."""
    state_dict: Dict[str, np.ndarray] = {
        "token_embeddings.weight": model.token_embeddings.weight.data,
        "final_norm.weight": model.final_norm.weight.data,
    }

    if not model.config.tie_weights:
        state_dict["lm_head.weight"] = _get_layer_weight_data(model.lm_head)
        if model.lm_head.bias is not None:
            state_dict["lm_head.bias"] = model.lm_head.bias.data

    for i, block in enumerate(model.blocks):
        prefix = f"blocks.{i}"
        state_dict[f"{prefix}.attn_norm.weight"] = block.attn_norm.weight.data
        state_dict[f"{prefix}.attn.q_proj.weight"] = _get_layer_weight_data(block.attn.q_proj)
        if block.attn.q_proj.bias is not None:
            state_dict[f"{prefix}.attn.q_proj.bias"] = block.attn.q_proj.bias.data
        state_dict[f"{prefix}.attn.k_proj.weight"] = _get_layer_weight_data(block.attn.k_proj)
        if block.attn.k_proj.bias is not None:
            state_dict[f"{prefix}.attn.k_proj.bias"] = block.attn.k_proj.bias.data
        state_dict[f"{prefix}.attn.v_proj.weight"] = _get_layer_weight_data(block.attn.v_proj)
        if block.attn.v_proj.bias is not None:
            state_dict[f"{prefix}.attn.v_proj.bias"] = block.attn.v_proj.bias.data
        state_dict[f"{prefix}.attn.out_proj.weight"] = _get_layer_weight_data(block.attn.out_proj)
        if getattr(block.attn, "q_norm", None) is not None:
            state_dict[f"{prefix}.attn.q_norm.weight"] = block.attn.q_norm.weight.data
        if getattr(block.attn, "k_norm", None) is not None:
            state_dict[f"{prefix}.attn.k_norm.weight"] = block.attn.k_norm.weight.data

        state_dict[f"{prefix}.mlp_norm.weight"] = block.mlp_norm.weight.data
        if block.num_experts is not None and block.num_experts > 1:
            state_dict[f"{prefix}.mlp.gate.weight"] = block.mlp.gate.weight.data
            for e, expert in enumerate(block.mlp.experts):
                e_pfx = f"{prefix}.mlp.experts.{e}"
                if block.use_swiglu:
                    state_dict[f"{e_pfx}.w_gate.weight"] = _get_layer_weight_data(expert.w_gate)
                    state_dict[f"{e_pfx}.w_up.weight"] = _get_layer_weight_data(expert.w_up)
                    state_dict[f"{e_pfx}.w_down.weight"] = _get_layer_weight_data(expert.w_down)
                else:
                    state_dict[f"{e_pfx}.fc1.weight"] = _get_layer_weight_data(expert.fc1)
                    if expert.fc1.bias is not None:
                        state_dict[f"{e_pfx}.fc1.bias"] = expert.fc1.bias.data
                    state_dict[f"{e_pfx}.fc2.weight"] = _get_layer_weight_data(expert.fc2)
                    if expert.fc2.bias is not None:
                        state_dict[f"{e_pfx}.fc2.bias"] = expert.fc2.bias.data
        elif block.use_swiglu:
            state_dict[f"{prefix}.mlp.w_gate.weight"] = _get_layer_weight_data(block.mlp.w_gate)
            state_dict[f"{prefix}.mlp.w_up.weight"] = _get_layer_weight_data(block.mlp.w_up)
            state_dict[f"{prefix}.mlp.w_down.weight"] = _get_layer_weight_data(block.mlp.w_down)
        else:
            state_dict[f"{prefix}.mlp.fc1.weight"] = _get_layer_weight_data(block.mlp.fc1)
            if block.mlp.fc1.bias is not None:
                state_dict[f"{prefix}.mlp.fc1.bias"] = block.mlp.fc1.bias.data
            state_dict[f"{prefix}.mlp.fc2.weight"] = _get_layer_weight_data(block.mlp.fc2)
            if block.mlp.fc2.bias is not None:
                state_dict[f"{prefix}.mlp.fc2.bias"] = block.mlp.fc2.bias.data

    return state_dict


def load_model_state_dict(model: ParallaxLM, state_dict: Dict[str, np.ndarray]) -> None:
    """Loads a named state dictionary into a ParallaxLM instance in-place."""
    curr_state = get_model_state_dict(model)
    for k, expected_arr in curr_state.items():
        if k not in state_dict:
            raise KeyError(f"Missing parameter '{k}' in state_dict")
        loaded_arr = state_dict[k]
        if not isinstance(loaded_arr, np.ndarray):
            raise TypeError(f"Parameter '{k}' must be a numpy ndarray")
        if not np.issubdtype(loaded_arr.dtype, np.number):
            raise TypeError(f"Parameter '{k}' must have numeric dtype, got {loaded_arr.dtype}")
        if loaded_arr.shape != expected_arr.shape:
            raise ValueError(
                f"Shape mismatch for parameter '{k}': expected {expected_arr.shape}, got {loaded_arr.shape}"
            )

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
        if block.attn.q_proj.bias is not None and f"{prefix}.attn.q_proj.bias" in state_dict:
            block.attn.q_proj.bias.data = state_dict[f"{prefix}.attn.q_proj.bias"].copy()
        block.attn.k_proj.weight.data = state_dict[f"{prefix}.attn.k_proj.weight"].copy()
        if block.attn.k_proj.bias is not None and f"{prefix}.attn.k_proj.bias" in state_dict:
            block.attn.k_proj.bias.data = state_dict[f"{prefix}.attn.k_proj.bias"].copy()
        block.attn.v_proj.weight.data = state_dict[f"{prefix}.attn.v_proj.weight"].copy()
        if block.attn.v_proj.bias is not None and f"{prefix}.attn.v_proj.bias" in state_dict:
            block.attn.v_proj.bias.data = state_dict[f"{prefix}.attn.v_proj.bias"].copy()
        block.attn.out_proj.weight.data = state_dict[f"{prefix}.attn.out_proj.weight"].copy()
        if getattr(block.attn, "q_norm", None) is not None:
            block.attn.q_norm.weight.data = state_dict[f"{prefix}.attn.q_norm.weight"].copy()
        if getattr(block.attn, "k_norm", None) is not None:
            block.attn.k_norm.weight.data = state_dict[f"{prefix}.attn.k_norm.weight"].copy()

        block.mlp_norm.weight.data = state_dict[f"{prefix}.mlp_norm.weight"].copy()
        if block.num_experts is not None and block.num_experts > 1:
            block.mlp.gate.weight.data = state_dict[f"{prefix}.mlp.gate.weight"].copy()
            for e, expert in enumerate(block.mlp.experts):
                e_pfx = f"{prefix}.mlp.experts.{e}"
                if block.use_swiglu:
                    expert.w_gate.weight.data = state_dict[f"{e_pfx}.w_gate.weight"].copy()
                    expert.w_up.weight.data = state_dict[f"{e_pfx}.w_up.weight"].copy()
                    expert.w_down.weight.data = state_dict[f"{e_pfx}.w_down.weight"].copy()
                else:
                    expert.fc1.weight.data = state_dict[f"{e_pfx}.fc1.weight"].copy()
                    if expert.fc1.bias is not None and f"{e_pfx}.fc1.bias" in state_dict:
                        expert.fc1.bias.data = state_dict[f"{e_pfx}.fc1.bias"].copy()
                    expert.fc2.weight.data = state_dict[f"{e_pfx}.fc2.weight"].copy()
                    if expert.fc2.bias is not None and f"{e_pfx}.fc2.bias" in state_dict:
                        expert.fc2.bias.data = state_dict[f"{e_pfx}.fc2.bias"].copy()
        elif block.use_swiglu:
            block.mlp.w_gate.weight.data = state_dict[f"{prefix}.mlp.w_gate.weight"].copy()
            block.mlp.w_up.weight.data = state_dict[f"{prefix}.mlp.w_up.weight"].copy()
            block.mlp.w_down.weight.data = state_dict[f"{prefix}.mlp.w_down.weight"].copy()
        else:
            block.mlp.fc1.weight.data = state_dict[f"{prefix}.mlp.fc1.weight"].copy()
            if block.mlp.fc1.bias is not None and f"{prefix}.mlp.fc1.bias" in state_dict:
                block.mlp.fc1.bias.data = state_dict[f"{prefix}.mlp.fc1.bias"].copy()
            block.mlp.fc2.weight.data = state_dict[f"{prefix}.mlp.fc2.weight"].copy()
            if block.mlp.fc2.bias is not None and f"{prefix}.mlp.fc2.bias" in state_dict:
                block.mlp.fc2.bias.data = state_dict[f"{prefix}.mlp.fc2.bias"].copy()


def save_checkpoint(
    file_path: str,
    model: ParallaxLM,
    optimizer: Optional[Any] = None,
    step: int = 0,
) -> str:
    """Saves model weights, configuration, and optimizer state to an NPZ archive.

    Returns the exact normalized file path of the saved checkpoint.
    """
    state_dict = get_model_state_dict(model)
    for k, v in state_dict.items():
        if not isinstance(v, np.ndarray) or not np.issubdtype(v.dtype, np.number):
            raise ValueError(f"Invalid state_dict entry '{k}': expected numeric ndarray")
    config_dict = asdict(model.config)
    config_json = json.dumps(config_dict)

    save_arrays: Dict[str, Any] = {
        "__config_json__": np.array(config_json),
        "__step__": np.array(step),
        "__meta_num_experts__": np.array(model.config.num_experts if model.config.num_experts is not None else 0),
        "__meta_top_k_experts__": np.array(model.config.top_k_experts),
    }
    save_arrays.update(state_dict)

    if optimizer is not None:
        save_arrays["__opt_step_count__"] = np.array(optimizer.step_count)
        save_arrays["__opt_lr__"] = np.array(optimizer.lr)
        if hasattr(optimizer, "m"):
            for i, m in enumerate(optimizer.m):
                save_arrays[f"__opt_m_{i}__"] = m
        if hasattr(optimizer, "v"):
            for i, v in enumerate(optimizer.v):
                save_arrays[f"__opt_v_{i}__"] = v
        if hasattr(optimizer, "m_buffers"):
            for i, buf in enumerate(optimizer.m_buffers):
                save_arrays[f"__opt_m_buf_{i}__"] = buf

    normalized_path = file_path if file_path.endswith(".npz") else f"{file_path}.npz"
    np.savez(normalized_path, **save_arrays)
    return normalized_path


def load_checkpoint(
    file_path: str,
    optimizer: Optional[Any] = None,
) -> Tuple[ParallaxLM, int]:
    """Loads a checkpoint archive, instantiating the model with stored config."""
    target_path = file_path
    if not os.path.exists(target_path) and os.path.exists(f"{target_path}.npz"):
        target_path = f"{target_path}.npz"

    archive = np.load(target_path, allow_pickle=False)
    try:
        if "__config_json__" not in archive:
            raise KeyError("Missing required '__config_json__' in checkpoint archive")
        if "__step__" not in archive:
            raise KeyError("Missing required '__step__' in checkpoint archive")

        config_json = str(archive["__config_json__"])
        step = int(archive["__step__"])
        config_dict = json.loads(config_json)
        config = ParallaxLMConfig(**config_dict)

        state_dict = {
            k: archive[k].copy()
            for k in archive.files
            if not k.startswith("__")
        }
        if not state_dict:
            raise ValueError("Checkpoint archive contains empty model state_dict")

        model = ParallaxLM(config)
        load_model_state_dict(model, state_dict)

        if optimizer is not None:
            optimizer.params = model.parameters()
            num_params = len(optimizer.params)
            if hasattr(optimizer, "m"):
                optimizer.m = [np.zeros_like(p.data) for p in optimizer.params]
            if hasattr(optimizer, "v"):
                optimizer.v = [np.zeros_like(p.data) for p in optimizer.params]
            if hasattr(optimizer, "m_buffers"):
                optimizer.m_buffers = [np.zeros_like(p.data) for p in optimizer.params]
            if "__opt_step_count__" in archive:
                optimizer.step_count = int(archive["__opt_step_count__"])
                optimizer.lr = float(archive["__opt_lr__"])
                if hasattr(optimizer, "m"):
                    for i in range(num_params):
                        m_key = f"__opt_m_{i}__"
                        if m_key in archive:
                            optimizer.m[i] = archive[m_key].copy()
                if hasattr(optimizer, "v"):
                    for i in range(num_params):
                        v_key = f"__opt_v_{i}__"
                        if v_key in archive:
                            optimizer.v[i] = archive[v_key].copy()
                if hasattr(optimizer, "m_buffers"):
                    for i in range(num_params):
                        buf_key = f"__opt_m_buf_{i}__"
                        if buf_key in archive:
                            optimizer.m_buffers[i] = archive[buf_key].copy()
            elif hasattr(optimizer, "step_count"):
                optimizer.step_count = 0
    finally:
        archive.close()

    return model, step
