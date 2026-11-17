"""Dispatch modules for hardware-accelerated and optimized neural network layers."""

from python.parallax.dispatch.alibi import (
    AlibiPositionalBias,
    apply_alibi_bias,
    build_alibi_bias,
    dispatch_alibi_bias,
    get_alibi_slopes,
)
from python.parallax.dispatch.axial_rope import (
    AxialRoPEFrequencyTable,
    apply_axial_rope,
    apply_axial_rope_grid,
    dispatch_axial_rope,
    precompute_axial_freqs,
    precompute_axial_freqs_cis,
)
from python.parallax.dispatch.conv2d import (
    DispatchedConv2d,
    conv2d_backward,
    conv2d_forward,
    dispatched_conv2d,
)
from python.parallax.dispatch.embedding_backward import (
    DispatchedEmbeddingBackward,
    accumulate_embedding_backward,
    embedding_backward,
    sparse_embedding_backward,
)
from python.parallax.dispatch.gelu_fast import (
    DispatchedGELUFast,
    FastGELU,
    GELUFast,
    dispatched_gelu_fast,
    gelu_fast,
    gelu_fast_backward,
    gelu_fast_dispatch,
    gelu_fast_forward_np,
    gelu_tanh,
)
from python.parallax.dispatch.layernorm_backward import (
    DispatchedLayerNormBackward,
    layernorm_backward,
    layernorm_backward_reference,
)
from python.parallax.dispatch.layernorm import (
    FusedLayerNorm,
    fused_layernorm,
)
from python.parallax.dispatch.kv_cache import (
    KVCacheBuffer,
    KVCacheManager,
    create_kv_cache,
)
from python.parallax.dispatch.rope import (
    RoPEFrequencyTable,
    apply_rotary_emb,
    dispatch_rope,
    precompute_freqs,
    precompute_freqs_cis,
    rotate_half,
)
from python.parallax.dispatch.sliding_window import (
    SlidingWindowAttention,
    build_sliding_window_mask,
    dispatched_sliding_window_attention,
    flash_sliding_window_tile,
    get_sliding_window_bounds,
    sliding_window_attention,
)
from python.parallax.dispatch.sdpa import (
    DispatchedSDPA,
    SDPAModule,
    dispatched_sdpa,
    scaled_dot_product_attention,
    sdpa_backward,
)

__all__ = [
    "AlibiPositionalBias",
    "DispatchedSDPA",
    "SDPAModule",
    "dispatched_sdpa",
    "scaled_dot_product_attention",
    "sdpa_backward",
    "AxialRoPEFrequencyTable",
    "DispatchedConv2d",
    "DispatchedEmbeddingBackward",
    "DispatchedGELUFast",
    "FastGELU",
    "DispatchedLayerNormBackward",
    "FusedLayerNorm",
    "GELUFast",
    "accumulate_embedding_backward",
    "embedding_backward",
    "layernorm_backward",
    "layernorm_backward_reference",
    "fused_layernorm",
    "sparse_embedding_backward",
    "KVCacheBuffer",
    "KVCacheManager",
    "RoPEFrequencyTable",
    "SlidingWindowAttention",
    "apply_alibi_bias",
    "apply_axial_rope",
    "apply_axial_rope_grid",
    "apply_rotary_emb",
    "build_alibi_bias",
    "build_sliding_window_mask",
    "conv2d_backward",
    "conv2d_forward",
    "create_kv_cache",
    "dispatch_alibi_bias",
    "dispatch_axial_rope",
    "dispatch_rope",
    "dispatched_conv2d",
    "dispatched_gelu_fast",
    "dispatched_sliding_window_attention",
    "flash_sliding_window_tile",
    "gelu_fast",
    "gelu_fast_backward",
    "gelu_fast_dispatch",
    "gelu_fast_forward_np",
    "gelu_tanh",
    "get_alibi_slopes",
    "get_sliding_window_bounds",
    "precompute_axial_freqs",
    "precompute_axial_freqs_cis",
    "precompute_freqs",
    "precompute_freqs_cis",
    "rotate_half",
    "sliding_window_attention",
]
