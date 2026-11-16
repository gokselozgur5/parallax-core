"""Dispatch subsystem modules and kernels."""

from python.parallax.dispatch.axial_rope import (
    AxialRoPEFrequencyTable,
    apply_axial_rope,
    apply_axial_rope_grid,
    dispatch_axial_rope,
    precompute_axial_freqs,
    precompute_axial_freqs_cis,
)
from python.parallax.dispatch.fused_norm import (
    FusedRMSNorm,
    fused_rmsnorm,
    stabilize_epsilon,
)
from python.parallax.dispatch.gqa import (
    GQAHeadRepeater,
    dispatched_repeat_kv,
    reduce_kv_grad,
    repeat_kv,
    repeat_kv_pair,
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
from python.parallax.dispatch.logit_cap import (
    DispatchedLogitCap,
    LogitSoftCap,
    logit_soft_cap,
    logit_soft_cap_backward,
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
from python.parallax.dispatch.swiglu import (
    DispatchedSwiGLU,
    SwiGLUActivation,
    dispatched_swiglu,
    swiglu_backward_dispatch,
    swiglu_dispatch,
)

__all__ = [
    "DispatchedSDPA",
    "SDPAModule",
    "dispatched_sdpa",
    "scaled_dot_product_attention",
    "sdpa_backward",
    "AxialRoPEFrequencyTable",
    "DispatchedSwiGLU",
    "FusedLayerNorm",
    "FusedRMSNorm",
    "GQAHeadRepeater",
    "fused_layernorm",
    "KVCacheBuffer",
    "KVCacheManager",
    "LogitSoftCap",
    "DispatchedLogitCap",
    "RoPEFrequencyTable",
    "SlidingWindowAttention",
    "SwiGLUActivation",
    "apply_axial_rope",
    "apply_axial_rope_grid",
    "apply_rotary_emb",
    "build_sliding_window_mask",
    "create_kv_cache",
    "dispatch_axial_rope",
    "dispatch_rope",
    "dispatched_repeat_kv",
    "dispatched_sliding_window_attention",
    "dispatched_swiglu",
    "flash_sliding_window_tile",
    "fused_rmsnorm",
    "get_sliding_window_bounds",
    "logit_soft_cap",
    "logit_soft_cap_backward",
    "precompute_axial_freqs",
    "precompute_axial_freqs_cis",
    "precompute_freqs",
    "precompute_freqs_cis",
    "reduce_kv_grad",
    "repeat_kv",
    "repeat_kv_pair",
    "rotate_half",
    "sliding_window_attention",
    "stabilize_epsilon",
    "swiglu_backward_dispatch",
    "swiglu_dispatch",
]
