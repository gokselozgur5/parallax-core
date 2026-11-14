#ifndef PARALLAX_ROPE_CACHE_H
#define PARALLAX_ROPE_CACHE_H

#include <stddef.h>

int c_build_rope_cache_f32(
    float *cos_out,
    float *sin_out,
    int max_seq_len,
    int dim,
    float theta
);

int c_apply_rope_cached_f32(
    float *x,
    const float *cos_cache,
    const float *sin_cache,
    int T,
    int H,
    int D,
    int start_pos,
    int max_seq_len,
    int interleaved
);

#endif /* PARALLAX_ROPE_CACHE_H */
