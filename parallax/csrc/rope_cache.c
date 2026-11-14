#include <math.h>
#include <stddef.h>
#include "parallax_simd.h"
#include "rope_cache_decl.h"

int c_build_rope_cache_f32(
    float *cos_out, float *sin_out,
    int max_seq_len, int dim, float theta
) {
    if (!cos_out || !sin_out) return PARALLAX_STATUS_NULL_POINTER;
    if (max_seq_len <= 0 || dim <= 0 || (dim % 2 != 0) || theta <= 0.0f) {
        return PARALLAX_STATUS_INVALID_DIMENSION;
    }
    int half = dim / 2;
    for (int t = 0; t < max_seq_len; ++t) {
        float pos = (float)t;
        for (int i = 0; i < half; ++i) {
            float exp_val = (float)(2 * i) / (float)dim;
            float angle = pos / powf(theta, exp_val);
            cos_out[t * half + i] = cosf(angle);
            sin_out[t * half + i] = sinf(angle);
        }
    }
    return PARALLAX_STATUS_OK;
}

static inline void apply_rope_scalar_step(
    float *head, const float *cos_ptr, const float *sin_ptr,
    int half, int interleaved
) {
    for (int i = 0; i < half; ++i) {
        float c = cos_ptr[i], s = sin_ptr[i];
        int idx1 = interleaved ? 2 * i : i;
        int idx2 = interleaved ? 2 * i + 1 : half + i;
        float v1 = head[idx1], v2 = head[idx2];
        head[idx1] = v1 * c - v2 * s;
        head[idx2] = v1 * s + v2 * c;
    }
}
