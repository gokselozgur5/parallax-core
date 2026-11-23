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
    const size_t seq_s = (size_t)max_seq_len, dim_s = (size_t)dim, half = dim_s / 2;
    for (size_t t = 0; t < seq_s; ++t) {
        float pos = (float)t;
        for (size_t i = 0; i < half; ++i) {
            float exp_val = (float)(2 * i) / (float)dim_s;
            float angle = pos / powf(theta, exp_val);
            cos_out[t * half + i] = cosf(angle);
            sin_out[t * half + i] = sinf(angle);
        }
    }
    return PARALLAX_STATUS_OK;
}

static inline void apply_rope_scalar_step(
    float *head, const float *cos_ptr, const float *sin_ptr,
    size_t half, int interleaved
) {
    for (size_t i = 0; i < half; ++i) {
        float c = cos_ptr[i], s = sin_ptr[i];
        size_t idx1 = interleaved ? 2 * i : i;
        size_t idx2 = interleaved ? 2 * i + 1 : half + i;
        float v1 = head[idx1], v2 = head[idx2];
        head[idx1] = v1 * c - v2 * s;
        head[idx2] = v1 * s + v2 * c;
    }
}

#if defined(PARALLAX_HAVE_NEON)
static inline void apply_rope_neon_step(
    float *head, const float *cos_ptr, const float *sin_ptr,
    size_t half, int interleaved
) {
    size_t i = 0;
    if (interleaved) {
        for (; half - i >= 4; i += 4) {
            float32x4_t c = vld1q_f32(cos_ptr + i);
            float32x4_t s = vld1q_f32(sin_ptr + i);
            float32x4x2_t v = vld2q_f32(head + 2 * i);
            float32x4_t r1 = vsubq_f32(vmulq_f32(v.val[0], c), vmulq_f32(v.val[1], s));
            float32x4_t r2 = vaddq_f32(vmulq_f32(v.val[0], s), vmulq_f32(v.val[1], c));
            float32x4x2_t res;
            res.val[0] = r1;
            res.val[1] = r2;
            vst2q_f32(head + 2 * i, res);
        }
    } else {
        for (; half - i >= 4; i += 4) {
            float32x4_t c = vld1q_f32(cos_ptr + i);
            float32x4_t s = vld1q_f32(sin_ptr + i);
            float32x4_t v1 = vld1q_f32(head + i);
            float32x4_t v2 = vld1q_f32(head + half + i);
            float32x4_t r1 = vsubq_f32(vmulq_f32(v1, c), vmulq_f32(v2, s));
            float32x4_t r2 = vaddq_f32(vmulq_f32(v1, s), vmulq_f32(v2, c));
            vst1q_f32(head + i, r1);
            vst1q_f32(head + half + i, r2);
        }
    }
}
#endif

static inline void apply_rope_vector_step(
    float *head, const float *cos_ptr, const float *sin_ptr,
    size_t half, int interleaved
) {
#if defined(PARALLAX_HAVE_NEON)
    apply_rope_neon_step(head, cos_ptr, sin_ptr, half, interleaved);
    size_t tail_start = (half / 4) * 4;
    for (size_t i = tail_start; i < half; ++i) {
        float c = cos_ptr[i], s = sin_ptr[i];
        size_t idx1 = interleaved ? 2 * i : i;
        size_t idx2 = interleaved ? 2 * i + 1 : half + i;
        float v1 = head[idx1], v2 = head[idx2];
        head[idx1] = v1 * c - v2 * s;
        head[idx2] = v1 * s + v2 * c;
    }
#else
    apply_rope_scalar_step(head, cos_ptr, sin_ptr, half, interleaved);
#endif
}

int c_apply_rope_cached_f32(
    float *x, const float *cos_cache, const float *sin_cache,
    int T, int H, int D, int start_pos, int max_seq_len, int interleaved
) {
    if (!x || !cos_cache || !sin_cache) return PARALLAX_STATUS_NULL_POINTER;
    if (T < 0 || H < 0 || D < 0 || (D % 2 != 0) || start_pos < 0 || max_seq_len < 0) {
        return PARALLAX_STATUS_INVALID_DIMENSION;
    }
    if (T == 0 || H == 0 || D == 0) return PARALLAX_STATUS_OK;
    if ((size_t)start_pos + (size_t)T > (size_t)max_seq_len) return PARALLAX_STATUS_INVALID_DIMENSION;
    const size_t Ts = (size_t)T, Hs = (size_t)H, Ds = (size_t)D, half = Ds / 2;
    for (size_t t = 0; t < Ts; ++t) {
        size_t pos = (size_t)start_pos + t;
        const float *c_ptr = cos_cache + pos * half;
        const float *s_ptr = sin_cache + pos * half;
        for (size_t h = 0; h < Hs; ++h) {
            float *head = x + (t * Hs + h) * Ds;
            apply_rope_vector_step(head, c_ptr, s_ptr, half, interleaved);
        }
    }
    return PARALLAX_STATUS_OK;
}
