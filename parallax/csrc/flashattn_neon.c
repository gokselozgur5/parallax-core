#include "parallax_simd.h"
#include <math.h>

#if !defined(PARALLAX_HAVE_NEON)
int c_flashattn_neon_f32(const float *q, const float *k, const float *v,
                         float *out, int seq_len, int head_dim,
                         float scale, int is_causal) {
    (void)q; (void)k; (void)v; (void)out; (void)seq_len;
    (void)head_dim; (void)scale; (void)is_causal;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
}
#else

static inline float dot_product_neon(const float *a, const float *b, int len) {
    int d = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; len - d >= 4; d += 4) {
        float32x4_t va = vld1q_f32(a + d);
        float32x4_t vb = vld1q_f32(b + d);
        sum_vec = vmlaq_f32(sum_vec, va, vb);
    }
    float total = vaddvq_f32(sum_vec);
    for (; d < len; ++d) total += a[d] * b[d];
    return total;
}
