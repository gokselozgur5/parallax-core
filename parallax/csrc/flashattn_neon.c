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

static inline void vec_scale_add_neon(float *out, float alpha, const float *v, float p, int len) {
    int d = 0;
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t vp = vdupq_n_f32(p);
    for (; len - d >= 4; d += 4) {
        float32x4_t vo = vld1q_f32(out + d);
        float32x4_t vv = vld1q_f32(v + d);
        vo = vmulq_f32(vo, valpha);
        vo = vmlaq_f32(vo, vp, vv);
        vst1q_f32(out + d, vo);
    }
    for (; d < len; ++d) out[d] = out[d] * alpha + p * v[d];
}

static inline void vec_scale_neon(float *out, float scale, int len) {
    int d = 0;
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; len - d >= 4; d += 4) {
        float32x4_t vo = vld1q_f32(out + d);
        vo = vmulq_f32(vo, vscale);
        vst1q_f32(out + d, vo);
    }
    for (; d < len; ++d) out[d] *= scale;
}

static inline void update_kv_step_neon(float *oi, const float *vj, float score,
                                       float *m_i, float *l_i, int head_dim) {
    if (score > *m_i) {
        float alpha = expf(*m_i - score);
        *m_i = score;
        *l_i = *l_i * alpha + 1.0f;
        vec_scale_add_neon(oi, alpha, vj, 1.0f, head_dim);
    } else {
        float p = expf(score - *m_i);
        *l_i += p;
        vec_scale_add_neon(oi, 1.0f, vj, p, head_dim);
    }
}
