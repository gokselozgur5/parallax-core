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

static inline float dot_product_neon(const float *a, const float *b, size_t len) {
    size_t d = 0;
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

static inline void vec_scale_add_neon(float *out, float alpha, const float *v, float p, size_t len) {
    size_t d = 0;
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

static inline void vec_scale_neon(float *out, float scale, size_t len) {
    size_t d = 0;
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; len - d >= 4; d += 4) {
        float32x4_t vo = vld1q_f32(out + d);
        vo = vmulq_f32(vo, vscale);
        vst1q_f32(out + d, vo);
    }
    for (; d < len; ++d) out[d] *= scale;
}

static inline void update_kv_step_neon(float *oi, const float *vj, float score,
                                       float *m_i, float *l_i, size_t head_dim) {
    if (isinf(score) && score < 0.0f) return;
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

static void flashattn_neon_kernel(const float *q, const float *k, const float *v,
                                  float *out, int seq_len, int head_dim,
                                  float scale, int is_causal) {
    const size_t seq_s = (size_t)seq_len, dim_s = (size_t)head_dim;
    for (size_t i = 0; i < seq_s; ++i) {
        const float *qi = q + i * dim_s;
        float *oi = out + i * dim_s;
        for (size_t d = 0; d < dim_s; ++d) oi[d] = 0.0f;
        float m_i = -INFINITY, l_i = 0.0f;
        size_t max_j = is_causal ? (i + 1) : seq_s;
        for (size_t j = 0; j < max_j; ++j) {
            float s = dot_product_neon(qi, k + j * dim_s, dim_s) * scale;
            update_kv_step_neon(oi, v + j * dim_s, s, &m_i, &l_i, dim_s);
        }
        if (l_i > 0.0f) vec_scale_neon(oi, 1.0f / l_i, dim_s);
    }
}

int c_flashattn_neon_f32(const float *q, const float *k, const float *v,
                         float *out, int seq_len, int head_dim,
                         float scale, int is_causal) {
    if (!q || !k || !v || !out) return PARALLAX_STATUS_NULL_POINTER;
    if (seq_len <= 0 || head_dim <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    flashattn_neon_kernel(q, k, v, out, seq_len, head_dim, scale, is_causal);
    return PARALLAX_STATUS_OK;
}
#endif
