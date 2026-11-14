#include <stddef.h>
#include <stdint.h>
#include <math.h>

#if __has_include("parallax_simd.h")
#include "parallax_simd.h"
#endif

#ifndef PARALLAX_STATUS_OK
#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER (-1)
#define PARALLAX_STATUS_INVALID_DIMENSION (-2)
#define PARALLAX_STATUS_UNSUPPORTED_ISA (-5)
#endif

int c_has_avx512_flashattn_f32(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512bw");
#else
    return 0;
#endif
}


#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
static float dot_product_avx512(const float *a, const float *b, int len) {
    int d = 0;
    __m512 sum_vec = _mm512_setzero_ps();
    for (; len - d >= 16; d += 16) {
        __m512 va = _mm512_loadu_ps(a + d);
        __m512 vb = _mm512_loadu_ps(b + d);
        sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
    }
    float total = _mm512_reduce_add_ps(sum_vec);
    for (; d < len; ++d) total += a[d] * b[d];
    return total;
}
#endif


#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
static void vec_scale_add_avx512(float *out, float alpha, const float *v, float p, int len) {
    int d = 0;
    __m512 valpha = _mm512_set1_ps(alpha);
    __m512 vp = _mm512_set1_ps(p);
    for (; len - d >= 16; d += 16) {
        __m512 vo = _mm512_loadu_ps(out + d);
        __m512 vv = _mm512_loadu_ps(v + d);
        vo = _mm512_mul_ps(vo, valpha);
        vo = _mm512_fmadd_ps(vp, vv, vo);
        _mm512_storeu_ps(out + d, vo);
    }
    for (; d < len; ++d) out[d] = out[d] * alpha + p * v[d];
}
#endif


#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
static void vec_scale_avx512(float *out, float scale, int len) {
    int d = 0;
    __m512 vscale = _mm512_set1_ps(scale);
    for (; len - d >= 16; d += 16) {
        __m512 vo = _mm512_loadu_ps(out + d);
        vo = _mm512_mul_ps(vo, vscale);
        _mm512_storeu_ps(out + d, vo);
    }
    for (; d < len; ++d) out[d] *= scale;
}
#endif


#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
static void update_kv_step_avx512(float *oi, const float *vj, float score,
                                  float *m_i, float *l_i, int head_dim) {
    if (isnan(score) || (isinf(score) && score < 0.0f)) return;
    if (score > *m_i) {
        float alpha = expf(*m_i - score);
        *m_i = score;
        *l_i = *l_i * alpha + 1.0f;
        vec_scale_add_avx512(oi, alpha, vj, 1.0f, head_dim);
    } else {
        float p = expf(score - *m_i);
        *l_i += p;
        vec_scale_add_avx512(oi, 1.0f, vj, p, head_dim);
    }
}
#endif


#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
static void flashattn_avx512_kernel(const float *q, const float *k, const float *v,
                                    float *out, int seq_len, int head_dim,
                                    float scale, int is_causal) {
    for (int i = 0; i < seq_len; ++i) {
        const float *qi = q + i * head_dim;
        float *oi = out + i * head_dim;
        for (int d = 0; d < head_dim; ++d) oi[d] = 0.0f;
        float m_i = -INFINITY, l_i = 0.0f;
        int max_j = is_causal ? (i + 1) : seq_len;
        for (int j = 0; j < max_j; ++j) {
            float s = dot_product_avx512(qi, k + j * head_dim, head_dim) * scale;
            update_kv_step_avx512(oi, v + j * head_dim, s, &m_i, &l_i, head_dim);
        }
        if (l_i > 0.0f) vec_scale_avx512(oi, 1.0f / l_i, head_dim);
    }
}
#endif


int c_flashattn_avx512_f32(const float *q, const float *k, const float *v,
                           float *out, int seq_len, int head_dim,
                           float scale, int is_causal) {
    if (!q || !k || !v || !out) return PARALLAX_STATUS_NULL_POINTER;
    if (seq_len <= 0 || head_dim <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (!c_has_avx512_flashattn_f32()) return PARALLAX_STATUS_UNSUPPORTED_ISA;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    flashattn_avx512_kernel(q, k, v, out, seq_len, head_dim, scale, is_causal);
    return PARALLAX_STATUS_OK;
#else
    (void)scale;
    (void)is_causal;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
#endif
}
