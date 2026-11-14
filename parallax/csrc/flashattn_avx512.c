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
    return __builtin_cpu_supports("avx512f");
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
