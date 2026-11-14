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
