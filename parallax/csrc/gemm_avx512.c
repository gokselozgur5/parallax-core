#include "parallax_simd.h"

PARALLAX_SCALAR_TARGET
int c_has_avx512_f32(void) {
    return PARALLAX_HAS_AVX512_ISA();
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((noinline, target("avx512f,avx512dq,avx512vl,avx512bw")))
static void avx512_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const size_t Ms = (size_t)M, Ns = (size_t)N, Ks = (size_t)K;
    for (size_t i = 0; i < Ms; ++i) {
        float *c_row = C + i * Ns;
        for (size_t j = 0; j < Ns; ++j) c_row[j] = 0.0f;
        for (size_t k = 0; k < Ks; ++k) {
            float a_val = A[i * Ks + k];
            const float *b_row = B + k * Ns;
            size_t j = 0;
            __m512 a_vec = _mm512_set1_ps(a_val);
            for (; Ns - j >= 16; j += 16) {
                __m512 c_vec = _mm512_loadu_ps(c_row + j);
                __m512 b_vec = _mm512_loadu_ps(b_row + j);
                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                _mm512_storeu_ps(c_row + j, c_vec);
            }
            for (; j < Ns; ++j) c_row[j] += a_val * b_row[j];
        }
    }
}
#endif

PARALLAX_SCALAR_TARGET
int c_gemm_avx512_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C) return PARALLAX_STATUS_NULL_POINTER;
    if (M <= 0 || N <= 0 || K < 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (!c_has_avx512_f32()) return PARALLAX_STATUS_UNSUPPORTED_ISA;
    if (K == 0) {
        const size_t total = (size_t)M * (size_t)N;
        for (size_t i = 0; i < total; ++i) C[i] = 0.0f;
        return PARALLAX_STATUS_OK;
    }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    avx512_kernel(A, B, C, M, N, K);
    return PARALLAX_STATUS_OK;
#else
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
#endif
}


