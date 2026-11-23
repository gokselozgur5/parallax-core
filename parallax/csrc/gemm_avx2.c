#include <stddef.h>

#ifndef PARALLAX_STATUS_OK
#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER (-1)
#define PARALLAX_STATUS_INVALID_DIMENSION (-2)
#define PARALLAX_STATUS_UNSUPPORTED_ISA (-5)
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define PARALLAX_NO_AVX_TARGET __attribute__((noinline, target("no-avx,no-fma")))
#else
#define PARALLAX_NO_AVX_TARGET
#endif

PARALLAX_NO_AVX_TARGET
int c_has_avx2_gemm(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((noinline, target("avx2,fma")))
static void gemm_avx2_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const size_t Ms = (size_t)M;
    const size_t Ns = (size_t)N;
    const size_t Ks = (size_t)K;
    if (K == 0) {
        for (size_t i = 0; i < Ms * Ns; ++i) C[i] = 0.0f;
        return;
    }
    for (size_t i = 0; i < Ms; ++i) {
        const float *a_row = A + i * Ks;
        float *c_row = C + i * Ns;
        for (size_t j = 0; j < Ns; ++j) c_row[j] = 0.0f;
        for (size_t k = 0; k < Ks; ++k) {
            float a_val = a_row[k];
            const float *b_row = B + k * Ns;
            __m256 a_vec = _mm256_set1_ps(a_val);
            size_t j = 0;
            for (; Ns - j >= 16; j += 16) {
                __m256 c0 = _mm256_loadu_ps(c_row + j);
                __m256 c1 = _mm256_loadu_ps(c_row + j + 8);
                __m256 b0 = _mm256_loadu_ps(b_row + j);
                __m256 b1 = _mm256_loadu_ps(b_row + j + 8);
                c0 = _mm256_fmadd_ps(a_vec, b0, c0);
                c1 = _mm256_fmadd_ps(a_vec, b1, c1);
                _mm256_storeu_ps(c_row + j, c0);
                _mm256_storeu_ps(c_row + j + 8, c1);
            }
            for (; Ns - j >= 8; j += 8) {
                __m256 c0 = _mm256_loadu_ps(c_row + j);
                __m256 b0 = _mm256_loadu_ps(b_row + j);
                c0 = _mm256_fmadd_ps(a_vec, b0, c0);
                _mm256_storeu_ps(c_row + j, c0);
            }
            for (; j < Ns; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}
#endif

PARALLAX_NO_AVX_TARGET
int c_gemm_avx2_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C) return PARALLAX_STATUS_NULL_POINTER;
    if (M <= 0 || N <= 0 || K < 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (!c_has_avx2_gemm()) return PARALLAX_STATUS_UNSUPPORTED_ISA;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    gemm_avx2_kernel(A, B, C, M, N, K);
    return PARALLAX_STATUS_OK;
#else
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
#endif
}
