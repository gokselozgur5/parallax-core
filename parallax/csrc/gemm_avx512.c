#include <stddef.h>
#include <stdint.h>

#if __has_include("parallax_simd.h")
#include "parallax_simd.h"
#endif

#ifndef PARALLAX_STATUS_OK
#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER (-1)
#define PARALLAX_STATUS_INVALID_DIMENSION (-2)
#define PARALLAX_STATUS_UNSUPPORTED_ISA (-5)
#endif

int c_has_avx512_f32(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f");
#else
    return 0;
#endif
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
static void avx512_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        float *c_row = C + i * N;
        for (int j = 0; j < N; ++j) c_row[j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            const float *b_row = B + k * N;
            int j = 0;
            __m512 a_vec = _mm512_set1_ps(a_val);
            for (; N - j >= 16; j += 16) {
                __m512 c_vec = _mm512_loadu_ps(c_row + j);
                __m512 b_vec = _mm512_loadu_ps(b_row + j);
                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                _mm512_storeu_ps(c_row + j, c_vec);
            }
            for (; j < N; ++j) c_row[j] += a_val * b_row[j];
        }
    }
}
#endif

int c_gemm_avx512_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C) return PARALLAX_STATUS_NULL_POINTER;
    if (M <= 0 || N <= 0 || K < 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (!c_has_avx512_f32()) return PARALLAX_STATUS_UNSUPPORTED_ISA;
    if (K == 0) {
        for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
        return PARALLAX_STATUS_OK;
    }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    avx512_kernel(A, B, C, M, N, K);
    return PARALLAX_STATUS_OK;
#else
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
#endif
}


