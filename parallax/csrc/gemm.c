#include "parallax_simd.h"

PARALLAX_SCALAR_TARGET
void c_gemm_scalar_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K < 0) return;
    const size_t Ms = (size_t)M;
    const size_t Ns = (size_t)N;
    const size_t Ks = (size_t)K;
    if (K == 0) {
        const size_t total = Ms * Ns;
        for (size_t i = 0; i < total; ++i) {
            C[i] = 0.0f;
        }
        return;
    }

    for (size_t i = 0; i < Ms; ++i) {
        const float *a_row = A + i * Ks;
        float *c_row = C + i * Ns;
        for (size_t j = 0; j < Ns; ++j) {
            c_row[j] = 0.0f;
        }
        for (size_t k = 0; k < Ks; ++k) {
            float a_val = a_row[k];
            const float *b_row = B + k * Ns;
            size_t j = 0;
            for (; Ns - j >= 4; j += 4) {
                c_row[j]     += a_val * b_row[j];
                c_row[j + 1] += a_val * b_row[j + 1];
                c_row[j + 2] += a_val * b_row[j + 2];
                c_row[j + 3] += a_val * b_row[j + 3];
            }
            for (; j < Ns; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

void c_gemm_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K < 0) return;
    if (c_has_avx512_f32() && c_gemm_avx512_f32(A, B, C, M, N, K) == PARALLAX_STATUS_OK) return;
    if (c_has_avx2_gemm() && c_gemm_avx2_f32(A, B, C, M, N, K) == PARALLAX_STATUS_OK) return;
    if (c_gemm_neon_f32(A, B, C, M, N, K) == PARALLAX_STATUS_OK) return;
    c_gemm_scalar_f32(A, B, C, M, N, K);
}
