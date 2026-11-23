#include "parallax_simd.h"
#include <string.h>

#if !defined(PARALLAX_HAVE_NEON)
int c_gemm_neon_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
}
#else

static void gemm_neon_8x8(const float *A, const float *B, float *C, size_t N, size_t K, size_t i, size_t j) {
    float32x4_t c[8][2];
    for (size_t r = 0; r < 8; ++r) {
        c[r][0] = vdupq_n_f32(0.0f);
        c[r][1] = vdupq_n_f32(0.0f);
    }
    for (size_t p = 0; p < K; ++p) {
        float32x4_t b0 = vld1q_f32(&B[p * N + j]);
        float32x4_t b1 = vld1q_f32(&B[p * N + j + 4]);
        for (size_t r = 0; r < 8; ++r) {
            c[r][0] = vmlaq_n_f32(c[r][0], b0, A[(i + r) * K + p]);
            c[r][1] = vmlaq_n_f32(c[r][1], b1, A[(i + r) * K + p]);
        }
    }
    for (size_t r = 0; r < 8; ++r) {
        vst1q_f32(&C[(i + r) * N + j], c[r][0]);
        vst1q_f32(&C[(i + r) * N + j + 4], c[r][1]);
    }
}

int c_gemm_neon_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C) return PARALLAX_STATUS_NULL_POINTER;
    if (M < 0 || N < 0 || K < 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (M == 0 || N == 0) return PARALLAX_STATUS_OK;
    const size_t Ms = (size_t)M, Ns = (size_t)N, Ks = (size_t)K;
    memset(C, 0, Ms * Ns * sizeof(float));
    if (K == 0) return PARALLAX_STATUS_OK;
    size_t i = 0;
    for (; Ms - i >= 8; i += 8) {
        size_t j = 0;
        for (; Ns - j >= 8; j += 8) gemm_neon_8x8(A, B, C, Ns, Ks, i, j);
        for (; j < Ns; ++j)
            for (size_t r = 0; r < 8; ++r)
                for (size_t p = 0; p < Ks; ++p)
                    C[(i + r) * Ns + j] += A[(i + r) * Ks + p] * B[p * Ns + j];
    }
    for (; i < Ms; ++i)
        for (size_t j = 0; j < Ns; ++j)
            for (size_t p = 0; p < Ks; ++p)
                C[i * Ns + j] += A[i * Ks + p] * B[p * Ns + j];
    return PARALLAX_STATUS_OK;
}
#endif
