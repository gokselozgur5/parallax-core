#include "parallax_simd.h"
#include <string.h>

#if !defined(PARALLAX_HAVE_NEON)
int c_gemm_neon_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
}
#else

static void gemm_neon_8x8(const float *A, const float *B, float *C, int N, int K, int i, int j) {
    float32x4_t c[8][2];
    for (int r = 0; r < 8; ++r) {
        c[r][0] = vdupq_n_f32(0.0f);
        c[r][1] = vdupq_n_f32(0.0f);
    }
    for (int p = 0; p < K; ++p) {
        float32x4_t b0 = vld1q_f32(&B[p * N + j]);
        float32x4_t b1 = vld1q_f32(&B[p * N + j + 4]);
        for (int r = 0; r < 8; ++r) {
            c[r][0] = vmlaq_n_f32(c[r][0], b0, A[(i + r) * K + p]);
            c[r][1] = vmlaq_n_f32(c[r][1], b1, A[(i + r) * K + p]);
        }
    }
    for (int r = 0; r < 8; ++r) {
        vst1q_f32(&C[(i + r) * N + j], c[r][0]);
        vst1q_f32(&C[(i + r) * N + j + 4], c[r][1]);
    }
}

int c_gemm_neon_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C) return PARALLAX_STATUS_NULL_POINTER;
    if (M < 0 || N < 0 || K < 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (M == 0 || N == 0) return PARALLAX_STATUS_OK;
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    if (K == 0) return PARALLAX_STATUS_OK;
    int i = 0;
    for (; i + 8 <= M; i += 8) {
        int j = 0;
        for (; j + 8 <= N; j += 8) gemm_neon_8x8(A, B, C, N, K, i, j);
        for (; j < N; ++j)
            for (int r = 0; r < 8; ++r)
                for (int p = 0; p < K; ++p)
                    C[(i + r) * N + j] += A[(i + r) * K + p] * B[p * N + j];
    }
    for (; i < M; ++i)
        for (int j = 0; j < N; ++j)
            for (int p = 0; p < K; ++p)
                C[i * N + j] += A[i * K + p] * B[p * N + j];
    return PARALLAX_STATUS_OK;
}
#endif
