#include "parallax_simd.h"
#include <string.h>

#if !defined(PARALLAX_HAVE_NEON)
int c_gemm_neon_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
}
#else

int c_gemm_neon_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C) return PARALLAX_STATUS_NULL_POINTER;
    if (M < 0 || N < 0 || K < 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (M == 0 || N == 0) return PARALLAX_STATUS_OK;
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    if (K == 0) return PARALLAX_STATUS_OK;
    return PARALLAX_STATUS_OK;
}
#endif
