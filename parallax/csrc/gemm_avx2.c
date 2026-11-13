#include <stddef.h>

#if (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void c_gemm_avx2_f32(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K
) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K < 0) return;
    const size_t Ms = (size_t)M;
    const size_t Ns = (size_t)N;
    const size_t Ks = (size_t)K;
