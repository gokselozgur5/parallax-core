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
#else
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
