#include <stddef.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

#if defined(__AVX__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX 1
#endif

/*
 * C = A * B
 * A: M x K (row-major)
 * B: K x N (row-major)
 * C: M x N (row-major)
 */
void c_gemm_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    if (!A || !B || !C || M <= 0 || N <= 0) return;
    if (K <= 0) {
        for (int i = 0; i < M * N; ++i) {
            C[i] = 0.0f;
        }
        return;
    }

    for (int i = 0; i < M; ++i) {
        const float *a_row = A + i * K;
        float *c_row = C + i * N;

        // Initialize row of C with 0
        for (int j = 0; j < N; ++j) {
            c_row[j] = 0.0f;
        }

        for (int k = 0; k < K; ++k) {
            float a_val = a_row[k];
            const float *b_row = B + k * N;
            int j = 0;

#if defined(USE_NEON)
            float32x4_t a_vec = vdupq_n_f32(a_val);
            for (; j + 15 < N; j += 16) {
                float32x4_t c0 = vld1q_f32(c_row + j);
                float32x4_t c1 = vld1q_f32(c_row + j + 4);
                float32x4_t c2 = vld1q_f32(c_row + j + 8);
                float32x4_t c3 = vld1q_f32(c_row + j + 12);

                float32x4_t b0 = vld1q_f32(b_row + j);
                float32x4_t b1 = vld1q_f32(b_row + j + 4);
                float32x4_t b2 = vld1q_f32(b_row + j + 8);
                float32x4_t b3 = vld1q_f32(b_row + j + 12);

                c0 = vmlaq_f32(c0, a_vec, b0);
                c1 = vmlaq_f32(c1, a_vec, b1);
                c2 = vmlaq_f32(c2, a_vec, b2);
                c3 = vmlaq_f32(c3, a_vec, b3);

                vst1q_f32(c_row + j, c0);
                vst1q_f32(c_row + j + 4, c1);
                vst1q_f32(c_row + j + 8, c2);
                vst1q_f32(c_row + j + 12, c3);
            }
            for (; j + 3 < N; j += 4) {
                float32x4_t c0 = vld1q_f32(c_row + j);
                float32x4_t b0 = vld1q_f32(b_row + j);
                c0 = vmlaq_f32(c0, a_vec, b0);
                vst1q_f32(c_row + j, c0);
            }
#elif defined(USE_AVX)
            __m256 a_vec = _mm256_set1_ps(a_val);
            for (; j + 7 < N; j += 8) {
                __m256 c0 = _mm256_loadu_ps(c_row + j);
                __m256 b0 = _mm256_loadu_ps(b_row + j);
                c0 = _mm256_fmadd_ps(a_vec, b0, c0);
                _mm256_storeu_ps(c_row + j, c0);
            }
#endif

            // Remainder / scalar loop unrolling
            for (; j + 3 < N; j += 4) {
                c_row[j]     += a_val * b_row[j];
                c_row[j + 1] += a_val * b_row[j + 1];
                c_row[j + 2] += a_val * b_row[j + 2];
                c_row[j + 3] += a_val * b_row[j + 3];
            }
            for (; j < N; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}
