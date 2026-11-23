#include "parallax_simd.h"
#include <math.h>

PARALLAX_SCALAR_TARGET
int c_has_avx512_softmax_f32(void) {
    return PARALLAX_HAS_AVX512_ISA();
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((noinline, target("avx512f,avx512dq,avx512vl,avx512bw")))
static void softmax_avx512_kernel(const float *in, float *out, int rows, int cols) {
    const size_t rows_s = (size_t)rows, cols_s = (size_t)cols;
    for (size_t r = 0; r < rows_s; ++r) {
        const float *in_row = in + r * cols_s;
        float *out_row = out + r * cols_s;
        float max_val = in_row[0];
        size_t c = 1;
        __m512 max_vec = _mm512_set1_ps(max_val);
        for (; cols_s - c >= 16; c += 16) {
            max_vec = _mm512_max_ps(max_vec, _mm512_loadu_ps(in_row + c));
        }
        float tmp[16];
        _mm512_storeu_ps(tmp, max_vec);
        for (size_t k = 0; k < 16; ++k) if (tmp[k] > max_val) max_val = tmp[k];
        for (; c < cols_s; ++c) if (in_row[c] > max_val) max_val = in_row[c];
        double sum = 0.0;
        for (c = 0; c < cols_s; ++c) {
            out_row[c] = expf(in_row[c] - max_val);
            sum += (double)out_row[c];
        }
        for (c = 0; c < cols_s; ++c) out_row[c] = (float)((double)out_row[c] / sum);
    }
}
#endif

PARALLAX_SCALAR_TARGET
int c_softmax_avx512_f32(const float *in, float *out, int rows, int cols) {
    if (!in || !out) return PARALLAX_STATUS_NULL_POINTER;
    if (rows <= 0 || cols <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (!c_has_avx512_softmax_f32()) return PARALLAX_STATUS_UNSUPPORTED_ISA;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    softmax_avx512_kernel(in, out, rows, cols);
    return PARALLAX_STATUS_OK;
#else
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
#endif
}


