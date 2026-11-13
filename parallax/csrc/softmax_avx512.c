#include <stddef.h>
#include <stdint.h>
#include <math.h>

#if __has_include("parallax_simd.h")
#include "parallax_simd.h"
#endif

#ifndef PARALLAX_STATUS_OK
#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER (-1)
#define PARALLAX_STATUS_INVALID_DIMENSION (-2)
#define PARALLAX_STATUS_UNSUPPORTED_ISA (-5)
#endif

int c_has_avx512_softmax_f32(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f");
#else
    return 0;
#endif
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,avx512vl,avx512bw")))
static void softmax_avx512_kernel(const float *in, float *out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float *in_row = in + r * cols;
        float *out_row = out + r * cols;
        float max_val = in_row[0];
        int c = 1;
        __m512 max_vec = _mm512_set1_ps(max_val);
        for (; cols - c >= 16; c += 16) {
            max_vec = _mm512_max_ps(max_vec, _mm512_loadu_ps(in_row + c));
        }
        float tmp[16];
        _mm512_storeu_ps(tmp, max_vec);
        for (int k = 0; k < 16; ++k) if (tmp[k] > max_val) max_val = tmp[k];
        for (; c < cols; ++c) if (in_row[c] > max_val) max_val = in_row[c];
        double sum = 0.0;
        for (c = 0; c < cols; ++c) {
            out_row[c] = expf(in_row[c] - max_val);
            sum += (double)out_row[c];
        }
        for (c = 0; c < cols; ++c) out_row[c] = (float)((double)out_row[c] / sum);
    }
}
#endif

