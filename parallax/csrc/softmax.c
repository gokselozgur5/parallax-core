#include "parallax_simd.h"
#include <math.h>

PARALLAX_SCALAR_TARGET
void c_softmax_scalar_f32(const float *input, float *output, int rows, int cols) {
    if (!input || !output || rows <= 0 || cols <= 0) return;

    const size_t rows_s = (size_t)rows;
    const size_t cols_s = (size_t)cols;
    for (size_t r = 0; r < rows_s; ++r) {
        const float *in_row = input + r * cols_s;
        float *out_row = output + r * cols_s;

        float max_val = in_row[0];
        for (size_t c = 1; c < cols_s; ++c) {
            if (in_row[c] > max_val) {
                max_val = in_row[c];
            }
        }

        double sum = 0.0;
        for (size_t c = 0; c < cols_s; ++c) {
            float exp_val = expf(in_row[c] - max_val);
            out_row[c] = exp_val;
            sum += (double)exp_val;
        }

        double inv_sum = 1.0 / sum;
        for (size_t c = 0; c < cols_s; ++c) {
            out_row[c] = (float)((double)out_row[c] * inv_sum);
        }
    }
}

void c_softmax_f32(const float *input, float *output, int rows, int cols) {
    if (!input || !output || rows <= 0 || cols <= 0) return;
    if (c_has_avx512_softmax_f32() && c_softmax_avx512_f32(input, output, rows, cols) == PARALLAX_STATUS_OK) return;
    if (c_has_avx2_softmax() && c_softmax_avx2_f32(input, output, rows, cols) == PARALLAX_STATUS_OK) return;
    if (c_softmax_neon_f32(input, output, rows, cols) == PARALLAX_STATUS_OK) return;
    c_softmax_scalar_f32(input, output, rows, cols);
}
