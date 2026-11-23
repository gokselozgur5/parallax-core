#include <math.h>
#include <stddef.h>

void c_layernorm_f32(const float *x, const float *weight, const float *bias, float *output, int rows, int cols, float eps) {
    if (!x || !output || rows <= 0 || cols <= 0) return;
    const size_t rows_s = (size_t)rows, cols_s = (size_t)cols;
    for (size_t r = 0; r < rows_s; ++r) {
        const float *in = x + r * cols_s;
        float *out = output + r * cols_s;
        double sum = 0.0;
        for (size_t c = 0; c < cols_s; ++c) sum += (double)in[c];
        double mean = sum / (double)cols_s;
        double var_sum = 0.0;
        for (size_t c = 0; c < cols_s; ++c) {
            double d = (double)in[c] - mean;
            var_sum += d * d;
        }
        float inv_std = (float)(1.0 / sqrt(var_sum / (double)cols_s + (double)eps));
        float f_mean = (float)mean;
        for (size_t c = 0; c < cols_s; ++c) {
            float w = weight ? weight[c] : 1.0f;
            float b = bias ? bias[c] : 0.0f;
            out[c] = (in[c] - f_mean) * inv_std * w + b;
        }
    }
}
