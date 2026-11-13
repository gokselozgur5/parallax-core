#include <math.h>

void c_layernorm_f32(const float *x, const float *weight, const float *bias, float *output, int rows, int cols, float eps) {
    if (!x || !output || rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const float *in = x + r * cols;
        float *out = output + r * cols;
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += (double)in[c];
        double mean = sum / (double)cols;
        double var_sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            double d = (double)in[c] - mean;
            var_sum += d * d;
        }
        float inv_std = (float)(1.0 / sqrt(var_sum / (double)cols + (double)eps));
        float f_mean = (float)mean;
        for (int c = 0; c < cols; ++c) {
            float w = weight ? weight[c] : 1.0f;
            float b = bias ? bias[c] : 0.0f;
            out[c] = (in[c] - f_mean) * inv_std * w + b;
        }
    }
}
