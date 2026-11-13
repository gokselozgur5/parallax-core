#include <stddef.h>
#include <math.h>

#if (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void c_softmax_avx2_f32(
    const float *input,
    float *output,
    int rows,
    int cols
) {
    if (!input || !output || rows <= 0 || cols <= 0) return;
    const size_t rows_s = (size_t)rows;
    const size_t cols_s = (size_t)cols;

    for (size_t r = 0; r < rows_s; ++r) {
        const float *in_row = input + r * cols_s;
        float *out_row = output + r * cols_s;
        float max_val = in_row[0];
        size_t c = 0;
        if (cols_s >= 8) {
            __m256 max_vec = _mm256_loadu_ps(in_row);
            c = 8;
            for (; cols_s - c >= 8; c += 8) {
                __m256 v = _mm256_loadu_ps(in_row + c);
                max_vec = _mm256_max_ps(max_vec, v);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, max_vec);
            for (int k = 0; k < 8; ++k) {
                if (tmp[k] > max_val) max_val = tmp[k];
            }
        }
        for (; c < cols_s; ++c) {
            if (in_row[c] > max_val) max_val = in_row[c];
        }
        double sum = 0.0;
        for (c = 0; c < cols_s; ++c) {
            float exp_val = expf(in_row[c] - max_val);
            out_row[c] = exp_val;
            sum += (double)exp_val;
        }
