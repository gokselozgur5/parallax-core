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
