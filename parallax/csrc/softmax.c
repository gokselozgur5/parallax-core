#include <stddef.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

/*
 * Numerically stable Softmax along the last dimension (dim = cols)
 * input: rows x cols (row-major)
 * output: rows x cols (row-major)
 */
void c_softmax_f32(const float *input, float *output, int rows, int cols) {
    if (!input || !output || rows <= 0 || cols <= 0) return;

    for (int r = 0; r < rows; ++r) {
        const float *in_row = input + r * cols;
        float *out_row = output + r * cols;

        // Step 1: Find max value for numerical stability
        float max_val = in_row[0];
        int c = 1;
#if defined(USE_NEON)
        float32x4_t max_vec = vdupq_n_f32(max_val);
        for (; c + 3 < cols; c += 4) {
            float32x4_t v = vld1q_f32(in_row + c);
            max_vec = vmaxq_f32(max_vec, v);
        }
        float tmp[4];
        vst1q_f32(tmp, max_vec);
        for (int k = 0; k < 4; ++k) {
            if (tmp[k] > max_val) max_val = tmp[k];
        }
#endif
        for (; c < cols; ++c) {
            if (in_row[c] > max_val) {
                max_val = in_row[c];
            }
        }

        // Step 2: Compute exp(x - max_val) and accumulate sum
        float sum = 0.0f;
        for (c = 0; c < cols; ++c) {
            float exp_val = expf(in_row[c] - max_val);
            out_row[c] = exp_val;
            sum += exp_val;
        }

        // Step 3: Normalize
        float inv_sum = 1.0f / sum;
        c = 0;
#if defined(USE_NEON)
        float32x4_t inv_vec = vdupq_n_f32(inv_sum);
        for (; c + 3 < cols; c += 4) {
            float32x4_t v = vld1q_f32(out_row + c);
            v = vmulq_f32(v, inv_vec);
            vst1q_f32(out_row + c, v);
        }
#endif
        for (; c < cols; ++c) {
            out_row[c] *= inv_sum;
        }
    }
}
