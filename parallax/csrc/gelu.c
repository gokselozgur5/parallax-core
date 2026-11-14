#include <math.h>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEF 0.044715f

void c_gelu_f32(const float *x, float *out, int n) {
    if (!x || !out || n <= 0) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        float val = x[i];
        if (val <= -10.0f) {
            out[i] = 0.0f;
        } else if (val >= 10.0f) {
            out[i] = val;
        } else {
            float inner = SQRT_2_OVER_PI * (val + COEF * val * val * val);
            out[i] = 0.5f * val * (1.0f + tanhf(inner));
        }
    }
}
