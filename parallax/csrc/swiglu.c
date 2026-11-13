#include <math.h>

void c_swiglu_f32(const float *a, const float *b, float *out, int n) {
    if (!a || !b || !out || n <= 0) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        float x = a[i];
        float silu_val;
        if (x <= -88.0f) {
            silu_val = 0.0f;
        } else if (x >= 88.0f) {
            silu_val = x;
        } else {
            silu_val = x / (1.0f + expf(-x));
        }
        out[i] = silu_val * b[i];
    }
}
