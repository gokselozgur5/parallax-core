/* Fused Multi-Head Attention QKV projection splitter and transposer. */
#include <stddef.h>

int c_mha_split_qkv_f32(const float *qkv, float *q, float *k, float *v,
                        int b, int t, int hq, int hkv, int d) {
    if (!qkv || !q || !k || !v) return -1;
    if (b <= 0 || t <= 0 || hq <= 0 || hkv <= 0 || d <= 0) return -1;
    long fused_dim = (long)(hq + 2 * hkv) * d;
    long q_offset = 0;
    long k_offset = (long)hq * d;
    long v_offset = (long)(hq + hkv) * d;
    for (int bi = 0; bi < b; ++bi) {
        for (int ti = 0; ti < t; ++ti) {
            const float *src = qkv + ((long)bi * t + ti) * fused_dim;
            for (int h = 0; h < hq; ++h) {
                float *dst_q = q + (((long)bi * hq + h) * t + ti) * d;
                const float *src_q = src + q_offset + (long)h * d;
                for (int i = 0; i < d; ++i) dst_q[i] = src_q[i];
            }
            for (int h = 0; h < hkv; ++h) {
                float *dst_k = k + (((long)bi * hkv + h) * t + ti) * d;
                const float *src_k = src + k_offset + (long)h * d;
                for (int i = 0; i < d; ++i) dst_k[i] = src_k[i];
                float *dst_v = v + (((long)bi * hkv + h) * t + ti) * d;
                const float *src_v = src + v_offset + (long)h * d;
                for (int i = 0; i < d; ++i) dst_v[i] = src_v[i];
            }
        }
    }
    return 0;
}
