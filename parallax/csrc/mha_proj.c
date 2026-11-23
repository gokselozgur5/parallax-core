/* Fused Multi-Head Attention QKV projection splitter and transposer. */
#include <stddef.h>

int c_mha_split_qkv_f32(const float *qkv, float *q, float *k, float *v,
                        int b, int t, int hq, int hkv, int d) {
    if (!qkv || !q || !k || !v) return -1;
    if (b <= 0 || t <= 0 || hq <= 0 || hkv <= 0 || d <= 0) return -1;
    const size_t bs = (size_t)b, ts = (size_t)t;
    const size_t hqs = (size_t)hq, hkvs = (size_t)hkv, ds = (size_t)d;
    const size_t fused_dim = (hqs + 2 * hkvs) * ds;
    const size_t q_offset = 0;
    const size_t k_offset = hqs * ds;
    const size_t v_offset = (hqs + hkvs) * ds;
    for (size_t bi = 0; bi < bs; ++bi) {
        for (size_t ti = 0; ti < ts; ++ti) {
            const float *src = qkv + (bi * ts + ti) * fused_dim;
            for (size_t h = 0; h < hqs; ++h) {
                float *dst_q = q + ((bi * hqs + h) * ts + ti) * ds;
                const float *src_q = src + q_offset + h * ds;
                for (size_t i = 0; i < ds; ++i) dst_q[i] = src_q[i];
            }
            for (size_t h = 0; h < hkvs; ++h) {
                float *dst_k = k + ((bi * hkvs + h) * ts + ti) * ds;
                const float *src_k = src + k_offset + h * ds;
                for (size_t i = 0; i < ds; ++i) dst_k[i] = src_k[i];
                float *dst_v = v + ((bi * hkvs + h) * ts + ti) * ds;
                const float *src_v = src + v_offset + h * ds;
                for (size_t i = 0; i < ds; ++i) dst_v[i] = src_v[i];
            }
        }
    }
    return 0;
}
