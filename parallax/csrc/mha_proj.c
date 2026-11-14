/* Fused Multi-Head Attention QKV projection splitter and transposer. */
#include <stddef.h>

int c_mha_split_qkv_f32(const float *qkv, float *q, float *k, float *v,
                        int b, int t, int hq, int hkv, int d);
