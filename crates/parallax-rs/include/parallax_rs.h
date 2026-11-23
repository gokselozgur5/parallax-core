#ifndef PARALLAX_RS_H
#define PARALLAX_RS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status return codes */
#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER -1
#define PARALLAX_STATUS_INVALID_DIMENSION -2
#define PARALLAX_STATUS_UNALIGNED_POINTER -3
#define PARALLAX_STATUS_ALIASING_POINTER -4
#define PARALLAX_STATUS_NOT_FOUND -5

/* Backend device type identifiers */
#define PARALLAX_DEVICE_TYPE_CPU 0
#define PARALLAX_DEVICE_TYPE_CUDA 1
#define PARALLAX_DEVICE_TYPE_METAL 2
#define PARALLAX_DEVICE_TYPE_NEON 3

/* BPE merge table entry */
typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t rank;
    uint32_t new_id;
} ParallaxMergeEntry;

/* Q8_0 quantized block of 32 elements */
typedef struct {
    float scale;
    int8_t qs[32];
} ParallaxBlockQ8_0;

/* Returns library ABI version */
uint32_t parallax_version(void);

/* GEMM: C = alpha * (A @ B) + beta * C */
int32_t parallax_gemm_f32(
    size_t m, size_t n, size_t k, float alpha,
    const float *a_ptr, size_t a_len,
    const float *b_ptr, size_t b_len,
    float beta,
    float *c_ptr, size_t c_len
);

/* In-place softmax along last dimension */
int32_t parallax_softmax_f32(
    float *data_ptr, size_t data_len, size_t rows, size_t cols
);

/* Binary search for BPE merge rule */
int32_t parallax_bpe_find_merge(
    const ParallaxMergeEntry *table_ptr,
    size_t table_len,
    uint32_t left,
    uint32_t right,
    uint32_t *out_rank,
    uint32_t *out_new_id
);

/* Validate merge table strictly sorted */
int32_t parallax_bpe_is_sorted(
    const ParallaxMergeEntry *table_ptr,
    size_t table_len
);

/* Single-head flash attention f32 */
int32_t parallax_flash_attn_f32(
    const float *q, const float *k, const float *v, float *out,
    size_t seq_len, size_t head_dim, float scale, int32_t is_causal, int32_t f16_corr
);

/* Find bucket boundary for sequence length */
int32_t parallax_find_bucket(
    size_t seq_len,
    const size_t *buckets_ptr,
    size_t buckets_len,
    size_t *out_bucket
);

/* Collate variable length sequences into padded buffer */
int32_t parallax_collate_batch_u32(
    const uint32_t *const *seq_ptrs,
    const size_t *seq_lens,
    size_t num_seqs,
    const size_t *buckets_ptr,
    size_t buckets_len,
    uint32_t pad_token,
    uint32_t *out_tokens,
    uint32_t *out_mask,
    size_t out_len,
    size_t *out_bucket_len
);

/* Number of registered compute backends */
uint32_t parallax_backend_count(void);

/* Get device type for backend index */
int32_t parallax_get_device_type(
    uint32_t backend_idx,
    uint32_t *out_device_type
);

/* Quantize 32 float32 values into BlockQ8_0 */
int32_t parallax_quantize_q8_0(
    const float *input_f32,
    ParallaxBlockQ8_0 *out_block
);

/* Dequantize BlockQ8_0 into 32 float32 values */
int32_t parallax_dequantize_q8_0(
    const ParallaxBlockQ8_0 *block,
    float *out_f32
);

#ifdef __cplusplus
}
#endif

#endif /* PARALLAX_RS_H */
