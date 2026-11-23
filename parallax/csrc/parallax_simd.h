#ifndef PARALLAX_SIMD_H
#define PARALLAX_SIMD_H

#include <stddef.h>
#include <stdint.h>

#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER (-1)
#define PARALLAX_STATUS_INVALID_DIMENSION (-2)
#define PARALLAX_STATUS_UNALIGNED_POINTER (-3)
#define PARALLAX_STATUS_ALIASING_POINTER (-4)
#define PARALLAX_STATUS_UNSUPPORTED_ISA (-5)

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#define PARALLAX_HAVE_NEON 1
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define PARALLAX_ARCH_X86 1
#if defined(__AVX2__)
#define PARALLAX_HAVE_AVX2 1
#endif
#if defined(__AVX512F__)
#define PARALLAX_HAVE_AVX512F 1
#endif
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define PARALLAX_SCALAR_TARGET __attribute__((noinline, target("no-avx,no-fma")))
#define PARALLAX_HAS_AVX2_ISA() ( \
    __builtin_cpu_supports("avx2") && \
    __builtin_cpu_supports("fma"))
#define PARALLAX_HAS_AVX512_ISA() ( \
    __builtin_cpu_supports("avx512f") && \
    __builtin_cpu_supports("avx512dq") && \
    __builtin_cpu_supports("avx512vl") && \
    __builtin_cpu_supports("avx512bw"))
#elif defined(__GNUC__) || defined(__clang__)
#define PARALLAX_SCALAR_TARGET __attribute__((noinline))
#define PARALLAX_HAS_AVX2_ISA() 0
#define PARALLAX_HAS_AVX512_ISA() 0
#else
#define PARALLAX_SCALAR_TARGET
#define PARALLAX_HAS_AVX2_ISA() 0
#define PARALLAX_HAS_AVX512_ISA() 0
#endif

#define PARALLAX_ALIGN_BYTES 64
#define PARALLAX_IS_ALIGNED(p, a) (((uintptr_t)(p) & ((uintptr_t)(a) - 1)) == 0)

int c_has_avx512_f32(void);
int c_gemm_avx512_f32(const float *A, const float *B, float *C, int M, int N, int K);
int c_has_avx2_gemm(void);
int c_gemm_avx2_f32(const float *A, const float *B, float *C, int M, int N, int K);
int c_gemm_neon_f32(const float *A, const float *B, float *C, int M, int N, int K);
void c_gemm_scalar_f32(const float *A, const float *B, float *C, int M, int N, int K);

int c_has_avx512_softmax_f32(void);
int c_softmax_avx512_f32(const float *in, float *out, int rows, int cols);
int c_has_avx2_softmax(void);
int c_softmax_avx2_f32(const float *input, float *output, int rows, int cols);
int c_softmax_neon_f32(const float *in, float *out, int rows, int cols);
void c_softmax_scalar_f32(const float *input, float *output, int rows, int cols);

int c_has_avx512_rmsnorm_f32(void);
int c_rmsnorm_avx512_f32(const float *x, const float *w, float *out, int rows, int cols, float eps);
int c_has_avx2_rmsnorm(void);
int c_rmsnorm_avx2_f32(const float *x, const float *w, float *o, int r, int c, float eps);
int c_has_neon_rmsnorm(void);
void c_rmsnorm_neon_f32(const float *x, const float *w, float *o, int r, int c, float eps);
void c_rmsnorm_scalar_f32(const float *x, const float *w, float *out, int rows, int cols, float eps);

#endif /* PARALLAX_SIMD_H */
