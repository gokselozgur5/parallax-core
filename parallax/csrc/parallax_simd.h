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

#define PARALLAX_ALIGN_BYTES 64
#define PARALLAX_IS_ALIGNED(p, a) (((uintptr_t)(p) & ((uintptr_t)(a) - 1)) == 0)

#endif /* PARALLAX_SIMD_H */
