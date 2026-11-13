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

#define PARALLAX_ALIGN_BYTES 64
#define PARALLAX_IS_ALIGNED(p, a) (((uintptr_t)(p) & ((uintptr_t)(a) - 1)) == 0)

#endif /* PARALLAX_SIMD_H */
