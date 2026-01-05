#pragma once
#include <cstddef>

#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

inline void prefetch_vector(const char* vec, size_t vecsize) {
#if defined(USE_AVX)
    constexpr size_t prefetch_stride = 32;  // AVX: 256 bits = 32 bytes
#elif defined(USE_SSE)
    constexpr size_t prefetch_stride = 16;  // SSE: 128 bits = 16 bytes
#else
    constexpr size_t prefetch_stride = 64;  // Default to cache line size
#endif

    size_t max_prefetch_size = (vecsize / prefetch_stride) * prefetch_stride;

    for (size_t offset = 0; offset < max_prefetch_size; offset += prefetch_stride) {
#if defined(USE_SSE) || defined(USE_AVX)
        _mm_prefetch(vec + offset, _MM_HINT_T0);
#else
        // Portable fallback for other compilers/architectures
        #if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(vec + offset, 0, 3);
        #endif
#endif
    }
}
