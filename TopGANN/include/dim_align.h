// dim_align_xplat.hpp
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

#ifndef ANN_ALIGN_BYTES
#define ANN_ALIGN_BYTES 64
#endif

// --- SIMD block size for your float distance loop ---
// x86: your AVX *and* SSE versions consume 16 floats per iteration.
// ARM NEON: use 4 as a sensible lane multiple; others fall back to 1.
constexpr std::size_t ann_simd_block_f32() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    return 16; // matches qty >> 4 in your L2SqrSIMD16Ext paths
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return 4;
#else
    return 1;
#endif
}

constexpr std::size_t round_up(std::size_t n, std::size_t k) {
    return k ? ((n + k - 1) / k) * k : n;
}

// Choose padded dim so dim % 16 == 0 on x86 (or %4 on NEON).
inline std::size_t choose_padded_dim_f32(std::size_t dim) {
    return round_up(dim, ann_simd_block_f32());
}

// Optional: also make each row a cache-line multiple.
inline std::size_t choose_padded_dim_f32_row_aligned(std::size_t dim,
                                                     std::size_t row_align_bytes = ANN_ALIGN_BYTES) {
    const std::size_t row_align_elems = row_align_bytes / sizeof(float);
    return round_up(choose_padded_dim_f32(dim), row_align_elems);
}

// ---- aligned alloc/free ----
inline void* ann_aligned_malloc(std::size_t bytes, std::size_t align = ANN_ALIGN_BYTES) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, align);
#elif (__cplusplus >= 201703L)
    std::size_t size = ((bytes + align - 1) / align) * align;
    return std::aligned_alloc(align, size);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
#endif
}
inline void ann_aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

// ---- load + pad to [N x D_pad]; returns D_pad ----
inline std::size_t align_and_load_f32(const char* path,
                                      float** out,
                                      std::size_t orig_dim,
                                      std::size_t num_vecs,
                                      std::size_t offset_vecs = 0,
                                      bool row_align_to_cacheline = true)
{
    if (!path) throw std::invalid_argument("path null");
    *out = nullptr;

    const std::size_t D_pad = row_align_to_cacheline
        ? choose_padded_dim_f32_row_aligned(orig_dim)
        : choose_padded_dim_f32(orig_dim);

    // overflow-safe sizing
    if (num_vecs && D_pad > (SIZE_MAX / num_vecs)) throw std::bad_alloc();
    const std::size_t elems = num_vecs * D_pad;
    if (elems > (SIZE_MAX / sizeof(float))) throw std::bad_alloc();

    float* dst = static_cast<float*>(ann_aligned_malloc(elems * sizeof(float)));
    if (!dst) throw std::bad_alloc();
    std::memset(dst, 0, elems * sizeof(float));

    FILE* f = std::fopen(path, "rb");
    if (!f) { ann_aligned_free(dst); throw std::runtime_error("open failed"); }

    // 64-bit seek to (offset_vecs * orig_dim) floats
    const unsigned long long eloff =
        static_cast<unsigned long long>(offset_vecs) *
        static_cast<unsigned long long>(orig_dim);
#if defined(_WIN32)
    if (_fseeki64(f, static_cast<__int64>(eloff * sizeof(float)), SEEK_SET) != 0) {
        std::fclose(f); ann_aligned_free(dst); throw std::runtime_error("seek failed");
    }
#else
    if (fseeko(f, static_cast<off_t>(static_cast<long long>(eloff * sizeof(float))), SEEK_SET) != 0) {
        std::fclose(f); ann_aligned_free(dst); throw std::runtime_error("seek failed");
    }
#endif

    // read each row; tail stays zero
    for (std::size_t i = 0; i < num_vecs; ++i) {
        float* row = dst + i * D_pad;
        const std::size_t n = std::fread(row, sizeof(float), orig_dim, f);
        if (n != orig_dim) { std::fclose(f); ann_aligned_free(dst); throw std::runtime_error("short read"); }
    }

    std::fclose(f);
    *out = dst;
    return D_pad;
}