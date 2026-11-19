// Lossless ggml split-half -> adjacent-nibble MXFP4 layout conversion.
// Copyright 2026 Strata adaptation contributors. SPDX-License-Identifier: Apache-2.0
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

// Returns zero on success, otherwise the one-based block with NaN scale255.
// Buffers are disjoint: source[17*n], packed[16*n], scales[n]. No FP arithmetic.
extern "C" std::size_t repack_avx2(const std::uint8_t* source,
    std::uint8_t* packed, std::uint8_t* scales, std::size_t n) {
    const __m256i mask=_mm256_set1_epi8(15);
    const __m256i coefficients=_mm256_set1_epi16(0x1001);
    std::size_t i=0;
    for (; i+1<n; i+=2) {
        const auto* p=source+17*i;
        if (p[0]==255) return i+1;
        if (p[17]==255) return i+2;
        const __m128i first=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+1));
        const __m128i second=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+18));
        const __m256i bytes=_mm256_set_m128i(second,first);
        const __m256i low=_mm256_and_si256(bytes,mask);
        const __m256i high=_mm256_and_si256(_mm256_srli_epi16(bytes,4),mask);
        // Each pair: even_nibble*1 + odd_nibble*16 <=255. Signed coefficients
        // are positive; neither multiply-add nor pack can saturate.
        const __m256i lo_pairs=_mm256_maddubs_epi16(low,coefficients);
        const __m256i hi_pairs=_mm256_maddubs_epi16(high,coefficients);
        const __m256i result=_mm256_packus_epi16(lo_pairs,hi_pairs);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(packed+16*i),result);
        scales[i]=p[0];scales[i+1]=p[17];
    }
    if (i<n) {
        const auto* p=source+17*i;
        if (p[0]==255) return i+1;
        for (std::size_t j=0;j<8;++j) {
            const auto a=p[1+2*j],b=p[2+2*j];
            packed[16*i+j]=(a&15)|((b&15)<<4);
            packed[16*i+8+j]=(a>>4)|(b&240);
        }
        scales[i]=p[0];
    }
    return 0;
}
