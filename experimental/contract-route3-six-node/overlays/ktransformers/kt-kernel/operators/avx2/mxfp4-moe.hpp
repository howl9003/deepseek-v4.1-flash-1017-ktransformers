/**
 * @Description  : AVX2 MXFP4 MoE operator for DeepSeek V4 native inference
 * SPDX-License-Identifier: Apache-2.0
 *
 * MXFP4 stores FP4 E2M1 weights nibble-packed (2 values per byte), plus
 * per-group-32 FP32 scales. This AVX2 backend dequantizes FP4→FP32 via SSSE3
 * PSHUFB lookup tables, then accumulates with BF16 activations using
 * _mm256_fmadd_ps. No AVX-512 required.
 *
 * FP4 E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
 **/
#ifndef CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H
#define CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H

#include <immintrin.h>
#include <tmmintrin.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <cmath>
#include <cstring>
#include <mutex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../cpu_backend/kt_knobs.h"  // kt-kernel-nvfp4-cpu-a4.py
#include <cstdlib>

// KT_MXFP4_SCALE_CODES=1: the FP4 decode reads a one-byte scale index per
// group plus a per-matrix 256-entry FP32 table (patches/kt-mxfp4-scale-codes.py).
// Read at every load, so one process can load both ways.
static inline bool kt_mxfp4_scale_codes() {
  const char* e = std::getenv("KT_MXFP4_SCALE_CODES");
  return e && e[0] == '1';
}
static inline bool kt_scale_codes_verify() {
  static const bool v = [] { const char* e = std::getenv("KT_MXFP4_SCALE_CODES_VERIFY"); return e && e[0] == '1'; }();
  return v;
}
static std::atomic<size_t> kt_scale_codes_checked{0};
static std::atomic<size_t> kt_scale_codes_bad_mats{0};

static inline bool kt_scale_codes_load_all() {
  const char* e = std::getenv("KT_LOAD_ALL_EXPERTS");
  return e && e[0] == '1';
}


#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "moe_base.hpp"


// kt-kernel-nvfp4-cpu-a4.py: flashinfer's NVFP4 activation quantize, on the CPU
namespace kt_cpu_a4 {
inline float e4m3_to_f32(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  return e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
}
// x >= 0; round to nearest even, saturating at 448
inline uint8_t f32_to_e4m3_rn(float x) {
  if (!(x > 0.f)) return 0;
  if (x >= 448.f) return 0x7E;
  if (x < 0.015625f) {  // subnormal: steps of 2^-9
    const int q = (int)std::nearbyint(x * 512.f);
    return (uint8_t)q;  // q == 8 is 2^-6, code 0x08, the smallest normal
  }
  int e;
  const float fr = std::frexp(x, &e);  // x = fr * 2^e, fr in [0.5, 1)
  int m = (int)std::nearbyint((fr * 2.f - 1.f) * 8.f);
  int ex = e - 1;
  if (m == 8) m = 0, ex++;
  const int field = ex + 7;
  if (field > 15 || (field == 15 && m == 7)) return 0x7E;
  return (uint8_t)((field << 3) | m);
}
// nearest e2m1 value, ties to the even code, saturating at 6
inline float e2m1_rn(float v) {
  const float a = std::fabs(v);
  float q;
  if (a <= 0.25f) q = 0.f;
  else if (a < 0.75f) q = 0.5f;
  else if (a <= 1.25f) q = 1.f;
  else if (a < 1.75f) q = 1.5f;
  else if (a <= 2.5f) q = 2.f;
  else if (a < 3.5f) q = 3.f;
  else if (a <= 5.f) q = 4.f;
  else q = 6.f;
  return v < 0.f ? -q : q;
}
// kt-kernel-nvfp4-a4-int8.py: the same quantize on AVX2 (the scalar loop above cost
// Flash-Next decode 67.6 -> 50 tok/s): e4m3 by bits, e2m1 as a sum of seven threshold
// steps (> for ties down to the even code, >= for ties up), bit-identical to it.
// qa / qs (may be null) get the codes doubled as int8 in the int8 GEMM's lane order
// (per 64 values: even elements, then odd) and sf * g / 4.
inline uint8_t f32_to_e4m3_bits(float x) {  // x >= 0, nearest even, saturating at 448
  if (!(x > 0.f)) return 0;
  if (x >= 448.f) return 0x7E;
  uint32_t u;
  std::memcpy(&u, &x, 4);
  int e = (int)(u >> 23) - 127;
  if (e < -6) return (uint8_t)(int)std::nearbyint(x * 512.f);  // subnormal: steps of 2^-9 (8 is 2^-6)
  uint32_t m = u & 0x7FFFFFu;
  m += 0x7FFFFu + ((m >> 20) & 1u);  // round to 3 mantissa bits, ties to even
  if (m >> 23) m = 0, e++;
  m = (m >> 20) & 7u;
  const int field = e + 7;
  if (field > 15 || (field == 15 && m == 7)) return 0x7E;
  return (uint8_t)((field << 3) | (int)m);
}
inline const float* e4m3_table() {
  static const float* t = [] {
    float* v = new float[128];
    for (int b = 0; b < 128; b++) v[b] = e4m3_to_f32((uint8_t)b);
    return v;
  }();
  return t;
}
inline void fake_quant_rows(ggml_bf16_t* rows, int m0, int m1, size_t k, float g, int8_t* qa = nullptr,
                            float* qs = nullptr) {
  const float a1 = 1.f / g;  // flashinfer's input_scale_quant
  const float* t8 = e4m3_table();
  const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  const __m256 th[7] = {_mm256_set1_ps(0.25f), _mm256_set1_ps(0.75f), _mm256_set1_ps(1.25f), _mm256_set1_ps(1.75f),
                        _mm256_set1_ps(2.5f),  _mm256_set1_ps(3.5f),  _mm256_set1_ps(5.f)};
  const __m256 st[7] = {_mm256_set1_ps(0.5f), _mm256_set1_ps(0.5f), _mm256_set1_ps(0.5f), _mm256_set1_ps(0.5f),
                        _mm256_set1_ps(1.f),  _mm256_set1_ps(1.f),  _mm256_set1_ps(2.f)};
  for (int mi = m0; mi < m1; mi++) {
    ggml_bf16_t* r = rows + (size_t)mi * k;
    int8_t* qr = qa ? qa + (size_t)mi * k : nullptr;
    float* sr = qs ? qs + (size_t)mi * (k / 16) : nullptr;
    for (size_t b = 0; b + 16 <= k; b += 16) {
      __m256 x[2];
      for (int h = 0; h < 2; h++)
        x[h] = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(r + b + 8 * h))), 16));
      __m256 mx = _mm256_max_ps(_mm256_and_ps(x[0], absm), _mm256_and_ps(x[1], absm));
      __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
      m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
      m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
      const float amax = _mm_cvtss_f32(m4);
      const float sf = t8[f32_to_e4m3_bits(amax / 6.f * a1)];
      const float outs = sf != 0.f ? a1 / sf : 0.f;
      const __m256 vo = _mm256_set1_ps(outs), vd = _mm256_set1_ps(sf * g);
      alignas(32) float qv[16];
      for (int h = 0; h < 2; h++) {
        const __m256 v = _mm256_mul_ps(x[h], vo);
        const __m256 a = _mm256_and_ps(v, absm);
        __m256 q = _mm256_and_ps(_mm256_cmp_ps(a, th[0], _CMP_GT_OQ), st[0]);
        q = _mm256_add_ps(q, _mm256_and_ps(_mm256_cmp_ps(a, th[1], _CMP_GE_OQ), st[1]));
        q = _mm256_add_ps(q, _mm256_and_ps(_mm256_cmp_ps(a, th[2], _CMP_GT_OQ), st[2]));
        q = _mm256_add_ps(q, _mm256_and_ps(_mm256_cmp_ps(a, th[3], _CMP_GE_OQ), st[3]));
        q = _mm256_add_ps(q, _mm256_and_ps(_mm256_cmp_ps(a, th[4], _CMP_GT_OQ), st[4]));
        q = _mm256_add_ps(q, _mm256_and_ps(_mm256_cmp_ps(a, th[5], _CMP_GE_OQ), st[5]));
        q = _mm256_add_ps(q, _mm256_and_ps(_mm256_cmp_ps(a, th[6], _CMP_GT_OQ), st[6]));
        q = _mm256_or_ps(q, _mm256_andnot_ps(absm, v));  // the sign back
        _mm256_store_ps(qv + 8 * h, q);
        // dequantized, rounded to bf16 as GGML_FP32_TO_BF16 does (nearest even)
        const __m256i u = _mm256_castps_si256(_mm256_mul_ps(q, vd));
        const __m256i rb = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1)));
        const __m256i hi = _mm256_srli_epi32(_mm256_add_epi32(u, rb), 16);
        const __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(hi), _mm256_extracti128_si256(hi, 1));
        _mm_storeu_si128((__m128i*)(r + b + 8 * h), p16);
      }
      if (qr) {  // the 16 codes x 2 as int8, even elements to one half of the chunk, odd to the other
        const __m256 two = _mm256_set1_ps(2.f);
        const __m256i c0 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_load_ps(qv), two));
        const __m256i c1 = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_load_ps(qv + 8), two));
        const __m128i w0 = _mm_packs_epi32(_mm256_castsi256_si128(c0), _mm256_extracti128_si256(c0, 1));
        const __m128i w1 = _mm_packs_epi32(_mm256_castsi256_si128(c1), _mm256_extracti128_si256(c1, 1));
        const __m128i b16 = _mm_shuffle_epi8(_mm_packs_epi16(w0, w1),
                                             _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
        const size_t c = b / 64, p0 = b % 64;
        _mm_storel_epi64((__m128i*)(qr + c * 64 + p0 / 2), b16);
        _mm_storel_epi64((__m128i*)(qr + c * 64 + 32 + p0 / 2), _mm_srli_si128(b16, 8));
      }
      if (sr) sr[b / 16] = sf * g * 0.25f;
    }
  }
}
// One row through fake_quant_rows, remembered per thread: at decode every expert's
// gate/up input is the same token row, quantized once and copied after (the packing of
// those inputs runs serially below 10 tokens). dst holds the row (from_mat's copy).
struct RowCache {
  int gb = 0;
  size_t k = 0;
  bool has_q = false;
  std::vector<uint16_t> src, deq;
  std::vector<int8_t> qa;
  std::vector<float> qs;
};
inline void quant_row_cached(const ggml_bf16_t* src, ggml_bf16_t* dst, size_t k, float g, int gb, int8_t* qa,
                             float* qs) {
  thread_local RowCache c;
  const bool want_q = qa != nullptr;
  if (c.gb == gb && c.k == k && (c.has_q || !want_q) && std::memcmp(c.src.data(), src, k * 2) == 0) {
    std::memcpy(dst, c.deq.data(), k * 2);
    if (want_q) std::memcpy(qa, c.qa.data(), k), std::memcpy(qs, c.qs.data(), (k / 16) * sizeof(float));
    return;
  }
  c.src.assign((const uint16_t*)src, (const uint16_t*)src + k);
  fake_quant_rows(dst, 0, 1, k, g, qa, qs);
  c.gb = gb, c.k = k, c.has_q = want_q;
  c.deq.assign((const uint16_t*)dst, (const uint16_t*)dst + k);
  if (want_q) c.qa.assign(qa, qa + k), c.qs.assign(qs, qs + k / 16);
}
inline bool int8_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("KT_CPU_A4_INT8");
    return v ? std::atoi(v) != 0 : true;
  }();
  return on;
}
alignas(16) static const int8_t kKV2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
inline std::atomic<int>& int8_checked() {
  static std::atomic<int> c{0};
  return c;
}
}  // namespace kt_cpu_a4

namespace avx2 {

// ============================================================================
// AVX2 MXFP4 GemmKernel
// Weights: FP4 E2M1 nibble-packed, per-group-32 FP32 scales
// Activations: BF16; Output: FP32
// ============================================================================
struct GemmKernelAVX2MXFP4 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}
  static int recommended_nth(int n) { return std::max(1, div_up(n, N_BLOCK)); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }
  static std::string name() { return "AVX2_MXFP4"; }

  // FP4 E2M1 → BF16 lookup tables (low byte and high byte of BF16)
  // Identical to amx/fp4-moe.hpp GemmKernel224MXFP4SmallKGroup LUTs
  alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
                                                          0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
  alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
                                                          0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};

  // Dequantize 4 packed bytes (= 8 FP4 E2M1 nibbles) → 8 FP32 values.
  //
  // Nibble order: packed[j] = {lo_nib = column 2j, hi_nib = column 2j+1}
  // Output order: columns 0, 1, 2, ..., 7 (sequential).
  //
  // Algorithm: SSSE3 PSHUFB table lookup → 8 BF16 values → BF16-to-FP32
  // (zero-extend uint16 to uint32, shift left 16).
  __attribute__((always_inline)) static inline __m256 fp4x8_to_fp32(const uint8_t* packed, __m128i lut_lo,
                                                                    __m128i lut_hi) {
    int32_t raw;
    std::memcpy(&raw, packed, 4);
    __m128i data = _mm_cvtsi32_si128(raw);
    __m128i lo_mask = _mm_set1_epi8(0x0F);

    // Extract lo nibbles (bits 0-3) and hi nibbles (bits 4-7)
    __m128i lo_nib = _mm_and_si128(data, lo_mask);
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(data, 4), lo_mask);

    // PSHUFB: BF16 byte lookup for each set of nibbles
    __m128i lo_bf16 = _mm_unpacklo_epi8(_mm_shuffle_epi8(lut_lo, lo_nib),
                                        _mm_shuffle_epi8(lut_hi, lo_nib));  // 4 BF16: cols {0,2,4,6}
    __m128i hi_bf16 = _mm_unpacklo_epi8(_mm_shuffle_epi8(lut_lo, hi_nib),
                                        _mm_shuffle_epi8(lut_hi, hi_nib));  // 4 BF16: cols {1,3,5,7}

    // Interleave: {col0,col1,col2,...,col7} in sequential order
    __m128i bf16_8 = _mm_unpacklo_epi16(lo_bf16, hi_bf16);  // 8 × BF16 in 128 bits

    // BF16 → FP32: zero-extend u16 → u32, then shift left 16
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_8), 16));
  }

  // Scalar dequant of a single nibble index (for tail handling)
  static inline float fp4_scalar(uint8_t nib) {
    const uint32_t bf16_bits = (uint32_t)fp4_bf16_lo[nib] | ((uint32_t)fp4_bf16_hi[nib] << 8);
    const uint32_t fp32_bits = bf16_bits << 16;
    float result;
    std::memcpy(&result, &fp32_bits, sizeof(float));
    return result;
  }

  // ---- Buffer types --------------------------------------------------------

  // Decode emission order of the 256-bit PSHUFB group decode in gemm_mxfp4:
  // activations are stored pre-permuted so the inner loops pair weights and
  // activations with plain loads.
  static constexpr int kGroupPerm[32] = {0,  2,  4,  6,  1,  3,  5,  7,  8,  10, 12, 14, 9,  11, 13, 15,
                                         16, 18, 20, 22, 17, 19, 21, 23, 24, 26, 28, 30, 25, 27, 29, 31};

  struct BufferA {
    ggml_bf16_t* data = nullptr;
    // Pre-permuted FP32 copy of the same rows (null when k % 32 != 0). Filled
    // once per expert by from_mat; before 09-13 every gemm_mxfp4 call — one
    // per 64-row weight slice, 36-80 per expert — redid this conversion for
    // the whole m x k block, which was ~40% of a prefill task.
    float* perm = nullptr;
    // int8 copy for KT_MXFP4_G32_INT8 (patches/kt-mxfp4-g32-int8.py): u8 =
    // round(a / da) + 128 per row in the decode's lane order, da per row and
    // per block of 1024 values. Filled by from_mat with perm, shared by the
    // task's threads.
    uint8_t* q8 = nullptr;
    float* da8 = nullptr;
    int q8_m = 0;
    static constexpr int kQ8Block = 1024;
    static bool int8_enabled() {
      static const bool on = [] {
        const char* v = std::getenv("KT_MXFP4_G32_INT8");
        return v ? std::atoi(v) != 0 : false;
      }();
      return on;
    }
    // Rows the pre-permuted copy holds. Filled by from_mat only for prefill-
    // sized inputs: writing and reading it for the 1-8 rows of a decode step
    // cost 13% of Flash-Next decode (09-14, 36.4 -> 31.7 tok/s), while the
    // gemm's own per-thread scratch stays hot in cache at those sizes.
    int perm_m = 0;
    static constexpr int kPermMinRows = 16;
    size_t max_m = 0, k = 0;
    // kt-kernel-nvfp4-cpu-a4.py: the knob holding the float bits of the NVFP4 global
    // scale for these activations (0 = off), set by make_buffer_a_impl
    std::atomic<int>* a4_global_bits = nullptr;
    // kt-kernel-nvfp4-a4-int8.py: the A4 rows as int8 codes x 2 in lane order and the
    // per-group scales, for qa_m rows (0: the int8 GEMM does not apply)
    int8_t* qa = nullptr;
    float* qs = nullptr;
    int qa_m = 0;
    // KT_MOE_FP32_INTERMEDIATES: rows arrive as FP32 (the activation output kept
    // unrounded) and are consumed without a BF16 round trip. Set by from_mat_f32,
    // cleared by from_mat. The pointer must stay valid until the GEMM has run.
    const float* f32 = nullptr;
    int f32_m = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, int /*group_size*/, void* ptr) : max_m(m), k(k_) { set_data(ptr); }

    static size_t required_size(size_t m, size_t k, int /*group_size*/) {
      // bf16 rows + fp32 perm copy + u8 copy + per-block scales + A4 int8 rows + their group scales
      return m * k * (sizeof(ggml_bf16_t) + sizeof(float) + 1) + m * (k / kQ8Block + 1) * sizeof(float) + m * k +
             m * (k / 16) * sizeof(float);
    }
    void set_data(void* ptr) {
      data = (ggml_bf16_t*)ptr;
      perm = (k % 32 == 0) ? (float*)((uint8_t*)ptr + max_m * k * sizeof(ggml_bf16_t)) : nullptr;
      q8 = perm ? (uint8_t*)(perm + max_m * k) : nullptr;
      da8 = q8 ? (float*)(q8 + max_m * k) : nullptr;
      uint8_t* tail = (uint8_t*)ptr + max_m * k * (sizeof(ggml_bf16_t) + sizeof(float) + 1) +
                      max_m * (k / kQ8Block + 1) * sizeof(float);
      qa = (k % 64 == 0) ? (int8_t*)tail : nullptr;
      qs = qa ? (float*)(tail + max_m * k) : nullptr;
    }

    // FP32 rows, never rounded: the pre-permuted copy is filled for prefill-sized
    // inputs exactly as from_mat does, from the FP32 values instead of BF16 ones.
    void from_mat_f32(int m, const float* src, int ith, int nth) {
      int m_start = 0, m_end = m;
      if (!(ith == 0 && nth == 1)) {
        auto r = split_range(m, ith, nth);
        m_start = r.first;
        m_end = r.second;
      }
      f32 = src;
      f32_m = m;
      qa_m = 0;
      q8_m = 0;
      if (perm == nullptr || m < kPermMinRows) {
        perm_m = 0;
        return;
      }
      perm_m = m;
      const int groups = (int)(k / 32);
      for (int mi = m_start; mi < m_end; mi++) {
        const float* a_row = src + (size_t)mi * k;
        float* p_row = perm + (size_t)mi * k;
        for (int g = 0; g < groups; g++) {
          const int base = g * 32;
          float* dst = p_row + base;
          for (int j = 0; j < 32; j++) dst[j] = a_row[base + kGroupPerm[j]];
        }
      }
    }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      int m_start = 0, m_end = m;
      if (!(ith == 0 && nth == 1)) {
        auto r = split_range(m, ith, nth);
        m_start = r.first;
        m_end = r.second;
      }
      f32 = nullptr;
      f32_m = 0;
      std::memcpy(data + (size_t)m_start * k, src + (size_t)m_start * k,
                  (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      qa_m = 0;
      if (a4_global_bits != nullptr) {
        const int gb = a4_global_bits->load(std::memory_order_relaxed);
        if (gb != 0) {
          float g;
          std::memcpy(&g, &gb, sizeof(g));
          const bool i8 = qa != nullptr && kt_cpu_a4::int8_enabled();
          if (m_end - m_start == 1)
            kt_cpu_a4::quant_row_cached(src + (size_t)m_start * k, data + (size_t)m_start * k, k, g, gb,
                                        i8 ? qa + (size_t)m_start * k : nullptr,
                                        i8 ? qs + (size_t)m_start * (k / 16) : nullptr);
          else
            kt_cpu_a4::fake_quant_rows(data, m_start, m_end, k, g, i8 ? qa : nullptr, i8 ? qs : nullptr);
          qa_m = i8 ? m : 0;
        }
      }
      // the copies below read the rows as stored (quantized when the knob is set)
      src = data;
      if (perm == nullptr || m < kPermMinRows) {
        perm_m = 0;
        q8_m = 0;
        return;
      }
      perm_m = m;
      const int groups = (int)(k / 32);
      for (int mi = m_start; mi < m_end; mi++) {
        const ggml_bf16_t* a_row = src + (size_t)mi * k;
        float* p_row = perm + (size_t)mi * k;
        for (int g = 0; g < groups; g++) {
          const int base = g * 32;
          float* dst = p_row + base;
          for (int j = 0; j < 32; j++) dst[j] = GGML_BF16_TO_FP32(a_row[base + kGroupPerm[j]]);
        }
      }
      q8_m = 0;
      if (!int8_enabled() || q8 == nullptr || (k % kQ8Block) != 0) return;
      const int nblk = (int)(k / kQ8Block);
      for (int mi = m_start; mi < m_end; mi++) {
        const ggml_bf16_t* a_row = src + (size_t)mi * k;
        uint8_t* q_row = q8 + (size_t)mi * k;
        float* d_row = da8 + (size_t)mi * nblk;
        for (int blk = 0; blk < nblk; blk++) {
          const ggml_bf16_t* ab = a_row + (size_t)blk * kQ8Block;
          float amax = 0.f;
          for (int j = 0; j < kQ8Block; j++) amax = std::max(amax, std::fabs(GGML_BF16_TO_FP32(ab[j])));
          const float da = amax / 127.f;
          const float inv = da > 0.f ? 1.f / da : 0.f;
          d_row[blk] = da;
          for (int gg = 0; gg < kQ8Block / 32; gg++) {
            const ggml_bf16_t* sg = ab + gg * 32;
            uint8_t* dq = q_row + (size_t)blk * kQ8Block + gg * 32;
            for (int i = 0; i < 16; i++) {
              int q0 = (int)std::lrintf(GGML_BF16_TO_FP32(sg[2 * i]) * inv);
              int q1 = (int)std::lrintf(GGML_BF16_TO_FP32(sg[2 * i + 1]) * inv);
              q0 = q0 > 127 ? 127 : (q0 < -127 ? -127 : q0);
              q1 = q1 > 127 ? 127 : (q1 < -127 ? -127 : q1);
              dq[i] = (uint8_t)(q0 + 128);
              dq[16 + i] = (uint8_t)(q1 + 128);
            }
          }
        }
      }
      q8_m = m;
    }
  };

  struct BufferB {
    uint8_t* b = nullptr;  // nibble-packed FP4 (may be nullptr in scale-only mode)
    float* d = nullptr;    // FP32 group scales
    int n = 0, k = 0, k_group_size = 0, k_group_count = 0;

    BufferB() = default;

    // Full allocation: b and d packed into a single aligned block.
    BufferB(int n_, int k_, int k_group_size_, void* ptr)
        : b((uint8_t*)ptr), n(n_), k(k_), k_group_size(k_group_size_) {
      if (k_group_size <= 0 || k % k_group_size != 0 || k % 8 != 0)
        throw std::runtime_error("MXFP4 AVX2 requires k % group_size == 0 and k % 8 == 0");
      k_group_count = k / k_group_size;
      d = (float*)((uint8_t*)ptr + (size_t)n * k / 2);
    }

    // Scale-only: b is set externally from mmap'd safetensor data.
    BufferB(int n_, int k_, int k_group_size_, void* scale_ptr, std::nullptr_t)
        : b(nullptr), n(n_), k(k_), k_group_size(k_group_size_) {
      if (k_group_size <= 0 || k % k_group_size != 0 || k % 8 != 0)
        throw std::runtime_error("MXFP4 AVX2 requires k % group_size == 0 and k % 8 == 0");
      k_group_count = k / k_group_size;
      d = (float*)scale_ptr;
    }

    static size_t required_size(size_t n, size_t k, int k_group_size) {
      return n * k / 2 + n * (k / k_group_size) * sizeof(float);
    }
    static size_t required_size_scale_only(size_t n, size_t k, int k_group_size) {
      return n * (k / k_group_size) * sizeof(float);
    }

    // One-byte scale codes (KT_MXFP4_SCALE_CODES). lut holds this matrix's
    // distinct FP32 scales bit for bit; code[i] indexes it for group i.
    std::unique_ptr<uint8_t[]> code_store;
    uint8_t* code = nullptr;
    alignas(64) float lut[256];
    mutable std::atomic<int> code_checked{0};

    // KT_MXFP4_SCALE_CODES_VERIFY: once per matrix, at first use.
    void verify_scale_codes() const {
      int z = 0;
      if (!code_checked.compare_exchange_strong(z, 1)) return;
      const size_t cnt = (size_t)n * k_group_count;
      size_t bad = 0, first = 0;
      for (size_t i = 0; i < cnt; i++) {
        uint32_t want, got;
        std::memcpy(&want, d + i, 4);
        std::memcpy(&got, lut + code[i], 4);
        if (want != got) {
          if (bad == 0) first = i;
          bad++;
        }
      }
      const size_t c = ++kt_scale_codes_checked;
      if (bad) {
        const size_t m = ++kt_scale_codes_bad_mats;
        uint32_t want, got;
        std::memcpy(&want, d + first, 4);
        std::memcpy(&got, lut + code[first], 4);
        printf("KT_MXFP4_SCALE_CODES_VERIFY: MISMATCH matrix %p n=%d groups=%d: %zu of %zu groups differ, "
               "first at %zu d=%08x lut=%08x (bad matrices so far %zu)\n",
               (const void*)this, n, k_group_count, bad, cnt, first, want, got, m);
        fflush(stdout);
      }
      if (c % 20000 == 0) {
        printf("KT_MXFP4_SCALE_CODES_VERIFY: %zu matrices checked, %zu with mismatches\n", c,
               kt_scale_codes_bad_mats.load());
        fflush(stdout);
      }
    }

    // Returns false (and leaves the FP32 path in place) if the matrix has more
    // than 256 distinct scale bit patterns.
    bool build_scale_codes() {
      code = nullptr;
      if (d == nullptr || n <= 0 || k_group_count <= 0) return false;
      const size_t cnt = (size_t)n * k_group_count;
      std::unique_ptr<uint8_t[]> out(new uint8_t[cnt]);
      uint32_t keys[512];
      int16_t val[512];
      for (int i = 0; i < 512; i++) val[i] = -1;
      int nd = 0;
      for (size_t i = 0; i < cnt; i++) {
        uint32_t x;
        std::memcpy(&x, d + i, 4);
        uint32_t h = (x * 2654435761u) >> 23;
        while (val[h] >= 0 && keys[h] != x) h = (h + 1) & 511;
        if (val[h] < 0) {
          if (nd == 256) return false;
          keys[h] = x;
          val[h] = (int16_t)nd;
          std::memcpy(&lut[nd], &x, 4);
          nd++;
        }
        out[i] = (uint8_t)val[h];
      }
      for (int i = nd; i < 256; i++) lut[i] = 0.0f;
      code_store = std::move(out);
      code = code_store.get();
      return true;
    }

    void from_raw_mat(const uint8_t* proj, int ith, int nth) {
      if (b == nullptr) return;
      auto [n_start, n_end] = split_range(n, ith, nth);
      const size_t row_bytes = (size_t)k / 2;
      std::memcpy(b + n_start * row_bytes, proj + n_start * row_bytes, (size_t)(n_end - n_start) * row_bytes);
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0, n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : data((float*)ptr), max_m(m), n(n_) {}

    static size_t required_size(size_t m, size_t n) { return m * n * sizeof(float); }
    void set_data(void* ptr) { data = (float*)ptr; }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + (size_t)mi * n;
        ggml_bf16_t* dst_row = dst + (size_t)mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) store_fp32_to_bf16(dst_row + j, _mm256_loadu_ps(src_row + j));
        for (; j < n_end; j++) dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
      }
    }
  };
};

// ============================================================================
// Compute: M tokens × N output neurons, K inner dimension, group_size scale
//
// N-outer loop: each weight row loaded once and shared across M tokens.
// 4-token M-blocking for prefill: weight decode amortized over 4 activations.
// ============================================================================
static inline int32_t hsum_i32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
  lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
  return _mm_cvtsi128_si32(lo);
}

// kt-mxfp4-decode-prefetch.py: KT_MXFP4_DEC_VCONV=1, the permuted fp32 copy with PSHUFB
static inline bool kt_mxfp4_dec_vconv() {
  static const bool on = [] {
    const char* v = std::getenv("KT_MXFP4_DEC_VCONV");
    return v && std::atoi(v) != 0;
  }();
  return on;
}

static void gemm_mxfp4(int m, int n, int k, GemmKernelAVX2MXFP4::BufferA& a, GemmKernelAVX2MXFP4::BufferB& b,
                       GemmKernelAVX2MXFP4::BufferC& c, int ith, int nth) {
  // H3: Check weight buffer is not null
  if (b.b == nullptr) {
    throw std::runtime_error("gemm_mxfp4: weight buffer (b.b) is null");
  }
  auto [n_start, n_end] = split_range(n, ith, nth);
  const int group_count = b.k_group_count;
  const int group_size = b.k_group_size;
  const size_t row_bytes = (size_t)k / 2;
  if (a.f32 != nullptr && (group_size != 32 || (k % 32) != 0 ||
      (a.perm == nullptr && (size_t)m * (size_t)k > (size_t)(4 << 20))))
    throw std::runtime_error("FP32 intermediates require the group-32 permutation path");

  const __m128i lut_lo = _mm_load_si128((const __m128i*)GemmKernelAVX2MXFP4::fp4_bf16_lo);
  const __m128i lut_hi = _mm_load_si128((const __m128i*)GemmKernelAVX2MXFP4::fp4_bf16_hi);

  // kt-kernel-nvfp4-a4-int8.py: A4 activations x NVFP4 weights on int8
  if (group_size == 16 && a.qa != nullptr && a.qa_m >= m && (k % 64) == 0) {
    const int chunks = k / 64;
    const __m256i kv = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*)kt_cpu_a4::kKV2));
    const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    const __m256i dup = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    constexpr int R = 4;  // output columns a pass: the activation chunk is loaded once for four
    for (int n0 = n_start; n0 < n_end; n0 += R) {
      const int nr = std::min(R, n_end - n0);
      if (n0 + R < n_end) {
        const char* nx = (const char*)(b.b + (size_t)(n0 + R) * row_bytes);
        for (int r = 0; r < R && n0 + R + r < n_end; r++) _mm_prefetch(nx + (size_t)r * row_bytes, _MM_HINT_T0);
      }
      for (int mi = 0; mi < m; mi++) {
        const int8_t* qa = a.qa + (size_t)mi * k;
        const float* qs = a.qs + (size_t)mi * group_count;
        __m256 tot[R];
        for (int r = 0; r < R; r++) tot[r] = _mm256_setzero_ps();
        for (int ch = 0; ch < chunks; ch++) {
          const __m256i ae = _mm256_loadu_si256((const __m256i*)(qa + ch * 64));
          const __m256i ao = _mm256_loadu_si256((const __m256i*)(qa + ch * 64 + 32));
          const __m256 sa = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(_mm_loadu_ps(qs + ch * 4)), dup);
#pragma GCC unroll 4
          for (int r = 0; r < R; r++) {
            if (r >= nr) break;
            const int ni = n0 + r;
            const __m256i raw = _mm256_loadu_si256((const __m256i*)(b.b + (size_t)ni * row_bytes + ch * 32));
            const __m256i we = _mm256_shuffle_epi8(kv, _mm256_and_si256(raw, m4));
            const __m256i wo = _mm256_shuffle_epi8(kv, _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4));
            // vpmaddubsw is u8 x s8: |w| unsigned, the activation carrying w's sign
            const __m256i pe = _mm256_maddubs_epi16(_mm256_sign_epi8(we, we), _mm256_sign_epi8(ae, we));
            const __m256i po = _mm256_maddubs_epi16(_mm256_sign_epi8(wo, wo), _mm256_sign_epi8(ao, wo));
            const __m256i i32 = _mm256_madd_epi16(_mm256_add_epi16(pe, po), ones);  // lanes: groups 0,0,1,1,2,2,3,3
            __m128 ws;
            if (b.code) {
              const uint8_t* cd = b.code + (size_t)ni * group_count + ch * 4;
              ws = _mm_setr_ps(b.lut[cd[0]], b.lut[cd[1]], b.lut[cd[2]], b.lut[cd[3]]);
            } else {
              ws = _mm_loadu_ps(b.d + (size_t)ni * group_count + ch * 4);
            }
            const __m256 sc = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_castps128_ps256(ws), dup), sa);
            tot[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i32), sc, tot[r]);
          }
        }
        for (int r = 0; r < nr; r++) c.data[(size_t)mi * n + n0 + r] = hsum_avx2(tot[r]);
      }
    }
    // once a process: one element against a double sum over the stored weights and the
    // quantized activation row (a.data holds it in bf16, so ~1e-3 apart, not 1)
    int z = 0;
    if (ith == 0 && n_end > n_start && kt_cpu_a4::int8_checked().compare_exchange_strong(z, 1)) {
      double ref = 0;
      const int ni = n_start;
      for (int e = 0; e < k; e++) {
        const uint8_t byte = b.b[(size_t)ni * row_bytes + e / 2];
        const float wsc = b.code ? b.lut[b.code[(size_t)ni * group_count + e / 16]] : b.d[(size_t)ni * group_count + e / 16];
        ref += (double)kt_cpu_a4::kE2M1[(e & 1) ? byte >> 4 : byte & 15] * wsc * GGML_BF16_TO_FP32(a.data[e]);
      }
      const double got = c.data[ni];
      printf("[kt-cpu-a4] int8 GEMM check (m %d, n %d, k %d): %.6g against %.6g, relative %.2e\n", m, n, k, got, ref,
             std::fabs(got - ref) / std::max(1e-30, std::fabs(ref)));
      fflush(stdout);
    }
    return;
  }

  // --------------------------------------------------------------------------
  // Fast path (decode AND prefill): one whole 32-value k-group is decoded per
  // iteration with 256-bit PSHUFB.  The decode naturally emits values in a
  // fixed in-lane permutation; rather than reordering the weights (extra
  // shuffle uops on Zen2's two shuffle pipes, which is exactly what limits
  // the generic path), every activation row is converted to FP32 once in that
  // same permuted order, so the inner loops pair weights and activations with
  // plain loads and FMAs.  unpack(zero, u16) fuses the BF16→FP32 widen and
  // <<16 into a single shuffle.  Tokens are processed in blocks of 4 so each
  // decoded group is amortized over 4 accumulators (16 ymm live: 4 weights +
  // 4 group accumulators + 4 totals + LUTs); the remainder uses the
  // single-row loop.  Very large per-expert batches fall back to the generic
  // path to bound the per-thread FP32 staging buffer.
  // --------------------------------------------------------------------------
  if (group_size == 32 && (k % 32) == 0 && (a.perm != nullptr || (size_t)m * (size_t)k <= (size_t)(4 << 20))) {
    // Decode emission order within each 32-value group (see w0..w3 below).
    const int* kPerm = GemmKernelAVX2MXFP4::kGroupPerm;
    static thread_local std::vector<float> a_perm_storage;
    const float* a_perm = (a.perm != nullptr && a.perm_m >= m) ? a.perm : nullptr;
    if (a_perm == nullptr) {
      if (a_perm_storage.size() < (size_t)m * k) a_perm_storage.resize((size_t)m * k);
      float* tmp = a_perm_storage.data();
      if (a.f32 != nullptr && a.f32_m >= m) {
        // KT_MOE_FP32_INTERMEDIATES: permute the FP32 rows as given, no BF16 step.
        for (int mi = 0; mi < m; mi++) {
          const float* a_row = a.f32 + (size_t)mi * a.k;
          float* p_row = tmp + (size_t)mi * k;
          for (int g = 0; g < group_count; g++) {
            const int base = g * 32;
            float* dst = p_row + base;
            for (int j = 0; j < 32; j++) dst[j] = a_row[base + kPerm[j]];
          }
        }
      } else if (kt_mxfp4_dec_vconv()) {
        const __m128i even_odd = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        for (int mi = 0; mi < m; mi++) {
          const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
          float* p_row = tmp + (size_t)mi * k;
          for (int e = 0; e < k; e += 8) {
            const __m128i v = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(a_row + e)), even_odd);
            _mm256_storeu_si256((__m256i*)(p_row + e), _mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
          }
        }
      } else
      for (int mi = 0; mi < m; mi++) {
        const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
        float* p_row = tmp + (size_t)mi * k;
        for (int g = 0; g < group_count; g++) {
          const int base = g * 32;
          float* dst = p_row + base;
          for (int j = 0; j < 32; j++) dst[j] = GGML_BF16_TO_FP32(a_row[base + kPerm[j]]);
        }
      }
      a_perm = tmp;
    }

    const __m256i lut_lo256 = _mm256_broadcastsi128_si256(lut_lo);
    const __m256i lut_hi256 = _mm256_broadcastsi128_si256(lut_hi);
    const __m256i zero256 = _mm256_setzero_si256();
    const __m128i nib_mask = _mm_set1_epi8(0x0F);
    const __m256i nib_mask256 = _mm256_set1_epi8(0x0F);
    const __m256i lane_shift = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);

// Decode one 32-value k-group at b_row+g*16 into w0..w3:
// w0: cols {0,2,4,6 | 1,3,5,7};  w1: {8,10,12,14 | 9,11,13,15}
// w2: {16,...,22 | 17,...,23};   w3: {24,...,30 | 25,...,31}
#define KT_MXFP4_DECODE_GROUP(b_row, g)                                            \
  const __m256i r2 = _mm256_broadcastsi128_si256(                                 \
      _mm_loadu_si128((const __m128i*)((b_row) + (size_t)(g)*16)));                 \
  const __m256i v = _mm256_and_si256(_mm256_srlv_epi32(r2, lane_shift), nib_mask256); \
  const __m256i bl = _mm256_shuffle_epi8(lut_lo256, v);                            \
  const __m256i bh = _mm256_shuffle_epi8(lut_hi256, v);                            \
  const __m256i u16a = _mm256_unpacklo_epi8(bl, bh);                               \
  const __m256i u16b = _mm256_unpackhi_epi8(bl, bh);                               \
  const __m256 w0 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16a));     \
  const __m256 w1 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16a));     \
  const __m256 w2 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16b));     \
  const __m256 w3 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16b))

    // Rows outer, output columns inner (group-32). The four activation rows
    // (64 KiB at k = 4096) stay in L1 across every output column of this
    // thread instead of being re-streamed from L2 once per column; the
    // weights stream instead. Each output keeps its own untouched k loop and
    // is written exactly once, so the result is bit-identical.
#define KT_MXFP4_PREFETCH_NEXT_ROW(ni)                                         \
  if ((ni) + 1 < n_end) {                                                      \
    const char* nr = (const char*)(b.b + (size_t)((ni) + 1) * row_bytes);      \
    _mm_prefetch(nr, _MM_HINT_T0);                                             \
    _mm_prefetch(nr + 64, _MM_HINT_T0);                                        \
    _mm_prefetch(nr + 128, _MM_HINT_T0);                                       \
    _mm_prefetch(nr + 192, _MM_HINT_T0);                                       \
  }
    // Row decode per token tile (group-32). With a tile of TILE >= 8 rows,
    // each output column's weight row is decoded once into an L1 buffer and
    // read back by every 4-token block of the tile; the products, per-group
    // accumulation and scale fold per output are the same as below, so the
    // result is bit-identical.
    static const int kTile = [] {
      const char* v = std::getenv("KT_MXFP4_G32_TILE");
      return v ? std::max(0, std::atoi(v)) & ~3 : 0;
    }();
    // Verify-sized batches (group-32, 2 <= m <= 8, KT_MXFP4_SMALL_M, on): a speculative
    // verify step puts a few tokens on the same experts. Each 32-value group is decoded
    // once for up to four rows held in registers (rows 4-7 take a second pass, their
    // weights from L2), k-blocked so the rows' activations stay in L1. Every row keeps
    // the single-row path's arithmetic (one product chain a group, the scale folded
    // into two sums by group parity, groups in order), so each row's output is
    // bit-identical to computing it alone. Before: 2 rows 1.58x, 3 rows 2.17x a row.
    static const bool kSmallM = [] {
      const char* v = std::getenv("KT_MXFP4_SMALL_M");
      return v ? std::atoi(v) != 0 : true;
    }();
    if (kSmallM && m >= 2 && m <= 8 && n_end > n_start) {
      const int ncols = n_end - n_start;
      constexpr int kSmBlk = 32;  // groups a k block: four rows' activations are 16 KiB
      static thread_local std::vector<float> sm_park_storage;
      if (sm_park_storage.size() < (size_t)ncols * 64) sm_park_storage.resize((size_t)ncols * 64);
      float* park = sm_park_storage.data();
      const uint8_t* codes = b.code;
      if (codes != nullptr && kt_scale_codes_verify()) b.verify_scale_codes();
      auto rows_pass = [&](auto nr_const, int r0) {
        constexpr int NR = decltype(nr_const)::value;
        std::fill(park, park + (size_t)ncols * 64, 0.f);
        for (int gb = 0; gb < group_count; gb += kSmBlk) {
          const int ge = std::min(group_count, gb + kSmBlk);
          for (int ni = n_start; ni < n_end; ni++) {
            const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
            if (ni + 1 < n_end) {
              const char* nx = (const char*)(b_row + row_bytes + (size_t)gb * 16);
              for (int off = 0; off < (ge - gb) * 16; off += 64) _mm_prefetch(nx + off, _MM_HINT_T0);
            }
            float* pk = park + (size_t)(ni - n_start) * 64;
            __m256 ta[NR], tb[NR];
#pragma GCC unroll 4
            for (int r = 0; r < NR; r++) {
              ta[r] = _mm256_loadu_ps(pk + 16 * r);
              tb[r] = _mm256_loadu_ps(pk + 16 * r + 8);
            }
            for (int g = gb; g < ge; g++) {
              KT_MXFP4_DECODE_GROUP(b_row, g);
              const __m256 sv = _mm256_broadcast_ss(codes ? &b.lut[codes[(size_t)ni * group_count + g]]
                                                          : &b.d[(size_t)ni * group_count + g]);
#pragma GCC unroll 4
              for (int r = 0; r < NR; r++) {
                const float* ap = a_perm + (size_t)(r0 + r) * k + (size_t)g * 32;
                __m256 gacc = _mm256_mul_ps(_mm256_loadu_ps(ap), w0);
                gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), w1, gacc);
                gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 16), w2, gacc);
                gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), w3, gacc);
                if (g & 1)
                  tb[r] = _mm256_fmadd_ps(gacc, sv, tb[r]);
                else
                  ta[r] = _mm256_fmadd_ps(gacc, sv, ta[r]);
              }
            }
#pragma GCC unroll 4
            for (int r = 0; r < NR; r++) {
              _mm256_storeu_ps(pk + 16 * r, ta[r]);
              _mm256_storeu_ps(pk + 16 * r + 8, tb[r]);
            }
          }
        }
        for (int ni = n_start; ni < n_end; ni++) {
          const float* pk = park + (size_t)(ni - n_start) * 64;
          for (int r = 0; r < NR; r++)
            c.data[(size_t)(r0 + r) * n + ni] =
                hsum_avx2(_mm256_add_ps(_mm256_loadu_ps(pk + 16 * r), _mm256_loadu_ps(pk + 16 * r + 8)));
        }
      };
      for (int r0 = 0; r0 < m; r0 += 4) {
        switch (std::min(4, m - r0)) {
          case 1: rows_pass(std::integral_constant<int, 1>{}, r0); break;
          case 2: rows_pass(std::integral_constant<int, 2>{}, r0); break;
          case 3: rows_pass(std::integral_constant<int, 3>{}, r0); break;
          default: rows_pass(std::integral_constant<int, 4>{}, r0); break;
        }
      }
      return;
    }
    // int8 path v2 (group-32, KT_MXFP4_G32_INT8). See patches/kt-mxfp4-g32-int8.py.
    int mi8 = 0;
    if (GemmKernelAVX2MXFP4::BufferA::int8_enabled() && a.q8 != nullptr && a.q8_m >= m && m >= 4 &&
        n_end > n_start && group_count % 32 == 0) {
      const int ncols = n_end - n_start;
      const int nblk = group_count / 32;
      const uint8_t* aq = a.q8;
      const float* da = a.da8;
      static thread_local std::vector<int16_t> shmul_s;
      static thread_local std::vector<float> pow2_s, corr_s, outp_s;
      if (shmul_s.size() < (size_t)ncols * group_count) shmul_s.resize((size_t)ncols * group_count);
      if (pow2_s.size() < (size_t)ncols * nblk) pow2_s.resize((size_t)ncols * nblk);
      if (corr_s.size() < (size_t)ncols * nblk) corr_s.resize((size_t)ncols * nblk);
      if (outp_s.size() < (size_t)ncols * 4) outp_s.resize((size_t)ncols * 4);
      int16_t* shmul = shmul_s.data();
      float* pow2 = pow2_s.data();
      float* corr = corr_s.data();
      float* outp = outp_s.data();
      alignas(32) static const int8_t kLutI8[32] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                                    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
      const __m256i lut_i8 = _mm256_load_si256((const __m256i*)kLutI8);
      const __m256i nib_mask256 = _mm256_set1_epi8(0x0F);
      const __m256i lane_shift = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);
      const __m256i ones_u8 = _mm256_set1_epi8(1);
      const __m256i ones_i16 = _mm256_set1_epi16(1);
      // --- per-column tables from the scale bits (scales are exact powers of two)
      bool ok = true;
      for (int ni = n_start; ni < n_end && ok; ni++) {
        const float* sc = b.d + (size_t)ni * group_count;
        const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
        int16_t* shm = shmul + (size_t)(ni - n_start) * group_count;
        for (int blk = 0; blk < nblk; blk++) {
          int emin = 1000, emax = -1000;
          for (int g = blk * 32; g < blk * 32 + 32; g++) {
            uint32_t bits;
            std::memcpy(&bits, &sc[g], 4);
            const int e = (int)((bits >> 23) & 0xFF) - 127;  // sc = 2^e exactly
            emin = e < emin ? e : emin;
            emax = e > emax ? e : emax;
          }
          if (emax - emin > 12) { ok = false; break; }
          pow2[(size_t)(ni - n_start) * nblk + blk] = std::ldexp(0.5f, emin);
          int32_t csum = 0;
          for (int g = blk * 32; g < blk * 32 + 32; g++) {
            uint32_t bits;
            std::memcpy(&bits, &sc[g], 4);
            const int sh = (int)((bits >> 23) & 0xFF) - 127 - emin;
            shm[g] = (int16_t)(1 << sh);
            const __m256i r2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(b_row + (size_t)g * 16)));
            const __m256i v = _mm256_and_si256(_mm256_srlv_epi32(r2, lane_shift), nib_mask256);
            const __m256i w8 = _mm256_shuffle_epi8(lut_i8, v);
            const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, w8), ones_i16);
            csum += hsum_i32(p32) << sh;
          }
          corr[(size_t)(ni - n_start) * nblk + blk] = 128.f * (float)csum;
        }
      }
      if (ok) {
        for (; mi8 + 4 <= m; mi8 += 4) {
          std::fill(outp, outp + (size_t)ncols * 4, 0.f);
          const uint8_t* q0 = aq + (size_t)(mi8 + 0) * k;
          const uint8_t* q1 = aq + (size_t)(mi8 + 1) * k;
          const uint8_t* q2 = aq + (size_t)(mi8 + 2) * k;
          const uint8_t* q3 = aq + (size_t)(mi8 + 3) * k;
          for (int blk = 0; blk < nblk; blk++) {
            const int g0 = blk * 32, g1 = g0 + 32;
            const float d0 = da[(size_t)(mi8 + 0) * nblk + blk], d1 = da[(size_t)(mi8 + 1) * nblk + blk];
            const float d2 = da[(size_t)(mi8 + 2) * nblk + blk], d3 = da[(size_t)(mi8 + 3) * nblk + blk];
            for (int ni = n_start; ni < n_end; ni++) {
              const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
              const int16_t* shm = shmul + (size_t)(ni - n_start) * group_count;
              __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
              __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
              for (int g = g0; g < g1; g++) {
                const __m256i r2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(b_row + (size_t)g * 16)));
                const __m256i v = _mm256_and_si256(_mm256_srlv_epi32(r2, lane_shift), nib_mask256);
                const __m256i w8 = _mm256_shuffle_epi8(lut_i8, v);
                const __m256i mul = _mm256_set1_epi16(shm[g]);
                const size_t off = (size_t)g * 32;
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(q0 + off)), w8), mul));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(q1 + off)), w8), mul));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(q2 + off)), w8), mul));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)(q3 + off)), w8), mul));
              }
              const float cc = corr[(size_t)(ni - n_start) * nblk + blk];
              const float p2 = pow2[(size_t)(ni - n_start) * nblk + blk];
              float* o = outp + (size_t)(ni - n_start) * 4;
              o[0] += ((float)hsum_i32(acc0) - cc) * d0 * p2;
              o[1] += ((float)hsum_i32(acc1) - cc) * d1 * p2;
              o[2] += ((float)hsum_i32(acc2) - cc) * d2 * p2;
              o[3] += ((float)hsum_i32(acc3) - cc) * d3 * p2;
            }
          }
          for (int ni = n_start; ni < n_end; ni++) {
            const float* o = outp + (size_t)(ni - n_start) * 4;
            c.data[(size_t)(mi8 + 0) * n + ni] = o[0];
            c.data[(size_t)(mi8 + 1) * n + ni] = o[1];
            c.data[(size_t)(mi8 + 2) * n + ni] = o[2];
            c.data[(size_t)(mi8 + 3) * n + ni] = o[3];
          }
        }
      }
    }
    // k-blocked (group-32). A 4-token block's activations for kBlk groups
    // (16 KiB at 32 groups) stay in L1 while this thread's columns go by; each
    // column's 8-lane running sums are parked between k blocks. Exact parking,
    // groups summed in order: bit-identical.
    static const int kBlk = [] {
      const char* v = std::getenv("KT_MXFP4_G32_KBLK");
      return v ? std::max(0, std::atoi(v)) : 32;
    }();
    int mi = mi8;
    if (kBlk > 0 && kBlk < group_count && n_end > n_start) {
      const int ncols = n_end - n_start;
      static thread_local std::vector<float> park_storage;
      if (park_storage.size() < (size_t)ncols * 8 * 8) park_storage.resize((size_t)ncols * 8 * 8);
      float* park = park_storage.data();
      // 8-token blocks (group-32). Eight independent group chains per decoded
      // group; the running sums are loaded from and stored to the park buffer
      // around each group's fold (exact), so registers hold only the chains.
      static const bool kT8 = [] {
        const char* v = std::getenv("KT_MXFP4_G32_T8");
        return v ? std::atoi(v) != 0 : false;
      }();
      static thread_local std::vector<float> ibuf_storage;
      if (kT8 && ibuf_storage.size() < (size_t)8 * k) ibuf_storage.resize((size_t)8 * k);
      for (; kT8 && mi + 8 <= m; mi += 8) {
        // The block's eight rows, group-interleaved: group g of token t at
        // ibuf[(g * 8 + t) * 32]. One pointer and constant offsets in the hot
        // loop instead of eight row pointers (which spilled to the stack).
        float* ibuf = ibuf_storage.data();
        for (int g = 0; g < group_count; g++)
          for (int t = 0; t < 8; t++)
            std::memcpy(ibuf + ((size_t)g * 8 + t) * 32, a_perm + (size_t)(mi + t) * k + (size_t)g * 32,
                        32 * sizeof(float));
        std::fill(park, park + (size_t)ncols * 8 * 8, 0.f);
        for (int gb = 0; gb < group_count; gb += kBlk) {
          const int ge = std::min(group_count, gb + kBlk);
          for (int ni = n_start; ni < n_end; ni++) {
            const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
            const float* b_scales = b.d + (size_t)ni * group_count;
            float* pk = park + (size_t)(ni - n_start) * 64;
            for (int g = gb; g < ge; g++) {
              KT_MXFP4_DECODE_GROUP(b_row, g);
              const float* a0 = ibuf + (size_t)g * 256;
              const float* a1 = a0 + 32; const float* a2 = a0 + 64; const float* a3 = a0 + 96;
              const float* a4 = a0 + 128; const float* a5 = a0 + 160; const float* a6 = a0 + 192;
              const float* a7 = a0 + 224;
              __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(a0), w0);
              __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(a1), w0);
              __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(a2), w0);
              __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(a3), w0);
              __m256 g4 = _mm256_mul_ps(_mm256_loadu_ps(a4), w0);
              __m256 g5 = _mm256_mul_ps(_mm256_loadu_ps(a5), w0);
              __m256 g6 = _mm256_mul_ps(_mm256_loadu_ps(a6), w0);
              __m256 g7 = _mm256_mul_ps(_mm256_loadu_ps(a7), w0);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + 8), w1, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + 8), w1, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + 8), w1, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + 8), w1, g3);
              g4 = _mm256_fmadd_ps(_mm256_loadu_ps(a4 + 8), w1, g4);
              g5 = _mm256_fmadd_ps(_mm256_loadu_ps(a5 + 8), w1, g5);
              g6 = _mm256_fmadd_ps(_mm256_loadu_ps(a6 + 8), w1, g6);
              g7 = _mm256_fmadd_ps(_mm256_loadu_ps(a7 + 8), w1, g7);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + 16), w2, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + 16), w2, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + 16), w2, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + 16), w2, g3);
              g4 = _mm256_fmadd_ps(_mm256_loadu_ps(a4 + 16), w2, g4);
              g5 = _mm256_fmadd_ps(_mm256_loadu_ps(a5 + 16), w2, g5);
              g6 = _mm256_fmadd_ps(_mm256_loadu_ps(a6 + 16), w2, g6);
              g7 = _mm256_fmadd_ps(_mm256_loadu_ps(a7 + 16), w2, g7);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + 24), w3, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + 24), w3, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + 24), w3, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + 24), w3, g3);
              g4 = _mm256_fmadd_ps(_mm256_loadu_ps(a4 + 24), w3, g4);
              g5 = _mm256_fmadd_ps(_mm256_loadu_ps(a5 + 24), w3, g5);
              g6 = _mm256_fmadd_ps(_mm256_loadu_ps(a6 + 24), w3, g6);
              g7 = _mm256_fmadd_ps(_mm256_loadu_ps(a7 + 24), w3, g7);
              const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
              _mm256_storeu_ps(pk + 0, _mm256_fmadd_ps(g0, sv, _mm256_loadu_ps(pk + 0)));
              _mm256_storeu_ps(pk + 8, _mm256_fmadd_ps(g1, sv, _mm256_loadu_ps(pk + 8)));
              _mm256_storeu_ps(pk + 16, _mm256_fmadd_ps(g2, sv, _mm256_loadu_ps(pk + 16)));
              _mm256_storeu_ps(pk + 24, _mm256_fmadd_ps(g3, sv, _mm256_loadu_ps(pk + 24)));
              _mm256_storeu_ps(pk + 32, _mm256_fmadd_ps(g4, sv, _mm256_loadu_ps(pk + 32)));
              _mm256_storeu_ps(pk + 40, _mm256_fmadd_ps(g5, sv, _mm256_loadu_ps(pk + 40)));
              _mm256_storeu_ps(pk + 48, _mm256_fmadd_ps(g6, sv, _mm256_loadu_ps(pk + 48)));
              _mm256_storeu_ps(pk + 56, _mm256_fmadd_ps(g7, sv, _mm256_loadu_ps(pk + 56)));
            }
          }
        }
        for (int ni = n_start; ni < n_end; ni++) {
          const float* pk = park + (size_t)(ni - n_start) * 64;
          for (int t = 0; t < 8; t++)
            c.data[(size_t)(mi + t) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk + 8 * t));
        }
      }
      // Short chains (group-32, KT_MXFP4_G32_FAST). Scale folded into the
      // weights once per group, two running sums per token: each chain grows by
      // two FMAs per group instead of five. Not bit-identical (rounding order).
      static const bool kFast = [] {
        const char* v = std::getenv("KT_MXFP4_G32_FAST");
        return v ? std::atoi(v) != 0 : false;
      }();
      if (kFast && park_storage.size() < (size_t)ncols * 8 * 8) {
        park_storage.resize((size_t)ncols * 8 * 8);
        park = park_storage.data();
      }
      for (; kFast && mi + 4 <= m; mi += 4) {
        const float* p0 = a_perm + (size_t)(mi + 0) * k;
        const float* p1 = a_perm + (size_t)(mi + 1) * k;
        const float* p2 = a_perm + (size_t)(mi + 2) * k;
        const float* p3 = a_perm + (size_t)(mi + 3) * k;
        std::fill(park, park + (size_t)ncols * 8 * 8, 0.f);
        for (int gb = 0; gb < group_count; gb += kBlk) {
          const int ge = std::min(group_count, gb + kBlk);
          for (int ni = n_start; ni < n_end; ni++) {
            const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
            const float* b_scales = b.d + (size_t)ni * group_count;
            float* pk = park + (size_t)(ni - n_start) * 64;
            __m256 A0 = _mm256_loadu_ps(pk + 0), B0 = _mm256_loadu_ps(pk + 8);
            __m256 A1 = _mm256_loadu_ps(pk + 16), B1 = _mm256_loadu_ps(pk + 24);
            __m256 A2 = _mm256_loadu_ps(pk + 32), B2 = _mm256_loadu_ps(pk + 40);
            __m256 A3 = _mm256_loadu_ps(pk + 48), B3 = _mm256_loadu_ps(pk + 56);
            for (int g = gb; g < ge; g++) {
              const int base = g * 32;
              KT_MXFP4_DECODE_GROUP(b_row, g);
              const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
              const __m256 s0 = _mm256_mul_ps(w0, sv);
              const __m256 s1 = _mm256_mul_ps(w1, sv);
              const __m256 s2 = _mm256_mul_ps(w2, sv);
              const __m256 s3 = _mm256_mul_ps(w3, sv);
              A0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base), s0, A0);
              A1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base), s0, A1);
              A2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base), s0, A2);
              A3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base), s0, A3);
              B0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), s1, B0);
              B1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 8), s1, B1);
              B2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 8), s1, B2);
              B3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 8), s1, B3);
              A0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 16), s2, A0);
              A1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 16), s2, A1);
              A2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 16), s2, A2);
              A3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 16), s2, A3);
              B0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), s3, B0);
              B1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 24), s3, B1);
              B2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 24), s3, B2);
              B3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 24), s3, B3);
            }
            _mm256_storeu_ps(pk + 0, A0); _mm256_storeu_ps(pk + 8, B0);
            _mm256_storeu_ps(pk + 16, A1); _mm256_storeu_ps(pk + 24, B1);
            _mm256_storeu_ps(pk + 32, A2); _mm256_storeu_ps(pk + 40, B2);
            _mm256_storeu_ps(pk + 48, A3); _mm256_storeu_ps(pk + 56, B3);
          }
        }
        for (int ni = n_start; ni < n_end; ni++) {
          const float* pk = park + (size_t)(ni - n_start) * 64;
          for (int t = 0; t < 4; t++)
            c.data[(size_t)(mi + t) * n + ni] =
                hsum_avx2(_mm256_add_ps(_mm256_loadu_ps(pk + 16 * t), _mm256_loadu_ps(pk + 16 * t + 8)));
        }
      }
      for (; mi + 4 <= m; mi += 4) {
        const float* p0 = a_perm + (size_t)(mi + 0) * k;
        const float* p1 = a_perm + (size_t)(mi + 1) * k;
        const float* p2 = a_perm + (size_t)(mi + 2) * k;
        const float* p3 = a_perm + (size_t)(mi + 3) * k;
        std::fill(park, park + (size_t)ncols * 4 * 8, 0.f);
        for (int gb = 0; gb < group_count; gb += kBlk) {
          const int ge = std::min(group_count, gb + kBlk);
          for (int ni = n_start; ni < n_end; ni++) {
            const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
            const float* b_scales = b.d + (size_t)ni * group_count;
            float* pk = park + (size_t)(ni - n_start) * 32;
            __m256 tot0 = _mm256_loadu_ps(pk), tot1 = _mm256_loadu_ps(pk + 8);
            __m256 tot2 = _mm256_loadu_ps(pk + 16), tot3 = _mm256_loadu_ps(pk + 24);
            for (int g = gb; g < ge; g++) {
              const int base = g * 32;
              KT_MXFP4_DECODE_GROUP(b_row, g);
              __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(p0 + base), w0);
              __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(p1 + base), w0);
              __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(p2 + base), w0);
              __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(p3 + base), w0);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), w1, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 8), w1, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 8), w1, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 8), w1, g3);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 16), w2, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 16), w2, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 16), w2, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 16), w2, g3);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), w3, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 24), w3, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 24), w3, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 24), w3, g3);
              const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
              tot0 = _mm256_fmadd_ps(g0, sv, tot0);
              tot1 = _mm256_fmadd_ps(g1, sv, tot1);
              tot2 = _mm256_fmadd_ps(g2, sv, tot2);
              tot3 = _mm256_fmadd_ps(g3, sv, tot3);
            }
            _mm256_storeu_ps(pk, tot0);
            _mm256_storeu_ps(pk + 8, tot1);
            _mm256_storeu_ps(pk + 16, tot2);
            _mm256_storeu_ps(pk + 24, tot3);
          }
        }
        for (int ni = n_start; ni < n_end; ni++) {
          const float* pk = park + (size_t)(ni - n_start) * 32;
          c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk));
          c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk + 8));
          c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk + 16));
          c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(_mm256_loadu_ps(pk + 24));
        }
      }
      // 0-3 remaining rows fall through to the single-row loop below, which
      // is unchanged (its per-parity accumulators already match).
    }
    if (kTile >= 8 && m >= 8 && mi == 0 && mi8 == 0) {
      static thread_local std::vector<float> wrow_storage;
      if (wrow_storage.size() < (size_t)k) wrow_storage.resize((size_t)k);
      float* wrow = wrow_storage.data();
      for (int t0 = 0; t0 < m; t0 += kTile) {
        const int t1 = std::min(m, t0 + kTile);
        const int t1_4 = t0 + ((t1 - t0) & ~3);
        for (int ni = n_start; ni < n_end; ni++) {
          const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
          const float* b_scales = b.d + (size_t)ni * group_count;
          KT_MXFP4_PREFETCH_NEXT_ROW(ni)
          for (int g = 0; g < group_count; g++) {
            KT_MXFP4_DECODE_GROUP(b_row, g);
            float* wg = wrow + (size_t)g * 32;
            _mm256_storeu_ps(wg, w0);
            _mm256_storeu_ps(wg + 8, w1);
            _mm256_storeu_ps(wg + 16, w2);
            _mm256_storeu_ps(wg + 24, w3);
          }
          for (int bi = t0; bi < t1_4; bi += 4) {
            const float* p0 = a_perm + (size_t)(bi + 0) * k;
            const float* p1 = a_perm + (size_t)(bi + 1) * k;
            const float* p2 = a_perm + (size_t)(bi + 2) * k;
            const float* p3 = a_perm + (size_t)(bi + 3) * k;
            __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
            __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();
            for (int g = 0; g < group_count; g++) {
              const int base = g * 32;
              const float* wg = wrow + base;
              const __m256 w0 = _mm256_loadu_ps(wg);
              const __m256 w1 = _mm256_loadu_ps(wg + 8);
              const __m256 w2 = _mm256_loadu_ps(wg + 16);
              const __m256 w3 = _mm256_loadu_ps(wg + 24);
              __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(p0 + base), w0);
              __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(p1 + base), w0);
              __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(p2 + base), w0);
              __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(p3 + base), w0);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), w1, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 8), w1, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 8), w1, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 8), w1, g3);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 16), w2, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 16), w2, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 16), w2, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 16), w2, g3);
              g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), w3, g0);
              g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 24), w3, g1);
              g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 24), w3, g2);
              g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 24), w3, g3);
              const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
              tot0 = _mm256_fmadd_ps(g0, sv, tot0);
              tot1 = _mm256_fmadd_ps(g1, sv, tot1);
              tot2 = _mm256_fmadd_ps(g2, sv, tot2);
              tot3 = _mm256_fmadd_ps(g3, sv, tot3);
            }
            c.data[(size_t)(bi + 0) * n + ni] = hsum_avx2(tot0);
            c.data[(size_t)(bi + 1) * n + ni] = hsum_avx2(tot1);
            c.data[(size_t)(bi + 2) * n + ni] = hsum_avx2(tot2);
            c.data[(size_t)(bi + 3) * n + ni] = hsum_avx2(tot3);
          }
          // the tile's last 0-3 rows: the single-row arithmetic, weights from L1
          for (int ri = t1_4; ri < t1; ri++) {
            const float* ap_row = a_perm + (size_t)ri * k;
            __m256 total0 = _mm256_setzero_ps();
            __m256 total1 = _mm256_setzero_ps();
            for (int g = 0; g < group_count; g++) {
              const float* ap = ap_row + g * 32;
              const float* wg = wrow + (size_t)g * 32;
              __m256 gacc = _mm256_mul_ps(_mm256_loadu_ps(ap), _mm256_loadu_ps(wg));
              gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), _mm256_loadu_ps(wg + 8), gacc);
              gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 16), _mm256_loadu_ps(wg + 16), gacc);
              gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), _mm256_loadu_ps(wg + 24), gacc);
              const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
              if (g & 1)
                total1 = _mm256_fmadd_ps(gacc, sv, total1);
              else
                total0 = _mm256_fmadd_ps(gacc, sv, total0);
            }
            c.data[(size_t)ri * n + ni] = hsum_avx2(_mm256_add_ps(total0, total1));
          }
        }
      }
      mi = m;  // every row done; the loops below are skipped
    }
    for (; mi + 4 <= m; mi += 4) {
      const float* p0 = a_perm + (size_t)(mi + 0) * k;
      const float* p1 = a_perm + (size_t)(mi + 1) * k;
      const float* p2 = a_perm + (size_t)(mi + 2) * k;
      const float* p3 = a_perm + (size_t)(mi + 3) * k;
      for (int ni = n_start; ni < n_end; ni++) {
        const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
        const float* b_scales = b.d + (size_t)ni * group_count;
        KT_MXFP4_PREFETCH_NEXT_ROW(ni)
        __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
        __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();

        for (int g = 0; g < group_count; g++) {
          const int base = g * 32;
          KT_MXFP4_DECODE_GROUP(b_row, g);

          __m256 g0 = _mm256_mul_ps(_mm256_loadu_ps(p0 + base), w0);
          __m256 g1 = _mm256_mul_ps(_mm256_loadu_ps(p1 + base), w0);
          __m256 g2 = _mm256_mul_ps(_mm256_loadu_ps(p2 + base), w0);
          __m256 g3 = _mm256_mul_ps(_mm256_loadu_ps(p3 + base), w0);
          g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), w1, g0);
          g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 8), w1, g1);
          g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 8), w1, g2);
          g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 8), w1, g3);
          g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 16), w2, g0);
          g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 16), w2, g1);
          g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 16), w2, g2);
          g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 16), w2, g3);
          g0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), w3, g0);
          g1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 24), w3, g1);
          g2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 24), w3, g2);
          g3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 24), w3, g3);

          const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
          tot0 = _mm256_fmadd_ps(g0, sv, tot0);
          tot1 = _mm256_fmadd_ps(g1, sv, tot1);
          tot2 = _mm256_fmadd_ps(g2, sv, tot2);
          tot3 = _mm256_fmadd_ps(g3, sv, tot3);
        }
        c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(tot0);
        c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(tot1);
        c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(tot2);
        c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(tot3);
      }
    }

    // Single-row remainder (also the whole decode path when m == 1).
    // kt-mxfp4-decode-prefetch.py: KT_MXFP4_DEC_PFS=<bytes ahead>, one line every 4 groups
    static const int dec_pfs = [] {
      const char* v = std::getenv("KT_MXFP4_DEC_PFS");
      return v ? std::max(0, std::atoi(v)) : 0;
    }();
    for (; mi < m; mi++) {
      const float* ap_row = a_perm + (size_t)mi * k;
      for (int ni = n_start; ni < n_end; ni++) {
        const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
        const float* b_scales = b.d + (size_t)ni * group_count;
        KT_MXFP4_PREFETCH_NEXT_ROW(ni)
        __m256 total0 = _mm256_setzero_ps();
        __m256 total1 = _mm256_setzero_ps();
        if (b.code != nullptr) {
          if (kt_scale_codes_verify()) b.verify_scale_codes();
          // Same loop; the scale comes from the matrix's table by index.
          const uint8_t* b_codes = b.code + (size_t)ni * group_count;
          const float* lut = b.lut;
          for (int g = 0; g < group_count; g++) {
            const float* ap = ap_row + g * 32;
            if (dec_pfs > 0 && (g & 3) == 0) _mm_prefetch((const char*)b_row + (size_t)g * 16 + dec_pfs, _MM_HINT_T0);
            KT_MXFP4_DECODE_GROUP(b_row, g);

            __m256 gacc = _mm256_mul_ps(_mm256_loadu_ps(ap), w0);
            gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), w1, gacc);
            gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 16), w2, gacc);
            gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), w3, gacc);

            const __m256 sv = _mm256_broadcast_ss(&lut[b_codes[g]]);
            if (g & 1)
              total1 = _mm256_fmadd_ps(gacc, sv, total1);
            else
              total0 = _mm256_fmadd_ps(gacc, sv, total0);
          }
        } else
        for (int g = 0; g < group_count; g++) {
          const float* ap = ap_row + g * 32;
          KT_MXFP4_DECODE_GROUP(b_row, g);

          __m256 gacc = _mm256_mul_ps(_mm256_loadu_ps(ap), w0);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), w1, gacc);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 16), w2, gacc);
          gacc = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), w3, gacc);

          const __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
          if (g & 1)
            total1 = _mm256_fmadd_ps(gacc, sv, total1);
          else
            total0 = _mm256_fmadd_ps(gacc, sv, total0);
        }
        c.data[(size_t)mi * n + ni] = hsum_avx2(_mm256_add_ps(total0, total1));
      }
    }
#undef KT_MXFP4_PREFETCH_NEXT_ROW
#undef KT_MXFP4_DECODE_GROUP
    return;
  }

  // --------------------------------------------------------------------------
  // group_size == 16 fast path (NVFP4-style layouts: E2M1 nibbles with a scale
  // every 16 values).  The 32-value block decode above is reused unchanged: a
  // 16-byte load spans exactly TWO consecutive 16-value groups, and the decode
  // emits w0/w1 entirely from group 2b and w2/w3 entirely from group 2b+1, so
  // the only difference from the group-32 path is that each half of the block
  // is folded with its own scale.  The activation pre-permutation is identical
  // because kPerm never crosses the 16-value halves (indices 0-15 stay in the
  // first half, 16-31 in the second).
  // --------------------------------------------------------------------------
  if (group_size == 16 && (k % 32) == 0 && (a.perm != nullptr || (size_t)m * (size_t)k <= (size_t)(4 << 20))) {
    const int* kPerm = GemmKernelAVX2MXFP4::kGroupPerm;
    static thread_local std::vector<float> a_perm_storage;
    const int block_count = k / 32;  // 32-value decode blocks; 2 groups each
    const float* a_perm = (a.perm != nullptr && a.perm_m >= m) ? a.perm : nullptr;
    if (a_perm == nullptr) {
      if (a_perm_storage.size() < (size_t)m * k) a_perm_storage.resize((size_t)m * k);
      float* tmp = a_perm_storage.data();
      for (int mi = 0; mi < m; mi++) {
        const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
        float* p_row = tmp + (size_t)mi * k;
        for (int blk = 0; blk < block_count; blk++) {
          const int base = blk * 32;
          float* dst = p_row + base;
          for (int j = 0; j < 32; j++) dst[j] = GGML_BF16_TO_FP32(a_row[base + kPerm[j]]);
        }
      }
      a_perm = tmp;
    }

    const __m256i lut_lo256 = _mm256_broadcastsi128_si256(lut_lo);
    const __m256i lut_hi256 = _mm256_broadcastsi128_si256(lut_hi);
    const __m256i zero256 = _mm256_setzero_si256();
    const __m128i nib_mask = _mm_set1_epi8(0x0F);

// Same decode as the group-32 path; here w0/w1 = group 2*blk, w2/w3 = 2*blk+1.
#define KT_MXFP4_DECODE_GROUP(b_row, g)                                              \
  const __m128i raw = _mm_loadu_si128((const __m128i*)((b_row) + (size_t)(g) * 16)); \
  const __m128i lo = _mm_and_si128(raw, nib_mask);                                   \
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nib_mask);                \
  const __m256i v = _mm256_set_m128i(hi, lo);                                        \
  const __m256i bl = _mm256_shuffle_epi8(lut_lo256, v);                              \
  const __m256i bh = _mm256_shuffle_epi8(lut_hi256, v);                              \
  const __m256i u16a = _mm256_unpacklo_epi8(bl, bh);                                 \
  const __m256i u16b = _mm256_unpackhi_epi8(bl, bh);                                 \
  const __m256 w0 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16a));       \
  const __m256 w1 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16a));       \
  const __m256 w2 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16b));       \
  const __m256 w3 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16b))

    // Rows outer, output columns inner. The four activation rows (64 KiB
    // at k = 4096) then stay in L1 across every output column instead of being
    // re-streamed once per column; the weights stream instead and are an
    // eighth the size, and the stores become contiguous. Each output element
    // keeps its own untouched k loop and is written exactly once, so the
    // result is bit-identical.
    int mi = 0;
    for (; mi + 4 <= m; mi += 4) {
      const float* p0 = a_perm + (size_t)(mi + 0) * k;
      const float* p1 = a_perm + (size_t)(mi + 1) * k;
      const float* p2 = a_perm + (size_t)(mi + 2) * k;
      const float* p3 = a_perm + (size_t)(mi + 3) * k;
      for (int ni = n_start; ni < n_end; ni++) {
        const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
        const float* b_scales = b.d + (size_t)ni * group_count;
        const uint8_t* b_codes = b.code ? b.code + (size_t)ni * group_count : nullptr;
        if (b_codes && kt_scale_codes_verify()) b.verify_scale_codes();
        if (ni + 1 < n_end) {
          const char* nr = (const char*)(b.b + (size_t)(ni + 1) * row_bytes);
          _mm_prefetch(nr, _MM_HINT_T0);
          _mm_prefetch(nr + 64, _MM_HINT_T0);
          _mm_prefetch(nr + 128, _MM_HINT_T0);
          _mm_prefetch(nr + 192, _MM_HINT_T0);
        }
          __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
          __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();

          for (int blk = 0; blk < block_count; blk++) {
            const int base = blk * 32;
            KT_MXFP4_DECODE_GROUP(b_row, blk);
            const __m256 sa = _mm256_broadcast_ss(b_codes ? &b.lut[b_codes[2 * blk]] : &b_scales[2 * blk]);
            const __m256 sb = _mm256_broadcast_ss(b_codes ? &b.lut[b_codes[2 * blk + 1]] : &b_scales[2 * blk + 1]);

            __m256 h0 = _mm256_mul_ps(_mm256_loadu_ps(p0 + base), w0);
            __m256 h1 = _mm256_mul_ps(_mm256_loadu_ps(p1 + base), w0);
            __m256 h2 = _mm256_mul_ps(_mm256_loadu_ps(p2 + base), w0);
            __m256 h3 = _mm256_mul_ps(_mm256_loadu_ps(p3 + base), w0);
            h0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 8), w1, h0);
            h1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 8), w1, h1);
            h2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 8), w1, h2);
            h3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 8), w1, h3);
            tot0 = _mm256_fmadd_ps(h0, sa, tot0);
            tot1 = _mm256_fmadd_ps(h1, sa, tot1);
            tot2 = _mm256_fmadd_ps(h2, sa, tot2);
            tot3 = _mm256_fmadd_ps(h3, sa, tot3);

            h0 = _mm256_mul_ps(_mm256_loadu_ps(p0 + base + 16), w2);
            h1 = _mm256_mul_ps(_mm256_loadu_ps(p1 + base + 16), w2);
            h2 = _mm256_mul_ps(_mm256_loadu_ps(p2 + base + 16), w2);
            h3 = _mm256_mul_ps(_mm256_loadu_ps(p3 + base + 16), w2);
            h0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0 + base + 24), w3, h0);
            h1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1 + base + 24), w3, h1);
            h2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2 + base + 24), w3, h2);
            h3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3 + base + 24), w3, h3);
            tot0 = _mm256_fmadd_ps(h0, sb, tot0);
            tot1 = _mm256_fmadd_ps(h1, sb, tot1);
            tot2 = _mm256_fmadd_ps(h2, sb, tot2);
            tot3 = _mm256_fmadd_ps(h3, sb, tot3);
          }
          c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(tot0);
          c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(tot1);
          c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(tot2);
          c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(tot3);      }
    }

      // Single-row remainder (also the whole decode path when m == 1).  The two
      // half-scales per block go to their own accumulators; group-16 folds a
      // scale for each 16-value half, which is twice the group-32 rate, and at
      // m==1 (no token blocking to amortize the decode) that scale work is the
      // path's floor -- roughly 1.4x the group-32 single-row cost.  It is a
      // property of the format's scale density, not the accumulator count: a
      // four-way split by block parity was measured and made no difference.
    for (; mi < m; mi++) {
      const float* ap_row = a_perm + (size_t)mi * k;
      for (int ni = n_start; ni < n_end; ni++) {
        const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
        const float* b_scales = b.d + (size_t)ni * group_count;
        if (ni + 1 < n_end) {
          const char* nr = (const char*)(b.b + (size_t)(ni + 1) * row_bytes);
          _mm_prefetch(nr, _MM_HINT_T0);
          _mm_prefetch(nr + 64, _MM_HINT_T0);
          _mm_prefetch(nr + 128, _MM_HINT_T0);
          _mm_prefetch(nr + 192, _MM_HINT_T0);
        }
          __m256 total0 = _mm256_setzero_ps();
          __m256 total1 = _mm256_setzero_ps();
          if (b.code != nullptr) {
            if (kt_scale_codes_verify()) b.verify_scale_codes();
            // Same loop; the scale comes from the matrix's table by index.
            const uint8_t* b_codes = b.code + (size_t)ni * group_count;
            const float* lut = b.lut;
            for (int blk = 0; blk < block_count; blk++) {
              const float* ap = ap_row + blk * 32;
              KT_MXFP4_DECODE_GROUP(b_row, blk);

              __m256 ha = _mm256_mul_ps(_mm256_loadu_ps(ap), w0);
              ha = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), w1, ha);
              __m256 hb = _mm256_mul_ps(_mm256_loadu_ps(ap + 16), w2);
              hb = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), w3, hb);

              total0 = _mm256_fmadd_ps(ha, _mm256_broadcast_ss(&lut[b_codes[2 * blk]]), total0);
              total1 = _mm256_fmadd_ps(hb, _mm256_broadcast_ss(&lut[b_codes[2 * blk + 1]]), total1);
            }
          } else
          for (int blk = 0; blk < block_count; blk++) {
            const float* ap = ap_row + blk * 32;
            KT_MXFP4_DECODE_GROUP(b_row, blk);

            __m256 ha = _mm256_mul_ps(_mm256_loadu_ps(ap), w0);
            ha = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8), w1, ha);
            __m256 hb = _mm256_mul_ps(_mm256_loadu_ps(ap + 16), w2);
            hb = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24), w3, hb);

            total0 = _mm256_fmadd_ps(ha, _mm256_broadcast_ss(&b_scales[2 * blk]), total0);
            total1 = _mm256_fmadd_ps(hb, _mm256_broadcast_ss(&b_scales[2 * blk + 1]), total1);
          }
          c.data[(size_t)mi * n + ni] = hsum_avx2(_mm256_add_ps(total0, total1));      }
    }
#undef KT_MXFP4_DECODE_GROUP
    return;
  }

  for (int ni = n_start; ni < n_end; ni++) {
    const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
    const float* b_scales = b.d + (size_t)ni * group_count;

    if (ni + 1 < n_end) {
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes), _MM_HINT_T0);
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes + 64), _MM_HINT_T0);
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes + 128), _MM_HINT_T0);
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes + 192), _MM_HINT_T0);
    }

    // 4-token blocked path: decode weight once, fmadd into 4 token accumulators
    int mi = 0;
    for (; mi + 4 <= m; mi += 4) {
      const ggml_bf16_t* a0 = a.data + (size_t)(mi + 0) * a.k;
      const ggml_bf16_t* a1 = a.data + (size_t)(mi + 1) * a.k;
      const ggml_bf16_t* a2 = a.data + (size_t)(mi + 2) * a.k;
      const ggml_bf16_t* a3 = a.data + (size_t)(mi + 3) * a.k;
      __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
      __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();

      for (int g = 0; g < group_count; g++) {
        const int k_base = g * group_size;
        __m256 g0 = _mm256_setzero_ps(), g1 = _mm256_setzero_ps();
        __m256 g2 = _mm256_setzero_ps(), g3 = _mm256_setzero_ps();

        int ki = 0;
        for (; ki + 32 <= group_size; ki += 32) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          __m256 wv0 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 0, lut_lo, lut_hi);
          __m256 wv1 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 4, lut_lo, lut_hi);
          __m256 wv2 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 8, lut_lo, lut_hi);
          __m256 wv3 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 12, lut_lo, lut_hi);

          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki), wv0, g0);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki + 8), wv1, g0);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki + 16), wv2, g0);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki + 24), wv3, g0);

          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki), wv0, g1);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki + 8), wv1, g1);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki + 16), wv2, g1);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki + 24), wv3, g1);

          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki), wv0, g2);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki + 8), wv1, g2);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki + 16), wv2, g2);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki + 24), wv3, g2);

          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki), wv0, g3);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki + 8), wv1, g3);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki + 16), wv2, g3);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki + 24), wv3, g3);
        }
        for (; ki + 8 <= group_size; ki += 8) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          __m256 wv = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w, lut_lo, lut_hi);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki), wv, g0);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki), wv, g1);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki), wv, g2);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki), wv, g3);
        }

        __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
        tot0 = _mm256_fmadd_ps(g0, sv, tot0);
        tot1 = _mm256_fmadd_ps(g1, sv, tot1);
        tot2 = _mm256_fmadd_ps(g2, sv, tot2);
        tot3 = _mm256_fmadd_ps(g3, sv, tot3);
      }
      c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(tot0);
      c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(tot1);
      c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(tot2);
      c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(tot3);
    }

    // M-tail: single token fallback
    for (; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
      __m256 total_acc = _mm256_setzero_ps();
      float scalar_tail = 0.0f;

      for (int g = 0; g < group_count; g++) {
        const float scale = b_scales[g];
        const int k_base = g * group_size;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

        int ki = 0;
        for (; ki + 32 <= group_size; ki += 32) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          acc0 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 0, lut_lo, lut_hi), acc0);
          acc1 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 8),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 4, lut_lo, lut_hi), acc1);
          acc2 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 16),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 8, lut_lo, lut_hi), acc2);
          acc3 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 24),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 12, lut_lo, lut_hi), acc3);
        }
        __m256 g_acc = _mm256_add_ps(_mm256_add_ps(acc0, acc2), _mm256_add_ps(acc1, acc3));

        for (; ki + 8 <= group_size; ki += 8) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          g_acc = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki),
                                  GemmKernelAVX2MXFP4::fp4x8_to_fp32(w, lut_lo, lut_hi), g_acc);
        }

        total_acc = _mm256_fmadd_ps(g_acc, _mm256_broadcast_ss(&scale), total_acc);

        for (; ki < group_size; ki++) {
          const int pos = k_base + ki;
          const uint8_t packed = b_row[pos / 2];
          const uint8_t nib = (pos & 1) ? (packed >> 4) : (packed & 0x0F);
          scalar_tail += GGML_BF16_TO_FP32(a_row[pos]) * GemmKernelAVX2MXFP4::fp4_scalar(nib) * scale;
        }
      }

      c.data[(size_t)mi * n + ni] = hsum_avx2(total_acc) + scalar_tail;
    }
  }
}

}  // namespace avx2

// ============================================================================
// CRTP MoE wrapper — mirrors AVX2_RAW_INT4_MOE_TP
// ============================================================================
template <class T = avx2::GemmKernelAVX2MXFP4>
class AVX2_MXFP4_MOE_TP : public AVX2_MOE_BASE<T, AVX2_MXFP4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_MXFP4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AVX2_MXFP4_MOE_TP() = default;
  AVX2_MXFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    if (config_.quant_config.group_size == 0 || config_.quant_config.zero_point)
      throw std::runtime_error("MXFP4 AVX2 MoE requires KGroup FP4 without zero point");
    printf("Created AVX2_MXFP4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()),
           config_.quant_config.group_size);
  }

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k, config_.quant_config.group_size);
  }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    // Always allocate full buffer (weights + scales) for both per-expert and flat-buffer modes.
    // Per-expert mode used to only allocate scales and point b into mmap, but that caused
    // use-after-free when Python releases the mmap after load_weights().
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    auto b = std::make_shared<typename T::BufferA>(m, k, config_.quant_config.group_size, data);
    // kt-kernel-nvfp4-cpu-a4.py: NVFP4 (group 16) activations can be quantized as the GPU does
    if (config_.quant_config.group_size == 16) {
      const bool gu = k == (size_t)config_.hidden_size, dn = k == (size_t)config_.intermediate_size;
      if (gu != dn) {
        char name[64];
        snprintf(name, sizeof(name), "KT_CPU_A4_G%s_L%d", gu ? "13" : "2", config_.layer_idx);
        b->a4_global_bits = &kt_knob(name, 0);
      }
    }
    return b;
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    // Always create full BufferB with owned weight buffer for both modes.
    return std::make_shared<typename T::BufferB>((int)n, (int)k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_scale_only_impl(size_t n, size_t k, void* scales) const {
    return std::make_shared<typename T::BufferB>((int)n, (int)k, config_.quant_config.group_size, scales, nullptr);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avx2::gemm_mxfp4(m, config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert_idx], *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    avx2::gemm_mxfp4(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx], *down_bb_[expert_idx],
                     *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    const bool use_per_expert = !config_.gate_projs.empty();
    if (!use_per_expert && config_.gate_proj == nullptr)
      throw std::runtime_error("MXFP4 AVX2 MoE requires weight pointers");
    if (!use_per_expert && config_.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 AVX2 MoE requires scale pointers");

    if (use_per_expert) {
      // C1: Per-expert mode with TP > 1 must use TP_MOE wrapper which has full_intermediate_size.
      // Direct use of AVX2_MXFP4_MOE_TP with tp_part_idx > 0 in per-expert mode is not supported
      // because we cannot compute correct offsets without knowing full_intermediate_size.
      if (tp_part_idx > 0) {
        throw std::runtime_error(
            "AVX2_MXFP4_MOE_TP::load_weights() per-expert mode with tp_part_idx > 0 is not supported. "
            "Use TP_MOE<AVX2_MXFP4_MOE_TP> wrapper for multi-TP scenarios.");
      }

      // Copy weights from mmap into owned buffers (aligned with AMX behavior).
      // Previously used direct pointers into mmap, but Python releases mmap after load_weights(),
      // causing use-after-free. Now we memcpy to decouple from mmap lifecycle.
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_idx) {
            if (expert_idx < 0 || expert_idx >= config_.expert_num || gate_bb_[expert_idx] == nullptr) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            // H1 & H2: Validate source pointers and lid bounds
            if (lid >= config_.gate_projs[0].size()) {
              throw std::runtime_error("load_weights: lid " + std::to_string(lid) +
                                       " out of bounds (size=" + std::to_string(config_.gate_projs[0].size()) + ")");
            }
            if (config_.gate_projs[0][lid] == nullptr || config_.up_projs[0][lid] == nullptr ||
                config_.down_projs[0][lid] == nullptr) {
              throw std::runtime_error("load_weights: null weight pointer for expert " + std::to_string(lid));
            }
            // tp_part_idx == 0 guaranteed here, so offset is 0. Copy full expert weights.
            gate_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.gate_projs[0][lid], 0, 1);
            up_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.up_projs[0][lid], 0, 1);
            down_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.down_projs[0][lid], 0, 1);
          },
          nullptr);

      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id;
            if (expert_idx >= (uint64_t)config_.expert_num || gate_bb_[expert_idx] == nullptr) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            // H2: Bounds check (already validated in weight loading, but be safe)
            if (lid >= config_.gate_scales[0].size()) return;
            if (config_.gate_scales[0][lid] == nullptr || config_.up_scales[0][lid] == nullptr ||
                config_.down_scales[0][lid] == nullptr)
              return;
            size_t scale_elem_count = ((size_t)config_.hidden_size * config_.intermediate_size) / group_size;
            // tp_part_idx == 0 guaranteed here, so offset is 0
            convert_or_copy(gate_bb_[expert_idx]->d, (const ggml_bf16_t*)config_.gate_scales[0][lid], scale_elem_count);
            convert_or_copy(up_bb_[expert_idx]->d, (const ggml_bf16_t*)config_.up_scales[0][lid], scale_elem_count);
            convert_or_copy(down_bb_[expert_idx]->d, (const ggml_bf16_t*)config_.down_scales[0][lid], scale_elem_count);
          },
          nullptr);
    } else {
      // Flat-buffer mode: copy TP-sliced weights, convert BF16 scales → FP32.
      int nth_gate = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth_gate * config_.expert_num, nullptr,
          [this, nth_gate, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth_gate;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth_gate;
            size_t weight_offset = ((size_t)lid * config_.intermediate_size * config_.hidden_size) / 2;
            gate_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.gate_proj + weight_offset, ith, nth_gate);
            up_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.up_proj + weight_offset, ith, nth_gate);
          },
          nullptr);

      int nth_down = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth_down * config_.expert_num, nullptr,
          [this, nth_down, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth_down;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth_down;
            size_t weight_offset = ((size_t)lid * config_.hidden_size * config_.intermediate_size) / 2;
            down_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.down_proj + weight_offset, ith, nth_down);
          },
          nullptr);

      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            size_t scale_elem_count = ((size_t)config_.hidden_size * config_.intermediate_size) / group_size;
            convert_or_copy(gate_bb_[expert_idx]->d, (ggml_bf16_t*)config_.gate_scale + lid * scale_elem_count,
                            scale_elem_count);
            convert_or_copy(up_bb_[expert_idx]->d, (ggml_bf16_t*)config_.up_scale + lid * scale_elem_count,
                            scale_elem_count);
            convert_or_copy(down_bb_[expert_idx]->d, (ggml_bf16_t*)config_.down_scale + lid * scale_elem_count,
                            scale_elem_count);
          },
          nullptr);
    }
  }

  static inline void fp32_to_bf16(ggml_bf16_t* dst, const float* src, size_t count) {
    convert_or_copy(dst, src, count);
  }

  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    if (expert_id < 0 || expert_id >= config_.expert_num || gate_bb_[expert_id] == nullptr ||
        up_bb_[expert_id] == nullptr || down_bb_[expert_id] == nullptr)
      throw std::runtime_error("MXFP4 AVX2 write_weights_to_buffer: invalid expert");

    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    size_t cpu_tp_weight_bytes = (size_t)config_.intermediate_size * config_.hidden_size / 2;
    size_t cpu_tp_scale_elem_count = (size_t)config_.intermediate_size * config_.hidden_size / group_size;
    size_t gpu_tp_weight_bytes = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count / 2;
    size_t gpu_tp_scale_elem_count =
        (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count / group_size;
    // KT_WRITE_UP_FIRST=1: w13 as [up; gate] instead of [gate; up], weights
    // and scales alike, for consumers that store up first (FlashInfer CUTLASS
    // MXFP4): one contiguous copy per expert instead of two.
    static const bool up_first = getenv("KT_WRITE_UP_FIRST") != nullptr && getenv("KT_WRITE_UP_FIRST")[0] == '1';
    const size_t gate_w_half = up_first ? gpu_tp_weight_bytes : 0, up_w_half = up_first ? 0 : gpu_tp_weight_bytes;
    const size_t gate_s_half = up_first ? gpu_tp_scale_elem_count : 0,
                 up_s_half = up_first ? 0 : gpu_tp_scale_elem_count;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);
      uint8_t* w13_wdst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_sdst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_wdst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_sdst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];
      size_t w_off = (size_t)local_idx * cpu_tp_weight_bytes;
      size_t s_off = (size_t)local_idx * cpu_tp_scale_elem_count;

      constexpr int NWT = 8;
      int ndt = std::min(std::max(1, config_.hidden_size / 128), 32);
      size_t wchunk = ((cpu_tp_weight_bytes + NWT - 1) / NWT + 63) & ~63ULL;
      pool->do_work_stealing_job(
          NWT * 2 + ndt + 2, nullptr,
          [=, this](int tid) {
            if (tid < NWT) {
              size_t s = (size_t)tid * wchunk, e = std::min(s + wchunk, cpu_tp_weight_bytes);
              if (s < e) std::memcpy(w13_wdst + w_off + gate_w_half + s, gate_bb_[expert_id]->b + s, e - s);
            } else if (tid < NWT * 2) {
              size_t s = (size_t)(tid - NWT) * wchunk, e = std::min(s + wchunk, cpu_tp_weight_bytes);
              if (s < e) std::memcpy(w13_wdst + w_off + up_w_half + s, up_bb_[expert_id]->b + s, e - s);
            } else if (tid < NWT * 2 + ndt) {
              size_t cols_per = (config_.hidden_size + ndt - 1) / ndt;
              size_t cs = (size_t)(tid - NWT * 2) * cols_per;
              size_t ce = std::min(cs + cols_per, (size_t)config_.hidden_size);
              size_t wpc = config_.intermediate_size >> 1;
              size_t spc = config_.intermediate_size / group_size;
              size_t gws = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gss = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gwo = (size_t)local_idx * wpc, gso = (size_t)local_idx * spc;
              for (size_t col = cs; col < ce; col++) {
                std::memcpy(w2_wdst + col * gws + gwo, down_bb_[expert_id]->b + col * wpc, wpc);
                fp32_to_bf16(w2_sdst + col * gss + gso, down_bb_[expert_id]->d + col * spc, spc);
              }
            } else if (tid == NWT * 2 + ndt) {
              fp32_to_bf16(w13_sdst + s_off + gate_s_half, gate_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            } else {
              fp32_to_bf16(w13_sdst + s_off + up_s_half, up_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      // cpu_tp_count < gpu_tp_count: one CPU partition feeds multiple GPU TPs
      int gpu_per_cpu = gpu_tp_count / cpu_tp_count;
      int start_gtp = tp_part_idx * gpu_per_cpu;
      size_t dw = cpu_tp_weight_bytes / gpu_per_cpu;
      size_t ds = cpu_tp_scale_elem_count / gpu_per_cpu;
      constexpr int NWT = 8;
      int ndt = std::min(std::max(1, config_.hidden_size / 128), 32);
      size_t wchunk = ((dw + NWT - 1) / NWT + 63) & ~63ULL;
      int tpg = NWT * 2 + ndt + 2;
      pool->do_work_stealing_job(
          tpg * gpu_per_cpu, nullptr,
          [=, this, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs](int task_id) {
            int lg = task_id / tpg;
            int tt = task_id % tpg;
            int gtp = start_gtp + lg;
            uint8_t* w13_wdst = (uint8_t*)w13_weight_ptrs[gtp];
            ggml_bf16_t* w13_sdst = (ggml_bf16_t*)w13_scale_ptrs[gtp];
            uint8_t* w2_wdst = (uint8_t*)w2_weight_ptrs[gtp];
            ggml_bf16_t* w2_sdst = (ggml_bf16_t*)w2_scale_ptrs[gtp];
            size_t cow = (size_t)lg * dw, cos = (size_t)lg * ds;
            if (tt < NWT) {
              size_t s = (size_t)tt * wchunk, e = std::min(s + wchunk, dw);
              if (s < e) std::memcpy(w13_wdst + gate_w_half + s, gate_bb_[expert_id]->b + cow + s, e - s);
            } else if (tt < NWT * 2) {
              size_t s = (size_t)(tt - NWT) * wchunk, e = std::min(s + wchunk, dw);
              if (s < e) std::memcpy(w13_wdst + up_w_half + s, up_bb_[expert_id]->b + cow + s, e - s);
            } else if (tt < NWT * 2 + ndt) {
              size_t cols_per = (config_.hidden_size + ndt - 1) / ndt;
              size_t cs = (size_t)(tt - NWT * 2) * cols_per;
              size_t ce = std::min(cs + cols_per, (size_t)config_.hidden_size);
              size_t wgc = (config_.intermediate_size / gpu_per_cpu) >> 1;
              size_t sgc = (config_.intermediate_size / gpu_per_cpu) / group_size;
              for (size_t col = cs; col < ce; col++) {
                size_t cwoff = col * config_.intermediate_size / 2 + lg * dw / config_.hidden_size;
                size_t csoff = col * (config_.intermediate_size / group_size) + lg * ds / config_.hidden_size;
                std::memcpy(w2_wdst + col * wgc, down_bb_[expert_id]->b + cwoff, wgc);
                fp32_to_bf16(w2_sdst + col * sgc, down_bb_[expert_id]->d + csoff, sgc);
              }
            } else if (tt == NWT * 2 + ndt) {
              fp32_to_bf16(w13_sdst + gate_s_half, gate_bb_[expert_id]->d + cos, ds);
            } else {
              fp32_to_bf16(w13_sdst + up_s_half, up_bb_[expert_id]->d + cos, ds);
            }
          },
          nullptr);
    }
  }
};

// TP_MOE specialization (boilerplate, identical to rawint4 pattern)
template <typename K>
class TP_MOE<AVX2_MXFP4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>> {
  std::vector<void*> tp_owned_down_bufs_;
  std::vector<void*> tp_owned_gate_bufs_;
  std::vector<void*> tp_owned_up_bufs_;
  using ExpertArena = typename AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>::ExpertArena;
  // KT_EXPERT_SHM=1: per part [gate_w, up_w, down_w] memfd arenas holding
  // the packed weights (expert eid at base + eid * stride).
  std::vector<std::vector<ExpertArena>> tp_weight_arenas_;

 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>>;
  using Base::Base;

  ~TP_MOE() {
    for (auto& part : tp_weight_arenas_)
      for (auto& a : part) AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>::free_expert_arena(a);
    if (!tp_weight_arenas_.empty()) return;  // those pointers were arena bases
    for (void* p : tp_owned_down_bufs_)
      if (p) std::free(p);
    for (void* p : tp_owned_gate_bufs_)
      if (p) std::free(p);
    for (void* p : tp_owned_up_bufs_)
      if (p) std::free(p);
  }

  // Per TP part: [(fd, size, stride)] for gate_w, up_w, down_w, gate_s,
  // up_s, down_s (KT_EXPERT_SHM=1; empty otherwise). Weights are packed
  // e2m1 bytes, N x K/2 (gate/up) or K x N/2 (down) per expert; scales fp32.
  std::vector<std::vector<std::tuple<int, size_t, size_t>>> expert_arena_infos() const {
    std::vector<std::vector<std::tuple<int, size_t, size_t>>> out;
    for (size_t i = 0; i < this->tps.size(); i++) {
      std::vector<std::tuple<int, size_t, size_t>> v;
      if (i < tp_weight_arenas_.size())
        for (auto& a : tp_weight_arenas_[i]) v.emplace_back(a.fd, a.size, a.stride);
      for (auto& t : this->tps[i]->expert_arena_info()) v.push_back(t);
      out.push_back(v);
    }
    return out;
  }

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    bool use_per_expert = !config.gate_projs.empty();
    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 AVX2 MoE only supports packed FP4 with KGroup scales");

    if (use_per_expert) {
      int group_size = config.quant_config.group_size;
      int full_interm = config.intermediate_size;

      // Allocate per-partition repack buffers for all projections.
      // Previously gate/up used direct mmap pointers, causing use-after-free when Python
      // releases mmap. Now we memcpy all weights to decouple from mmap lifecycle (aligned with AMX).
      tp_owned_down_bufs_.resize(this->tp_count, nullptr);
      tp_owned_gate_bufs_.resize(this->tp_count, nullptr);
      tp_owned_up_bufs_.resize(this->tp_count, nullptr);
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto* tp = tps[i].get();
        auto& tpc = tp->config_;
        tpc.physical_to_logical_map = config.physical_to_logical_map;
        int per_tp_interm = tpc.intermediate_size;
        if (per_tp_interm % 2 != 0)
          throw std::runtime_error("MXFP4 TP: per_tp_interm must be even for nibble-aligned addressing, got " +
                                   std::to_string(per_tp_interm));
        // gate/up: [intermediate × hidden] = [N × K], TP slices along N → contiguous rows
        size_t gate_up_wt_per_expert = (size_t)per_tp_interm * tpc.hidden_size / 2;
        size_t gate_up_alloc = ((size_t)tpc.expert_num * gate_up_wt_per_expert + 63) & ~(size_t)63;
        size_t down_wt_per_expert = (size_t)tpc.hidden_size * per_tp_interm / 2;
        size_t alloc_size = ((size_t)tpc.expert_num * down_wt_per_expert + 63) & ~(size_t)63;
        uint8_t *gate_buf, *up_buf, *down_buf;
        if (AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>::expert_shm_enabled()) {
          // Weight bytes per expert are 64-aligned already for these shapes,
          // so arena stride == gate_up_wt_per_expert and the per-expert
          // pointer arithmetic below stays valid.
          if (gate_up_wt_per_expert % 64 || down_wt_per_expert % 64)
            throw std::runtime_error("KT_EXPERT_SHM: per-expert weight bytes must be 64-aligned");
          std::vector<ExpertArena> arenas;
          arenas.push_back(AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>::make_expert_arena(
              gate_up_wt_per_expert, tpc.expert_num, "kt_gate_w"));
          arenas.push_back(AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>::make_expert_arena(
              gate_up_wt_per_expert, tpc.expert_num, "kt_up_w"));
          arenas.push_back(AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>::make_expert_arena(
              down_wt_per_expert, tpc.expert_num, "kt_down_w"));
          gate_buf = arenas[0].base;
          up_buf = arenas[1].base;
          down_buf = arenas[2].base;
          {
            static std::mutex m;
            std::lock_guard<std::mutex> g(m);
            if (tp_weight_arenas_.size() < this->tps.size()) tp_weight_arenas_.resize(this->tps.size());
            tp_weight_arenas_[i] = std::move(arenas);
          }
        } else {
          gate_buf = (uint8_t*)std::aligned_alloc(64, gate_up_alloc);
          up_buf = (uint8_t*)std::aligned_alloc(64, gate_up_alloc);
          if (!gate_buf || !up_buf) throw std::runtime_error("aligned_alloc failed for MXFP4 gate/up_buf");
          // down: [hidden × intermediate], TP slices along K → non-contiguous
          down_buf = (uint8_t*)std::aligned_alloc(64, alloc_size);
          if (!down_buf) throw std::runtime_error("aligned_alloc failed for MXFP4 down_buf");
        }
        tp_owned_gate_bufs_[i] = gate_buf;
        tp_owned_up_bufs_[i] = up_buf;
        tp_owned_down_bufs_[i] = down_buf;

        auto subpool = pool->get_subpool(i);
        subpool->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i, per_tp_interm, full_interm, gate_buf, up_buf, down_buf, gate_up_wt_per_expert,
             down_wt_per_expert](int eid) {
              if (tpc.should_skip_expert(eid)) return;
              uint64_t lid = expert_map(physical_to_logical_map, eid);

              // H1 & H2: Validate source pointers and lid bounds
              if (lid >= config.gate_projs[0].size()) {
                throw std::runtime_error("TP_MOE load_weights: lid " + std::to_string(lid) +
                                         " out of bounds (size=" + std::to_string(config.gate_projs[0].size()) + ")");
              }
              if (config.gate_projs[0][lid] == nullptr || config.up_projs[0][lid] == nullptr ||
                  config.down_projs[0][lid] == nullptr) {
                throw std::runtime_error("TP_MOE load_weights: null weight pointer for expert " + std::to_string(lid));
              }

              // gate/up weights: N-split, contiguous rows → memcpy TP slice into owned buffer
              // Each TP partition handles per_tp_interm rows starting at row i*per_tp_interm
              size_t n_byte_off = (size_t)i * per_tp_interm * tpc.hidden_size / 2;
              uint8_t* dst_gate = gate_buf + (size_t)eid * gate_up_wt_per_expert;
              uint8_t* dst_up = up_buf + (size_t)eid * gate_up_wt_per_expert;
              std::memcpy(dst_gate, (const uint8_t*)config.gate_projs[0][lid] + n_byte_off, gate_up_wt_per_expert);
              std::memcpy(dst_up, (const uint8_t*)config.up_projs[0][lid] + n_byte_off, gate_up_wt_per_expert);
              tp->gate_bb_[eid]->b = dst_gate;
              tp->up_bb_[eid]->b = dst_up;

              // gate/up scales: contiguous → convert BF16→FP32
              size_t scale_count = (size_t)(tpc.hidden_size / group_size) * per_tp_interm;
              size_t scale_off = (size_t)i * per_tp_interm * (tpc.hidden_size / group_size);
              if (config.gate_scales[0][lid] == nullptr || config.up_scales[0][lid] == nullptr) {
                throw std::runtime_error("TP_MOE load_weights: null scale pointer for expert " + std::to_string(lid));
              }
              convert_or_copy(tp->gate_bb_[eid]->d, (const ggml_bf16_t*)config.gate_scales[0][lid] + scale_off,
                              scale_count);
              convert_or_copy(tp->up_bb_[eid]->d, (const ggml_bf16_t*)config.up_scales[0][lid] + scale_off,
                              scale_count);

              // down weights: K-split, non-contiguous → per-row memcpy into repack buf
              uint8_t* src_down = (uint8_t*)config.down_projs[0][lid];
              uint8_t* dst_down = down_buf + (size_t)eid * down_wt_per_expert;
              for (int row = 0; row < tpc.hidden_size; row++) {
                std::memcpy(dst_down + (size_t)row * per_tp_interm / 2,
                            src_down + (size_t)row * full_interm / 2 + (size_t)i * per_tp_interm / 2,
                            per_tp_interm / 2);
              }
              tp->down_bb_[eid]->b = dst_down;

              // down scales: K-split, non-contiguous → per-row convert
              if (config.down_scales[0][lid] == nullptr) {
                throw std::runtime_error("TP_MOE load_weights: null down_scale pointer for expert " +
                                         std::to_string(lid));
              }
              int full_interm_groups = full_interm / group_size;
              int per_tp_groups = per_tp_interm / group_size;
              const ggml_bf16_t* src_ds = (const ggml_bf16_t*)config.down_scales[0][lid];
              float* dst_ds = tp->down_bb_[eid]->d;
              for (int row = 0; row < tpc.hidden_size; row++) {
                convert_or_copy(dst_ds + (size_t)row * per_tp_groups,
                                src_ds + (size_t)row * full_interm_groups + (size_t)i * per_tp_groups, per_tp_groups);
              }
            },
            nullptr);
      });
    } else {
      int group_size = config.quant_config.group_size;
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = tps[i]->config_;
        if (tpc.intermediate_size % 2 != 0)
          throw std::runtime_error(
              "MXFP4 TP flat-buffer: intermediate_size must be even for nibble-aligned addressing, got " +
              std::to_string(tpc.intermediate_size));
        size_t weight_elem_count = (size_t)tpc.intermediate_size * tpc.hidden_size;
        size_t scales_elem_count = ((size_t)tpc.hidden_size / group_size) * tpc.intermediate_size;
        tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.gate_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
        tpc.up_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
        tpc.down_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      });

      // Fill the TP-sliced flat buffers from full-model data, then load.
      // TP slicing is along the intermediate dimension. gate/up are
      // [intermediate × hidden] with intermediate as outer (N) dim — contiguous
      // memcpy slice works. down is [hidden × intermediate] with intermediate
      // as inner (K) dim — must loop per hidden row.
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = tps[i]->config_;
        size_t weight_elem_count = (size_t)tpc.intermediate_size * tpc.hidden_size;
        size_t scales_elem_count = ((size_t)tpc.hidden_size / group_size) * tpc.intermediate_size;
        size_t n_per_tp = tpc.intermediate_size;
        size_t hidden_groups_per_row = (size_t)tpc.hidden_size / group_size;
        size_t interm_groups_per_row = (size_t)tpc.intermediate_size / group_size;
        size_t full_interm_groups_per_row = (size_t)config.intermediate_size / group_size;
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int eid) {
              uint64_t lid = expert_map(physical_to_logical_map, eid);
              uint8_t* src_gate = (uint8_t*)config.gate_proj +
                                  (lid * (size_t)config.intermediate_size + i * n_per_tp) * config.hidden_size / 2;
              uint8_t* src_up = (uint8_t*)config.up_proj +
                                (lid * (size_t)config.intermediate_size + i * n_per_tp) * config.hidden_size / 2;
              std::memcpy((uint8_t*)tpc.gate_proj + eid * weight_elem_count / 2, src_gate, weight_elem_count / 2);
              std::memcpy((uint8_t*)tpc.up_proj + eid * weight_elem_count / 2, src_up, weight_elem_count / 2);
              ggml_bf16_t* src_gate_scale =
                  (ggml_bf16_t*)config.gate_scale +
                  (lid * (size_t)config.intermediate_size + i * n_per_tp) * hidden_groups_per_row;
              ggml_bf16_t* src_up_scale =
                  (ggml_bf16_t*)config.up_scale +
                  (lid * (size_t)config.intermediate_size + i * n_per_tp) * hidden_groups_per_row;
              std::memcpy((ggml_bf16_t*)tpc.gate_scale + eid * scales_elem_count, src_gate_scale,
                          scales_elem_count * sizeof(ggml_bf16_t));
              std::memcpy((ggml_bf16_t*)tpc.up_scale + eid * scales_elem_count, src_up_scale,
                          scales_elem_count * sizeof(ggml_bf16_t));
              uint8_t* src_down =
                  (uint8_t*)config.down_proj + lid * (size_t)config.hidden_size * config.intermediate_size / 2;
              ggml_bf16_t* src_down_scale =
                  (ggml_bf16_t*)config.down_scale + lid * (size_t)config.hidden_size * full_interm_groups_per_row;
              for (size_t col = 0; col < (size_t)tpc.hidden_size; col++) {
                std::memcpy((uint8_t*)tpc.down_proj + eid * weight_elem_count / 2 + col * n_per_tp / 2,
                            src_down + (col * config.intermediate_size + i * n_per_tp) / 2, n_per_tp / 2);
                std::memcpy((ggml_bf16_t*)tpc.down_scale + eid * scales_elem_count + col * interm_groups_per_row,
                            src_down_scale + col * full_interm_groups_per_row + i * interm_groups_per_row,
                            interm_groups_per_row * sizeof(ggml_bf16_t));
              }
            },
            nullptr);
      });

      DO_TPS_LOAD_WEIGHTS(pool);

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = tps[i]->config_;
        delete[] (uint8_t*)tpc.gate_proj;
        delete[] (uint8_t*)tpc.up_proj;
        delete[] (uint8_t*)tpc.down_proj;
        delete[] (ggml_bf16_t*)tpc.gate_scale;
        delete[] (ggml_bf16_t*)tpc.up_scale;
        delete[] (ggml_bf16_t*)tpc.down_scale;
      });
    }

    if (kt_mxfp4_scale_codes()) {
      std::atomic<size_t> built{0}, total{0};
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto* tp = tps[i].get();
        auto& tpc = tp->config_;
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, tp](int eid) {
              if (tpc.should_skip_expert(eid) && !kt_scale_codes_load_all()) return;
              for (auto* bb : {tp->gate_bb_[eid].get(), tp->up_bb_[eid].get(), tp->down_bb_[eid].get()}) {
                if (bb == nullptr) continue;
                total++;
                if (bb->build_scale_codes()) built++;
              }
            },
            nullptr);
      });
      printf("KT_MXFP4_SCALE_CODES: %zu of %zu matrices read one-byte scale codes\n", built.load(), total.load());
    }

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w13_scale_ptrs.size() != gpu_tp_count ||
        (int)w2_weight_ptrs.size() != gpu_tp_count || (int)w2_scale_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }
    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H
