// Contract v3 CPU kernels, arena / row-major input mode: row-wise decode. See v3_rw.h and
// concept-r1/CPU_KERNEL_COMPUTE_r1.md. No OpenMP.
// Build: -O3 -mavx2 -mfma -ffp-contract=off -fno-fast-math (the flags of v3_rm.cpp; the same CMake source list).
//
// Per weight row and pair of 32-blocks (32 bytes):  lo = x & 15, hi = (x >> 4) & 15, one byte shuffle each through the
// E2M1 table plus 12 (0..24), then
//   s = maddubs(w_lo + 12, a_even) + maddubs(w_hi + 12, a_odd)      16 int16: 8 per block, 4 products each, |s| <= 12,288
// The 8 rows' s are reduced by hadd (int16, <= 24,576), madd (int32), hadd (int32) and a lane permute to the served
// layout: one int32 per column and block, lane c = column c. Subtracting 12 * (sum of the block's activations) gives
// the exact block dot (<= 49,152), the integer v3_rm.cpp's rows_dot returns. From there on the code is the served one:
//   gate / up: epilogue<R> of v3_common.inc, block g then g + 1, or its fast path alone when one pass over the panel's
//              codes and the call's activation exponents proves every block of the panel is on it;
//   down:      epi_i64<R>, or one int32 sum per (row, column) of dot << (shift - min shift) widened once, where every
//              shift of the panel is a left shift of at most 38 and a column's shifts differ by at most `spread` bits.
// pathb-cpu-r1 (13 Nov, ZEN3_OPTIMISATION_r1.md section 3.2 step 4): KT_V3_RW_RED=2 selects reduction 2 at run time:
// madd + unpack + add in place of the microcoded vphaddw / vphaddd, the activations of a block pair as memory operands of
// vpmaddubsw, and the down kernel's shift spread computed once a call. Unset (or 1): the 3bb076a kernel, unchanged.
#define V3_RW_NO_REDIRECT
#include "v3_rw.h"

#include <immintrin.h>

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef V3_RW_ORDER
// 0: all 8 rows' sums, then the reduction; 1: the reduction inside each quad of rows. Same bits (run kc-r2-1794497289:
// 0 differing words for both); 1 is as fast at 1 row and 4 to 19% faster from 2 rows up, so it is the default.
#define V3_RW_ORDER 1
#endif

namespace {
#include "v3_common.inc"

#define RW_INLINE __attribute__((always_inline)) inline
enum { RW_MAXG = 16 };

struct RwFpScope {  // RNE, FTZ = DAZ = 0 for this call; the caller's state is restored (v3_rm.cpp's FpScope)
  unsigned prev;
  RwFpScope() : prev(_mm_getcsr()) { _mm_setcsr(prev & ~(0x8040u | 0x6000u)); }
  ~RwFpScope() { _mm_setcsr(prev); }
};

std::atomic<int> rw_on{-1};
// pathb-cpu-r1: the reduction (1: order V3_RW_ORDER with vphadd, the 3bb076a kernel; 2: microcode-free, activations as
// memory operands, spread hoisted). KT_V3_RW_RED=2 (read once) or v3_rw_set_red; a build with -DV3_RW_RED_DEFAULT=2
// (the kc bench's third copy) starts at 2 and ignores the environment.
#ifdef V3_RW_RED_DEFAULT
std::atomic<int> rw_red{V3_RW_RED_DEFAULT};
#else
std::atomic<int> rw_red{-1};
#endif

// kJ + 12
alignas(32) const uint8_t rw_JB[32] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0,
                                       12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};

// The call's activations, de-interleaved: ae[m][i] = q[m][2i], ao[m][i] = q[m][2i + 1]; cb[m][g] = 12 * sum of block g.
struct RwAct {
  std::vector<int8_t> ae, ao;
  std::vector<int32_t> cb;
  int ld = 0, ldg = 0;
};

void rw_prep(RwAct& A, const int8_t* q, int ldq, int M, int K) {
  const int G = K / 32, hk = K / 2;
  const size_t n = (size_t)M * hk;
  if (A.ae.size() < n) {
    A.ae.resize(n);
    A.ao.resize(n);
  }
  if (A.cb.size() < (size_t)M * G) A.cb.resize((size_t)M * G);
  A.ld = hk;
  A.ldg = G;
  const __m256i lo8 = _mm256_set1_epi16(0x00FF), sx = _mm256_set1_epi8((char)0x80), zero = _mm256_setzero_si256();
  for (int m = 0; m < M; m++)
    for (int g = 0; g < G; g++) {
      const __m256i x = _mm256_loadu_si256((const __m256i*)(q + (size_t)m * ldq + (size_t)g * 32));
      // packus: [e0-7 o0-7 | e8-15 o8-15]; the 64-bit permute makes it [e0-15 | o0-15]
      const __m256i pk = _mm256_permute4x64_epi64(
          _mm256_packus_epi16(_mm256_and_si256(x, lo8), _mm256_srli_epi16(x, 8)), 0xD8);
      const size_t o = (size_t)m * hk + (size_t)g * 16;
      _mm_storeu_si128((__m128i*)(A.ae.data() + o), _mm256_castsi256_si128(pk));
      _mm_storeu_si128((__m128i*)(A.ao.data() + o), _mm256_extracti128_si256(pk, 1));
      // sum of 32 int8 = sum of (a + 128) - 4096
      const __m256i sad = _mm256_sad_epu8(_mm256_xor_si256(x, sx), zero);
      const __m128i s2 = _mm_add_epi64(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
      const long long sum = (long long)_mm_cvtsi128_si64(s2) + (long long)_mm_extract_epi64(s2, 1) - 4096;
      A.cb[(size_t)m * G + g] = (int32_t)(12 * sum);
    }
}

RwAct& rw_tl() {
  static thread_local RwAct a;
  return a;
}

RW_INLINE __m256i rw_ld(const void* p, const bool two) {
  return two ? _mm256_loadu_si256((const __m256i*)p) : _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)p));
}

// reduction 2: vpmaddubsw with the 32 signed activation bytes at p as its memory operand (unaligned is fine for VEX),
// = _mm256_maddubs_epi16(w, loadu(p)); written as asm so the compiler cannot keep the activations in registers
RW_INLINE __m256i rw_maddubs_m(__m256i w, const int8_t* p) {
  __m256i r;
  __asm__("vpmaddubsw {%2, %1, %0|%0, %1, %2}" : "=x"(r) : "x"(w), "m"(*(const __m256i_u*)p));
  return r;
}

// the largest shift spread with G * 49,152 * 2^spread <= 2^31 - 1 (-1 if none): rw_down_rows' loop, once a call
inline int rw_spread(int G) {
  int spread = -1;
  while (spread < 30 && 49152ll * G * (1ll << (spread + 1)) <= 0x7FFFFFFFll) spread++;
  return spread;
}

// One panel (8 weight rows at wp, stride rs), blocks g and g + 1 (two == false: block g only), R activation rows from
// m0. id0[r] / id1[r]: the int32 block dots, lane c = column c.
template <int R, int RED>
RW_INLINE void rw_pair(const uint8_t* wp, size_t rs, const bool two, const RwAct& A, size_t m0, int g, __m256i* id0,
                       __m256i* id1) {
  const __m256i m4 = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
  const __m256i lut = _mm256_load_si256((const __m256i*)rw_JB);
  const size_t off = (size_t)g * 16;
  __m256i e[R], o[R];
  for (int r = 0; r < R; r++) {
    const size_t ro = (m0 + r) * (size_t)A.ld + off;
    e[r] = rw_ld(A.ae.data() + ro, two);
    o[r] = rw_ld(A.ao.data() + ro, two);
  }
  __m256i ab[R][2];  // [0]: rows 0-3, [1]: rows 4-7; lanes (block g | block g + 1) x 4 rows
  if constexpr (RED == 2) {
    // pathb-cpu-r1 reduction 2 (ZEN3_OPTIMISATION_r1.md section 2.4/2.5): no vphaddw / vphaddd (microcoded on Zen 3).
    // Per weight row: m = madd(s, ones) = 4 int32 a block (sums of int16 pairs 0+1, 2+3, 4+5, 6+7). For rows a, b:
    // unpacklo32(ma, mb) + unpackhi32(ma, mb) = [a0+a2, b0+b2, a1+a3, b1+b3]; for the two row pairs of a quad,
    // unpacklo64 + unpackhi64 = [a, b, c, d]: the quad's block sums, the lanes order 1's hadd32 gives. Every value is
    // the same exact int32 (a biased block sum is at most 24 * 127 * 32 = 97,536); only the order of integer adds
    // differs. With two blocks the activations are memory operands of vpmaddubsw (no e[r], o[r] registers).
    const int8_t* aep[R];
    const int8_t* aop[R];
    for (int r = 0; r < R; r++) {
      const size_t ro = (m0 + r) * (size_t)A.ld + off;
      aep[r] = A.ae.data() + ro;
      aop[r] = A.ao.data() + ro;
    }
#pragma GCC unroll 2
    for (int h = 0; h < 2; h++) {
      __m256i u[R][2];
#pragma GCC unroll 2
      for (int j = 0; j < 2; j++) {
        const uint8_t* p = wp + (size_t)(4 * h + 2 * j) * rs;
        const __m256i x0 = rw_ld(p, two), x1 = rw_ld(p + rs, two);
        const __m256i wl0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(x0, m4));
        const __m256i wh0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x0, 4), m4));
        const __m256i wl1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(x1, m4));
        const __m256i wh1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x1, 4), m4));
#pragma GCC unroll 4
        for (int r = 0; r < R; r++) {
          __m256i s0, s1;
          if (two) {
            s0 = _mm256_add_epi16(rw_maddubs_m(wl0, aep[r]), rw_maddubs_m(wh0, aop[r]));
            s1 = _mm256_add_epi16(rw_maddubs_m(wl1, aep[r]), rw_maddubs_m(wh1, aop[r]));
          } else {
            s0 = _mm256_add_epi16(_mm256_maddubs_epi16(wl0, e[r]), _mm256_maddubs_epi16(wh0, o[r]));
            s1 = _mm256_add_epi16(_mm256_maddubs_epi16(wl1, e[r]), _mm256_maddubs_epi16(wh1, o[r]));
          }
          const __m256i ma = _mm256_madd_epi16(s0, ones), mb = _mm256_madd_epi16(s1, ones);
          u[r][j] = _mm256_add_epi32(_mm256_unpacklo_epi32(ma, mb), _mm256_unpackhi_epi32(ma, mb));
        }
      }
#pragma GCC unroll 4
      for (int r = 0; r < R; r++)
        ab[r][h] = _mm256_add_epi32(_mm256_unpacklo_epi64(u[r][0], u[r][1]), _mm256_unpackhi_epi64(u[r][0], u[r][1]));
    }
  } else {
#if V3_RW_ORDER == 1
#pragma GCC unroll 2
  for (int h = 0; h < 2; h++) {
    __m256i m[R][2];
#pragma GCC unroll 2
    for (int j = 0; j < 2; j++) {
      const uint8_t* p = wp + (size_t)(4 * h + 2 * j) * rs;
      const __m256i x0 = rw_ld(p, two), x1 = rw_ld(p + rs, two);
      const __m256i wl0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(x0, m4));
      const __m256i wh0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x0, 4), m4));
      const __m256i wl1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(x1, m4));
      const __m256i wh1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x1, 4), m4));
#pragma GCC unroll 4
      for (int r = 0; r < R; r++) {
        const __m256i s0 = _mm256_add_epi16(_mm256_maddubs_epi16(wl0, e[r]), _mm256_maddubs_epi16(wh0, o[r]));
        const __m256i s1 = _mm256_add_epi16(_mm256_maddubs_epi16(wl1, e[r]), _mm256_maddubs_epi16(wh1, o[r]));
        m[r][j] = _mm256_madd_epi16(_mm256_hadd_epi16(s0, s1), ones);
      }
    }
    for (int r = 0; r < R; r++) ab[r][h] = _mm256_hadd_epi32(m[r][0], m[r][1]);
  }
#else
  __m256i s[R][8];
#pragma GCC unroll 8
  for (int i = 0; i < 8; i++) {
    const __m256i x = rw_ld(wp + (size_t)i * rs, two);
    const __m256i wl = _mm256_shuffle_epi8(lut, _mm256_and_si256(x, m4));                        // even k
    const __m256i wh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), m4));  // odd k
#pragma GCC unroll 4
    for (int r = 0; r < R; r++)
      s[r][i] = _mm256_add_epi16(_mm256_maddubs_epi16(wl, e[r]), _mm256_maddubs_epi16(wh, o[r]));
  }
#pragma GCC unroll 4
  for (int r = 0; r < R; r++) {
    const __m256i m01 = _mm256_madd_epi16(_mm256_hadd_epi16(s[r][0], s[r][1]), ones);
    const __m256i m23 = _mm256_madd_epi16(_mm256_hadd_epi16(s[r][2], s[r][3]), ones);
    const __m256i m45 = _mm256_madd_epi16(_mm256_hadd_epi16(s[r][4], s[r][5]), ones);
    const __m256i m67 = _mm256_madd_epi16(_mm256_hadd_epi16(s[r][6], s[r][7]), ones);
    ab[r][0] = _mm256_hadd_epi32(m01, m23);
    ab[r][1] = _mm256_hadd_epi32(m45, m67);
  }
#endif
  }  // RED != 2
#pragma GCC unroll 4
  for (int r = 0; r < R; r++) {
    const int32_t* cb = A.cb.data() + (m0 + r) * (size_t)A.ldg + g;
    id0[r] = _mm256_sub_epi32(_mm256_permute2x128_si256(ab[r][0], ab[r][1], 0x20), _mm256_set1_epi32(cb[0]));
    if (two) id1[r] = _mm256_sub_epi32(_mm256_permute2x128_si256(ab[r][0], ab[r][1], 0x31), _mm256_set1_epi32(cb[1]));
  }
}

// The served epilogue's fast path alone: valid when every lane's biased exponent t = ea + e8m0 - 1 is in [1, 254].
template <int R>
RW_INLINE void rw_epi_fast(const __m256i* dot, const uint8_t* s8, const int32_t* ea_g, int ldea, __m256* acc) {
  const __m256i ew = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)s8));
#pragma GCC unroll 4
  for (int r = 0; r < R; r++) {
    const __m256i t = _mm256_add_epi32(ew, _mm256_set1_epi32(ea_g[r * ldea] - 1));
    acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot[r]), _mm256_castsi256_ps(_mm256_slli_epi32(t, 23)), acc[r]);
  }
}

// ---------------- FP32 chain (gate / up): one panel, rows [m0, m0 + R), blocks 0..G-1 ascending ----------------
template <int R, int FAST, int RED>
bool rw_fp32_rows(const v3_rm_mat* w, int p, const RwAct& A, size_t m0, const int32_t* ea, int ldea, float* out,
                  int ldo) {
  const int G = w->G;
  const size_t rs = w->row_bytes;
  const uint8_t* base = w->b + (size_t)p * 8 * rs;
  const uint8_t* sp = w->s + (size_t)p * G * 8;
  __m256 acc[R];
  for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_ps();
  __m256i id0[R], id1[R];
  int g = 0;
  for (; g + 1 < G; g += 2) {
    if ((g & 3) == 0)
      for (int i = 0; i < 8; i++) _mm_prefetch((const char*)(base + i * rs + (size_t)g * 16 + 256), _MM_HINT_T0);
    rw_pair<R, RED>(base + (size_t)g * 16, rs, true, A, m0, g, id0, id1);
    if (FAST) {
      rw_epi_fast<R>(id0, sp + (size_t)g * 8, ea + g, ldea, acc);
      rw_epi_fast<R>(id1, sp + (size_t)(g + 1) * 8, ea + g + 1, ldea, acc);
    } else {
      if (!epilogue<R>(id0, sp + (size_t)g * 8, ea + g, ldea, acc)) return false;
      if (!epilogue<R>(id1, sp + (size_t)(g + 1) * 8, ea + g + 1, ldea, acc)) return false;
    }
  }
  if (g < G) {
    rw_pair<R, RED>(base + (size_t)g * 16, rs, false, A, m0, g, id0, id1);
    if (FAST) {
      rw_epi_fast<R>(id0, sp + (size_t)g * 8, ea + g, ldea, acc);
    } else {
      if (!epilogue<R>(id0, sp + (size_t)g * 8, ea + g, ldea, acc)) return false;
    }
  }
  for (int r = 0; r < R; r++) _mm256_storeu_ps(out + (size_t)r * ldo, acc[r]);
  return true;
}

// All M rows of panel p -> out [M][ldo] (8 floats per row), in the served row groups of 4.
template <int FAST, int RED>
bool rw_fp32_panel(const v3_rm_mat* w, int p, const RwAct& A, const int32_t* ea, int ldea, int M, float* out, int ldo) {
  for (int r0 = 0; r0 < M; r0 += 4) {
    const int R = M - r0 < 4 ? M - r0 : 4;
    const int32_t* e = ea + (size_t)r0 * ldea;
    float* o = out + (size_t)r0 * ldo;
    bool ok;
    switch (R) {
      case 1: ok = rw_fp32_rows<1, FAST, RED>(w, p, A, (size_t)r0, e, ldea, o, ldo); break;
      case 2: ok = rw_fp32_rows<2, FAST, RED>(w, p, A, (size_t)r0, e, ldea, o, ldo); break;
      case 3: ok = rw_fp32_rows<3, FAST, RED>(w, p, A, (size_t)r0, e, ldea, o, ldo); break;
      default: ok = rw_fp32_rows<4, FAST, RED>(w, p, A, (size_t)r0, e, ldea, o, ldo); break;
    }
    if (!ok) return false;
  }
  return true;
}

// ---------------- int64 down: one panel, all G blocks, S += ----------------
template <int R, int RED>
bool rw_down_rows(const v3_rm_mat* w, int p, const RwAct& A, size_t m0, const int32_t* ea, int ldea, int bL, int64_t* S,
                  int ldS, int spread_in) {
  const int G = w->G;
  const size_t rs = w->row_bytes;
  const uint8_t* base = w->b + (size_t)p * 8 * rs;
  const uint8_t* sp = w->s + (size_t)p * G * 8;
  __m256i id0[R], id1[R];
  __m256i bad = _mm256_setzero_si256(), flag = _mm256_setzero_si256();
  if (G <= RW_MAXG) {
    __m256i D[R][RW_MAXG];
    int g = 0;
    for (; g + 1 < G; g += 2) {
      rw_pair<R, RED>(base + (size_t)g * 16, rs, true, A, m0, g, id0, id1);
      for (int r = 0; r < R; r++) {
        D[r][g] = id0[r];
        D[r][g + 1] = id1[r];
      }
    }
    if (g < G) {
      rw_pair<R, RED>(base + (size_t)g * 16, rs, false, A, m0, g, id0, id1);
      for (int r = 0; r < R; r++) D[r][g] = id0[r];
    }
    // |dot| <= 32 * 12 * 128 = 49,152: the largest spread with G * 49,152 * 2^spread <= 2^31 - 1 (12 at G = 9)
    int spread = -1;
    if (RED == 2) {   // hoisted: it depends on G only (computed once a call, rw_spread)
      spread = spread_in;
    } else {
      while (spread < 30 && 49152ll * G * (1ll << (spread + 1)) <= 0x7FFFFFFFll) spread++;
    }
    const __m256i zero = _mm256_setzero_si256();
    for (int r = 0; r < R; r++) {
      __m256i sh[RW_MAXG];
      __m256i smin = _mm256_set1_epi32(INT_MAX), smax = _mm256_set1_epi32(INT_MIN);
      for (int gg = 0; gg < G; gg++) {
        const __m256i ew = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(sp + (size_t)gg * 8)));
        sh[gg] = _mm256_add_epi32(ew, _mm256_set1_epi32(ea[(size_t)r * ldea + gg] - 128 - bL));
        smin = _mm256_min_epi32(smin, sh[gg]);
        smax = _mm256_max_epi32(smax, sh[gg]);
      }
      const __m256i viol =
          _mm256_or_si256(_mm256_or_si256(_mm256_cmpgt_epi32(zero, smin), _mm256_cmpgt_epi32(smax, _mm256_set1_epi32(38))),
                          _mm256_cmpgt_epi32(_mm256_sub_epi32(smax, smin), _mm256_set1_epi32(spread)));
      int64_t* Sr = S + (size_t)r * ldS;
      __m256i a0 = _mm256_loadu_si256((const __m256i*)Sr), a1 = _mm256_loadu_si256((const __m256i*)(Sr + 4));
      if (spread >= 0 && _mm256_movemask_epi8(viol) == 0) {
        __m256i acc = zero;
        for (int gg = 0; gg < G; gg++)
          acc = _mm256_add_epi32(acc, _mm256_sllv_epi32(D[r][gg], _mm256_sub_epi32(sh[gg], smin)));
        a0 = _mm256_add_epi64(a0, _mm256_sllv_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc)),
                                                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(smin))));
        a1 = _mm256_add_epi64(a1, _mm256_sllv_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc, 1)),
                                                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256(smin, 1))));
      } else {
        __m256i a2[2] = {a0, a1};
        for (int gg = 0; gg < G; gg++)
          epi_i64<1>(&D[r][gg], sp + (size_t)gg * 8, ea + (size_t)r * ldea + gg, ldea, bL, a2, bad, flag);
        a0 = a2[0];
        a1 = a2[1];
      }
      _mm256_storeu_si256((__m256i*)Sr, a0);
      _mm256_storeu_si256((__m256i*)(Sr + 4), a1);
    }
    return _mm256_movemask_epi8(bad) == 0 && _mm256_movemask_pd(_mm256_castsi256_pd(flag)) == 0;
  }
  __m256i acc[2 * R];
  for (int r = 0; r < R; r++) {
    acc[2 * r] = _mm256_loadu_si256((const __m256i*)(S + (size_t)r * ldS));
    acc[2 * r + 1] = _mm256_loadu_si256((const __m256i*)(S + (size_t)r * ldS + 4));
  }
  int g = 0;
  for (; g + 1 < G; g += 2) {
    rw_pair<R, RED>(base + (size_t)g * 16, rs, true, A, m0, g, id0, id1);
    epi_i64<R>(id0, sp + (size_t)g * 8, ea + g, ldea, bL, acc, bad, flag);
    epi_i64<R>(id1, sp + (size_t)(g + 1) * 8, ea + g + 1, ldea, bL, acc, bad, flag);
  }
  if (g < G) {
    rw_pair<R, RED>(base + (size_t)g * 16, rs, false, A, m0, g, id0, id1);
    epi_i64<R>(id0, sp + (size_t)g * 8, ea + g, ldea, bL, acc, bad, flag);
  }
  for (int r = 0; r < R; r++) {
    _mm256_storeu_si256((__m256i*)(S + (size_t)r * ldS), acc[2 * r]);
    _mm256_storeu_si256((__m256i*)(S + (size_t)r * ldS + 4), acc[2 * r + 1]);
  }
  return _mm256_movemask_epi8(bad) == 0 && _mm256_movemask_pd(_mm256_castsi256_pd(flag)) == 0;
}

template <int RED>
bool rw_down_panel(const v3_rm_mat* w, int p, const RwAct& A, const int32_t* ea, int ldea, int M, int bL, int64_t* S,
                   int ldS, int spread) {
  bool ok = true;
  for (int r0 = 0; r0 < M; r0 += 4) {
    const int R = M - r0 < 4 ? M - r0 : 4;
    const int32_t* e = ea + (size_t)r0 * ldea;
    int64_t* o = S + (size_t)r0 * ldS + (size_t)p * 8;
    bool k;
    switch (R) {
      case 1: k = rw_down_rows<1, RED>(w, p, A, (size_t)r0, e, ldea, bL, o, ldS, spread); break;
      case 2: k = rw_down_rows<2, RED>(w, p, A, (size_t)r0, e, ldea, bL, o, ldS, spread); break;
      case 3: k = rw_down_rows<3, RED>(w, p, A, (size_t)r0, e, ldea, bL, o, ldS, spread); break;
      default: k = rw_down_rows<4, RED>(w, p, A, (size_t)r0, e, ldea, bL, o, ldS, spread); break;
    }
    ok = ok && k;
  }
  return ok;
}

inline int rw_check_mat(const v3_rm_mat* w, int n0, int n1) {
  if (!w || !w->b || !w->s || w->N % 8 || w->K % 32 || w->G != w->K / 32 || w->row_bytes < (size_t)w->K / 2)
    return V3_ERR_SHAPE;
  if (n0 % 8 || n1 % 8 || n0 < 0 || n1 > w->N || n0 > n1) return V3_ERR_SHAPE;
  return V3_OK;
}

// min / max of n code bytes
inline void rw_code_range(const uint8_t* s, int n, int& cmin, int& cmax) {
  __m256i mn = _mm256_set1_epi8((char)0xFF), mx = _mm256_setzero_si256();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i v = _mm256_loadu_si256((const __m256i*)(s + i));
    mn = _mm256_min_epu8(mn, v);
    mx = _mm256_max_epu8(mx, v);
  }
  alignas(32) uint8_t a[32], b[32];
  _mm256_store_si256((__m256i*)a, mn);
  _mm256_store_si256((__m256i*)b, mx);
  int lo = 255, hi = 0;
  for (int k = 0; k < 32; k++) {
    if (a[k] < lo) lo = a[k];
    if (b[k] > hi) hi = b[k];
  }
  for (; i < n; i++) {
    if (s[i] < lo) lo = s[i];
    if (s[i] > hi) hi = s[i];
  }
  cmin = lo;
  cmax = hi;
}

// acc != null: the pre-BF16 accumulators at acc + p * 8 (row stride ldacc); else BF16 at out (row stride ldo).
int rw_gemm(const v3_rm_mat* w, const int8_t* q, int ldq, const int32_t* ea, int ldea, int M, int n0, int n1, float* acc,
            int ldacc, uint16_t* out, int ldo, int red) {
  if (int e = rw_check_mat(w, n0, n1)) return e;
  RwFpScope fp;
  RwAct& A = rw_tl();
  rw_prep(A, q, ldq, M, w->K);
  long long emin = INT_MAX, emax = INT_MIN;
  for (int m = 0; m < M; m++)
    for (int g = 0; g < w->G; g++) {
      const long long v = ea[(size_t)m * ldea + g];
      if (v < emin) emin = v;
      if (v > emax) emax = v;
    }
  static thread_local std::vector<float> tmp;
  if (out && tmp.size() < (size_t)M * 8) tmp.resize((size_t)M * 8);
  for (int p = n0 / 8; p < n1 / 8; p++) {
    int cmin, cmax;
    rw_code_range(w->s + (size_t)p * w->G * 8, w->G * 8, cmin, cmax);
    const bool fast = cmin + emin - 1 >= 1 && cmax + emax - 1 <= 254;   // every t = code + ea - 1 of the panel is normal
    float* dst = out ? tmp.data() : acc + (size_t)p * 8;
    const int ldd = out ? 8 : ldacc;
    const bool ok = red == 2 ? (fast ? rw_fp32_panel<1, 2>(w, p, A, ea, ldea, M, dst, ldd) : rw_fp32_panel<0, 2>(w, p, A, ea, ldea, M, dst, ldd))
                             : (fast ? rw_fp32_panel<1, 1>(w, p, A, ea, ldea, M, dst, ldd) : rw_fp32_panel<0, 1>(w, p, A, ea, ldea, M, dst, ldd));
    if (!ok) return V3_ERR_SCALE_OVERFLOW;
    if (out)
      for (int m = 0; m < M; m++)
        for (int c = 0; c < 8; c++) out[(size_t)m * ldo + p * 8 + c] = bf16_rne(tmp[(size_t)m * 8 + c]);
  }
  return V3_OK;
}

}  // namespace

extern "C" {

int v3_rw_enabled(void) {
  int v = rw_on.load(std::memory_order_relaxed);
  if (v < 0) {
    const char* s = std::getenv("KT_V3_RW");
    v = (s && s[0] == '1' && s[1] == 0) ? 1 : 0;
    rw_on.store(v, std::memory_order_relaxed);
  }
  return v;
}

void v3_rw_set_enabled(int on) { rw_on.store(on ? 1 : 0, std::memory_order_relaxed); }

int v3_rw_red(void) {
  int v = rw_red.load(std::memory_order_relaxed);
  if (v < 0) {
    const char* s = std::getenv("KT_V3_RW_RED");
    v = (s && s[0] == '2' && s[1] == 0) ? 2 : 1;
    rw_red.store(v, std::memory_order_relaxed);
  }
  return v;
}

void v3_rw_set_red(int red) { rw_red.store(red == 2 ? 2 : 1, std::memory_order_relaxed); }

int v3_rw_gemm_acc(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                   int n1, float* acc, int ldacc) {
  if (!v3_rw_enabled() || tier == V3_TIER_C || M <= 0)
    return v3_rm_gemm_acc(w, tier, q, ldq, ea, ldea, M, n0, n1, acc, ldacc);
  return rw_gemm(w, (const int8_t*)q, ldq, ea, ldea, M, n0, n1, acc, ldacc, nullptr, 0, v3_rw_red());
}

int v3_rw_gemm_bf16(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                    int n1, uint16_t* out, int ldo) {
  if (!v3_rw_enabled() || tier == V3_TIER_C || M <= 0)
    return v3_rm_gemm_bf16(w, tier, q, ldq, ea, ldea, M, n0, n1, out, ldo);
  return rw_gemm(w, (const int8_t*)q, ldq, ea, ldea, M, n0, n1, nullptr, 0, out, ldo, v3_rw_red());
}

int v3_rw_down_partial(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M,
                       int bL, const int* blocks, int nb, int n0, int n1, int64_t* S, int ldS) {
  if (!v3_rw_enabled() || tier == V3_TIER_C || blocks != nullptr || M <= 0)
    return v3_rm_down_partial(w, tier, q, ldq, ea, ldea, M, bL, blocks, nb, n0, n1, S, ldS);
  if (int e = rw_check_mat(w, n0, n1)) return e;
  RwFpScope fp;
  RwAct& A = rw_tl();
  rw_prep(A, (const int8_t*)q, ldq, M, w->K);
  int err = V3_OK;
  const int red = v3_rw_red(), spread = rw_spread(w->G);
  for (int p = n0 / 8; p < n1 / 8; p++)
    if (!(red == 2 ? rw_down_panel<2>(w, p, A, ea, ldea, M, bL, S, ldS, spread)
                   : rw_down_panel<1>(w, p, A, ea, ldea, M, bL, S, ldS, spread)))
      err = V3_ERR_DOWN_OVERFLOW;
  return err;
}

}  // extern "C"
