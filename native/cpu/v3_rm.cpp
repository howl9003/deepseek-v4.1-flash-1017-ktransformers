// Contract v3 CPU kernels, arena / row-major input mode. See v3_rm.h. No OpenMP.
// Build: -O3 -mavx2 -mfma -ffp-contract=off -fno-fast-math (also when linked into a -ffast-math project).
#include "v3_rm.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
#include "v3_common.inc"

struct FpScope {  // RNE, FTZ = DAZ = 0 for this call; the caller's state is restored
  unsigned prev;
  FpScope() : prev(_mm_getcsr()) { _mm_setcsr(prev & ~(0x8040u | 0x6000u)); }
  ~FpScope() { _mm_setcsr(prev); }
};

// ---------------- decode: 8 checkpoint rows x one 32-block, two blocks at a time (one per 128-bit lane) ----------------

// v[t] lane L = block (L ? p1 : p0) bytes 2t (rows 0-7, low 8 bytes) and 2t+1 (rows 0-7, high 8 bytes).
inline void load_rows(const uint8_t* p0, const uint8_t* p1, size_t rs, __m256i r[8]) {
  if (p1 == p0 + 16) {  // consecutive blocks: one 32-byte load per row
    for (int i = 0; i < 8; i++) r[i] = _mm256_loadu_si256((const __m256i*)(p0 + i * rs));
  } else {
    for (int i = 0; i < 8; i++)
      r[i] = _mm256_loadu2_m128i((const __m128i*)(p1 + i * rs), (const __m128i*)(p0 + i * rs));
  }
}

inline void transpose_rows(const uint8_t* p0, const uint8_t* p1, size_t rs, __m256i v[8]) {
  __m256i r[8];
  load_rows(p0, p1, rs, r);
  const __m256i t0 = _mm256_unpacklo_epi8(r[0], r[1]), t1 = _mm256_unpackhi_epi8(r[0], r[1]);
  const __m256i t2 = _mm256_unpacklo_epi8(r[2], r[3]), t3 = _mm256_unpackhi_epi8(r[2], r[3]);
  const __m256i t4 = _mm256_unpacklo_epi8(r[4], r[5]), t5 = _mm256_unpackhi_epi8(r[4], r[5]);
  const __m256i t6 = _mm256_unpacklo_epi8(r[6], r[7]), t7 = _mm256_unpackhi_epi8(r[6], r[7]);
  const __m256i u0 = _mm256_unpacklo_epi16(t0, t2), u1 = _mm256_unpackhi_epi16(t0, t2);
  const __m256i u2 = _mm256_unpacklo_epi16(t1, t3), u3 = _mm256_unpackhi_epi16(t1, t3);
  const __m256i u4 = _mm256_unpacklo_epi16(t4, t6), u5 = _mm256_unpackhi_epi16(t4, t6);
  const __m256i u6 = _mm256_unpacklo_epi16(t5, t7), u7 = _mm256_unpackhi_epi16(t5, t7);
  v[0] = _mm256_unpacklo_epi32(u0, u4); v[1] = _mm256_unpackhi_epi32(u0, u4);
  v[2] = _mm256_unpacklo_epi32(u1, u5); v[3] = _mm256_unpackhi_epi32(u1, u5);
  v[4] = _mm256_unpacklo_epi32(u2, u6); v[5] = _mm256_unpackhi_epi32(u2, u6);
  v[6] = _mm256_unpacklo_epi32(u3, u7); v[7] = _mm256_unpackhi_epi32(u3, u7);
}

// x[j] lane L = rows 0-7 of block (L ? p1 : p0), 16-bit word j (bytes 2j, 2j+1) of each row: an 8 x 8 word transpose.
inline void transpose_words(const uint8_t* p0, const uint8_t* p1, size_t rs, __m256i x[8]) {
  __m256i r[8];
  load_rows(p0, p1, rs, r);
  const __m256i a0 = _mm256_unpacklo_epi16(r[0], r[1]), a1 = _mm256_unpackhi_epi16(r[0], r[1]);
  const __m256i a2 = _mm256_unpacklo_epi16(r[2], r[3]), a3 = _mm256_unpackhi_epi16(r[2], r[3]);
  const __m256i a4 = _mm256_unpacklo_epi16(r[4], r[5]), a5 = _mm256_unpackhi_epi16(r[4], r[5]);
  const __m256i a6 = _mm256_unpacklo_epi16(r[6], r[7]), a7 = _mm256_unpackhi_epi16(r[6], r[7]);
  const __m256i b0 = _mm256_unpacklo_epi32(a0, a2), b1 = _mm256_unpackhi_epi32(a0, a2);
  const __m256i b2 = _mm256_unpacklo_epi32(a1, a3), b3 = _mm256_unpackhi_epi32(a1, a3);
  const __m256i b4 = _mm256_unpacklo_epi32(a4, a6), b5 = _mm256_unpackhi_epi32(a4, a6);
  const __m256i b6 = _mm256_unpacklo_epi32(a5, a7), b7 = _mm256_unpackhi_epi32(a5, a7);
  x[0] = _mm256_unpacklo_epi64(b0, b4); x[1] = _mm256_unpackhi_epi64(b0, b4);
  x[2] = _mm256_unpacklo_epi64(b1, b5); x[3] = _mm256_unpackhi_epi64(b1, b5);
  x[4] = _mm256_unpacklo_epi64(b2, b6); x[5] = _mm256_unpackhi_epi64(b2, b6);
  x[6] = _mm256_unpacklo_epi64(b3, b7); x[7] = _mm256_unpackhi_epi64(b3, b7);
}

// Tier D decoded block: J[8] (quad j: lane c = column c's k 4j..4j+3, int8) then |J|[8].
inline void decode2_D(const uint8_t* p0, const uint8_t* p1, size_t rs, __m256i* d0, __m256i* d1) {
  const __m256i lut = _mm256_load_si256((const __m256i*)kJ), m4 = _mm256_set1_epi8(0x0F);
  __m256i xs[8];
  transpose_words(p0, p1, rs, xs);
#pragma GCC unroll 8
  for (int j = 0; j < 8; j++) {
    const __m256i x = xs[j];  // rows' (b2j, b2j+1) interleaved
    const __m256i lo = _mm256_and_si256(x, m4), hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), m4);
    const __m256i y0 = _mm256_unpacklo_epi8(lo, hi), y1 = _mm256_unpackhi_epi8(lo, hi);  // rows 0-3 | rows 4-7
    const __m256i q0 = _mm256_shuffle_epi8(lut, _mm256_permute2x128_si256(y0, y1, 0x20));
    d0[j] = q0;
    d0[8 + j] = _mm256_abs_epi8(q0);
    if (d1) {
      const __m256i q1 = _mm256_shuffle_epi8(lut, _mm256_permute2x128_si256(y0, y1, 0x31));
      d1[j] = q1;
      d1[8 + j] = _mm256_abs_epi8(q1);
    }
  }
}

// Tier C decoded block: 16 int16 vectors; [2w] = columns 0-3, [2w+1] = columns 4-7, each column's k 4w..4w+3
// (pairs (4w, 4w+1), (4w+2, 4w+3) in adjacent 32-bit lanes), from the 8 x 8 word transpose.
inline void decode2_C(const uint8_t* p0, const uint8_t* p1, size_t rs, __m256i* d0, __m256i* d1) {
  const __m256i lut = _mm256_load_si256((const __m256i*)kJ), m4 = _mm256_set1_epi8(0x0F);
  __m256i xs[8];
  transpose_words(p0, p1, rs, xs);
#pragma GCC unroll 8
  for (int w = 0; w < 8; w++) {
    const __m256i lo = _mm256_and_si256(xs[w], m4), hi = _mm256_and_si256(_mm256_srli_epi16(xs[w], 4), m4);
    const __m256i y0 = _mm256_shuffle_epi8(lut, _mm256_unpacklo_epi8(lo, hi));  // rows 0-3 x 4 k (lane = block)
    const __m256i y1 = _mm256_shuffle_epi8(lut, _mm256_unpackhi_epi8(lo, hi));  // rows 4-7
    d0[2 * w] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(y0));
    d0[2 * w + 1] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(y1));
    if (d1) {
      d1[2 * w] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(y0, 1));
      d1[2 * w + 1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(y1, 1));
    }
  }
}

template <int TIER>
inline void decode2(const uint8_t* p0, const uint8_t* p1, size_t rs, __m256i* d0, __m256i* d1) {
  if (TIER == V3_TIER_C) decode2_C(p0, p1 ? p1 : p0, rs, d0, p1 ? d1 : nullptr);
  else decode2_D(p0, p1 ? p1 : p0, rs, d0, p1 ? d1 : nullptr);
}

// Integer block dots for R rows from a decoded block. qb: row 0's activations at this block.
template <int R, int TIER>
inline void rows_dot(const __m256i* dec, const void* qb, int ldq, __m256i* id) {
  if (TIER == V3_TIER_C) {
    const int16_t* q = (const int16_t*)qb;
    __m256i ia[R], ib[R];
    for (int r = 0; r < R; r++) ia[r] = ib[r] = _mm256_setzero_si256();
#pragma GCC unroll 8
    for (int w = 0; w < 8; w++) {
#pragma GCC unroll 4
      for (int r = 0; r < R; r++) {
        int64_t a;
        std::memcpy(&a, q + (size_t)r * ldq + 4 * w, 8);
        const __m256i av = _mm256_set1_epi64x(a);
        ia[r] = _mm256_add_epi32(ia[r], _mm256_madd_epi16(dec[2 * w], av));
        ib[r] = _mm256_add_epi32(ib[r], _mm256_madd_epi16(dec[2 * w + 1], av));
      }
    }
    // hadd: [c0 c1 c4 c5 | c2 c3 c6 c7] -> column order
    const __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (int r = 0; r < R; r++) id[r] = _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(ia[r], ib[r]), perm);
  } else {
    const int8_t* q = (const int8_t*)qb;
    __m256i s16[R];
    for (int r = 0; r < R; r++) s16[r] = _mm256_setzero_si256();
#pragma GCC unroll 8
    for (int j = 0; j < 8; j++) {
#pragma GCC unroll 4
      for (int r = 0; r < R; r++) {
        int32_t a;
        std::memcpy(&a, q + (size_t)r * ldq + 4 * j, 4);
        s16[r] = _mm256_add_epi16(s16[r], _mm256_maddubs_epi16(dec[8 + j], _mm256_sign_epi8(_mm256_set1_epi32(a), dec[j])));
      }
    }
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < R; r++) id[r] = _mm256_madd_epi16(s16[r], ones);
  }
}

inline const void* qrow(int tier, const void* q, size_t off) {
  return tier == V3_TIER_C ? (const void*)((const int16_t*)q + off) : (const void*)((const int8_t*)q + off);
}

// The panel's blocks in processing order (list or 0..G-1).
struct BlockList {
  const int* list;
  int n;
  int at(int i) const { return list ? list[i] : i; }
};

// ---------------- FP32 chain (gate / up): one panel, rows [r0, r0+R), blocks 0..G-1 ascending ----------------

template <int R, int TIER>
bool fp32_rows_fused(const v3_rm_mat* w, int p, const void* q, int ldq, const int32_t* ea, int ldea, float* out, int ldo) {
  const int G = w->G;
  const size_t rs = w->row_bytes;
  const uint8_t* base = w->b + (size_t)p * 8 * rs;
  const uint8_t* sp = w->s + (size_t)p * G * 8;
  __m256 acc[R];
  for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_ps();
  alignas(32) __m256i d0[16], d1[16];
  for (int g = 0; g < G; g += 2) {
    const bool two = g + 1 < G;
    if ((g & 3) == 0)
      for (int i = 0; i < 8; i++) _mm_prefetch((const char*)(base + i * rs + (size_t)g * 16 + 256), _MM_HINT_T0);
    decode2<TIER>(base + (size_t)g * 16, two ? base + (size_t)(g + 1) * 16 : nullptr, rs, d0, d1);
    __m256i id[R];
    rows_dot<R, TIER>(d0, qrow(TIER, q, (size_t)g * 32), ldq, id);
    if (!epilogue<R>(id, sp + (size_t)g * 8, ea + g, ldea, acc)) return false;
    if (two) {
      rows_dot<R, TIER>(d1, qrow(TIER, q, (size_t)(g + 1) * 32), ldq, id);
      if (!epilogue<R>(id, sp + (size_t)(g + 1) * 8, ea + g + 1, ldea, acc)) return false;
    }
  }
  for (int r = 0; r < R; r++) _mm256_storeu_ps(out + (size_t)r * ldo, acc[r]);
  return true;
}

template <int R, int TIER>
bool fp32_rows_buffered(const __m256i* dec, const v3_rm_mat* w, int p, const void* q, int ldq, const int32_t* ea,
                        int ldea, float* out, int ldo) {
  const int G = w->G;
  const uint8_t* sp = w->s + (size_t)p * G * 8;
  __m256 acc[R];
  for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_ps();
  for (int g = 0; g < G; g++) {
    __m256i id[R];
    rows_dot<R, TIER>(dec + (size_t)g * 16, qrow(TIER, q, (size_t)g * 32), ldq, id);
    if (!epilogue<R>(id, sp + (size_t)g * 8, ea + g, ldea, acc)) return false;
  }
  for (int r = 0; r < R; r++) _mm256_storeu_ps(out + (size_t)r * ldo, acc[r]);
  return true;
}

template <int TIER>
void decode_panel(const v3_rm_mat* w, int p, BlockList bl, std::vector<__m256i>& buf) {
  const size_t rs = w->row_bytes;
  const uint8_t* base = w->b + (size_t)p * 8 * rs;
  buf.resize((size_t)bl.n * 16);
  for (int i = 0; i < bl.n; i += 2) {
    const bool two = i + 1 < bl.n;
    decode2<TIER>(base + (size_t)bl.at(i) * 16, two ? base + (size_t)bl.at(i + 1) * 16 : nullptr, rs,
                  buf.data() + (size_t)i * 16, two ? buf.data() + (size_t)(i + 1) * 16 : nullptr);
  }
}

// All M rows of panel p -> out [M][ldo] (8 floats per row at out + r*ldo).
template <int TIER>
bool fp32_panel(const v3_rm_mat* w, int p, const void* q, int ldq, const int32_t* ea, int ldea, int M, float* out, int ldo) {
  if (M <= 4) {
    switch (M) {
      case 1: return fp32_rows_fused<1, TIER>(w, p, q, ldq, ea, ldea, out, ldo);
      case 2: return fp32_rows_fused<2, TIER>(w, p, q, ldq, ea, ldea, out, ldo);
      case 3: return fp32_rows_fused<3, TIER>(w, p, q, ldq, ea, ldea, out, ldo);
      default: return fp32_rows_fused<4, TIER>(w, p, q, ldq, ea, ldea, out, ldo);
    }
  }
  static thread_local std::vector<__m256i> buf;
  decode_panel<TIER>(w, p, BlockList{nullptr, w->G}, buf);
  for (int r0 = 0; r0 < M; r0 += 4) {
    const int R = M - r0 < 4 ? M - r0 : 4;
    const void* qq = qrow(TIER, q, (size_t)r0 * ldq);
    const int32_t* e = ea + (size_t)r0 * ldea;
    float* o = out + (size_t)r0 * ldo;
    bool ok;
    switch (R) {
      case 1: ok = fp32_rows_buffered<1, TIER>(buf.data(), w, p, qq, ldq, e, ldea, o, ldo); break;
      case 2: ok = fp32_rows_buffered<2, TIER>(buf.data(), w, p, qq, ldq, e, ldea, o, ldo); break;
      case 3: ok = fp32_rows_buffered<3, TIER>(buf.data(), w, p, qq, ldq, e, ldea, o, ldo); break;
      default: ok = fp32_rows_buffered<4, TIER>(buf.data(), w, p, qq, ldq, e, ldea, o, ldo); break;
    }
    if (!ok) return false;
  }
  return true;
}

// ---------------- int64 down: one panel, listed blocks, S += ----------------

template <int R, int TIER>
bool down_rows_any(const __m256i* dec /* null: fused decode */, const v3_rm_mat* w, int p, BlockList bl, const void* q,
                   int ldq, const int32_t* ea, int ldea, int bL, int64_t* S, int ldS) {
  const int G = w->G;
  const size_t rs = w->row_bytes;
  const uint8_t* base = w->b + (size_t)p * 8 * rs;
  const uint8_t* sp = w->s + (size_t)p * G * 8;
  __m256i acc[2 * R];
  for (int r = 0; r < R; r++) {
    acc[2 * r] = _mm256_loadu_si256((const __m256i*)(S + (size_t)r * ldS));
    acc[2 * r + 1] = _mm256_loadu_si256((const __m256i*)(S + (size_t)r * ldS + 4));
  }
  __m256i bad = _mm256_setzero_si256(), flag = _mm256_setzero_si256();
  alignas(32) __m256i d0[16], d1[16];
  for (int i = 0; i < bl.n; i++) {
    const int g = bl.at(i);
    const __m256i* d;
    if (dec) {
      d = dec + (size_t)i * 16;
    } else {
      if ((i & 1) == 0) {
        const bool two = i + 1 < bl.n;
        decode2<TIER>(base + (size_t)g * 16, two ? base + (size_t)bl.at(i + 1) * 16 : nullptr, rs, d0, d1);
        d = d0;
      } else {
        d = d1;
      }
    }
    if (g < 0 || g >= G) return false;
    __m256i id[R];
    rows_dot<R, TIER>(d, qrow(TIER, q, (size_t)g * 32), ldq, id);
    epi_i64<R>(id, sp + (size_t)g * 8, ea + g, ldea, bL, acc, bad, flag);
  }
  for (int r = 0; r < R; r++) {
    _mm256_storeu_si256((__m256i*)(S + (size_t)r * ldS), acc[2 * r]);
    _mm256_storeu_si256((__m256i*)(S + (size_t)r * ldS + 4), acc[2 * r + 1]);
  }
  return _mm256_movemask_epi8(bad) == 0 && _mm256_movemask_pd(_mm256_castsi256_pd(flag)) == 0;
}

template <int TIER>
bool down_panel(const v3_rm_mat* w, int p, BlockList bl, const void* q, int ldq, const int32_t* ea, int ldea, int M,
                int bL, int64_t* S, int ldS) {
  const __m256i* dec = nullptr;
  static thread_local std::vector<__m256i> buf;
  if (M > 4) {
    decode_panel<TIER>(w, p, bl, buf);
    dec = buf.data();
  }
  bool ok = true;
  for (int r0 = 0; r0 < M; r0 += 4) {
    const int R = M - r0 < 4 ? M - r0 : 4;
    const void* qq = qrow(TIER, q, (size_t)r0 * ldq);
    const int32_t* e = ea + (size_t)r0 * ldea;
    int64_t* o = S + (size_t)r0 * ldS + (size_t)p * 8;
    bool k;
    switch (R) {
      case 1: k = down_rows_any<1, TIER>(dec, w, p, bl, qq, ldq, e, ldea, bL, o, ldS); break;
      case 2: k = down_rows_any<2, TIER>(dec, w, p, bl, qq, ldq, e, ldea, bL, o, ldS); break;
      case 3: k = down_rows_any<3, TIER>(dec, w, p, bl, qq, ldq, e, ldea, bL, o, ldS); break;
      default: k = down_rows_any<4, TIER>(dec, w, p, bl, qq, ldq, e, ldea, bL, o, ldS); break;
    }
    ok = ok && k;
  }
  return ok;
}

inline int check_mat(const v3_rm_mat* w, int n0, int n1) {
  if (!w || !w->b || !w->s || w->N % 8 || w->K % 32 || w->G != w->K / 32 || w->row_bytes < (size_t)w->K / 2)
    return V3_ERR_SHAPE;
  if (n0 % 8 || n1 % 8 || n0 < 0 || n1 > w->N || n0 > n1) return V3_ERR_SHAPE;
  return V3_OK;
}

}  // namespace

extern "C" {

int v3_rm_scales_to_codes(const float* d, int N, int G, uint8_t* s) {
  if (N % 8 || G <= 0) return V3_ERR_SHAPE;
  for (int n = 0; n < N; n++)
    for (int g = 0; g < G; g++) {
      uint32_t bits;
      std::memcpy(&bits, d + (size_t)n * G + g, 4);
      int code;
      if (bits == 0x00400000u) code = 0;                                       // 2^-127 (FP32 subnormal)
      else if ((bits & 0x807FFFFFu) == 0 && (bits >> 23) >= 1 && (bits >> 23) <= 254) code = (int)(bits >> 23);
      else return V3_ERR_SCALE_FORMAT;                                          // sign, mantissa, zero, inf / NaN
      s[((size_t)(n / 8) * G + g) * 8 + (n % 8)] = (uint8_t)code;
    }
  return V3_OK;
}

int v3_rm_e8m0_to_codes(const uint8_t* e8m0, int N, int G, uint8_t* s) {
  if (N % 8 || G <= 0) return V3_ERR_SHAPE;
  for (int n = 0; n < N; n++)
    for (int g = 0; g < G; g++) {
      const uint8_t c = e8m0[(size_t)n * G + g];
      if (c == 255) return V3_ERR_SCALE_FORMAT;
      s[((size_t)(n / 8) * G + g) * 8 + (n % 8)] = c;
    }
  return V3_OK;
}

int v3_quantise_bf16(const uint16_t* x, int M, int K, int ldx, int tier, void* q, int ldq, int32_t* ea, int ldea) {
  if (K % 32 || (tier != V3_TIER_C && tier != V3_TIER_D)) return V3_ERR_SHAPE;
  FpScope fp;
  const int G = K / 32, qmax = tier == V3_TIER_C ? 32767 : 127;
  alignas(32) float xf[32];
  alignas(32) int32_t qi[32];
  for (int m = 0; m < M; m++)
    for (int g = 0; g < G; g++) {
      const uint16_t* s = x + (size_t)m * ldx + g * 32;
      for (int t = 0; t < 32; t++) xf[t] = bf2f(s[t]);
      ea[(size_t)m * ldea + g] = quant_block(xf, qmax, qi);
      if (tier == V3_TIER_C) {
        int16_t* dq = (int16_t*)q + (size_t)m * ldq + g * 32;
        for (int t = 0; t < 32; t++) dq[t] = (int16_t)qi[t];
      } else {
        int8_t* dq = (int8_t*)q + (size_t)m * ldq + g * 32;
        for (int t = 0; t < 32; t++) dq[t] = (int8_t)qi[t];
      }
    }
  return V3_OK;
}

int v3_rm_gemm_acc(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                   int n1, float* acc, int ldacc) {
  if (int e = check_mat(w, n0, n1)) return e;
  FpScope fp;
  for (int p = n0 / 8; p < n1 / 8; p++) {
    const bool ok = tier == V3_TIER_C ? fp32_panel<V3_TIER_C>(w, p, q, ldq, ea, ldea, M, acc + (size_t)p * 8, ldacc)
                                      : fp32_panel<V3_TIER_D>(w, p, q, ldq, ea, ldea, M, acc + (size_t)p * 8, ldacc);
    if (!ok) return V3_ERR_SCALE_OVERFLOW;
  }
  return V3_OK;
}

int v3_rm_gemm_bf16(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                    int n1, uint16_t* out, int ldo) {
  if (int e = check_mat(w, n0, n1)) return e;
  FpScope fp;
  static thread_local std::vector<float> tmp;
  tmp.resize((size_t)M * 8);
  for (int p = n0 / 8; p < n1 / 8; p++) {
    const bool ok = tier == V3_TIER_C ? fp32_panel<V3_TIER_C>(w, p, q, ldq, ea, ldea, M, tmp.data(), 8)
                                      : fp32_panel<V3_TIER_D>(w, p, q, ldq, ea, ldea, M, tmp.data(), 8);
    if (!ok) return V3_ERR_SCALE_OVERFLOW;
    for (int m = 0; m < M; m++)
      for (int c = 0; c < 8; c++) out[(size_t)m * ldo + p * 8 + c] = bf16_rne(tmp[(size_t)m * 8 + c]);
  }
  return V3_OK;
}

int v3_swiglu_quant(const uint16_t* gate, const uint16_t* up, int ldgu, int M, const float* w_route, const float* silu,
                    int g0, int g1, int tier, uint16_t* xd, int ldxd, void* q, int ldq, int32_t* ea, int ldea) {
  if (g0 < 0 || g1 < g0 || (tier != V3_TIER_C && tier != V3_TIER_D)) return V3_ERR_SHAPE;
  FpScope fp;
  const int qmax = tier == V3_TIER_C ? 32767 : 127;
  alignas(32) float h[32];
  alignas(32) int32_t qi[32];
  for (int m = 0; m < M; m++) {
    const float wr = w_route[m];
    for (int g = g0; g < g1; g++) {
      const size_t base = (size_t)m * ldgu + g * 32;
      for (int t = 0; t < 32; t++) {
        float gv = bf2f(gate[base + t]);
        float uv = bf2f(up[base + t]);
        gv = gv < 10.0f ? gv : 10.0f;
        uv = uv < -10.0f ? -10.0f : (uv > 10.0f ? 10.0f : uv);
        volatile float p1 = silu[f2u(gv) >> 16] * uv;  // two separate RNE multiplies, no FMA
        const uint16_t hb = bf16_rne(p1 * wr);
        if (xd) xd[(size_t)m * ldxd + g * 32 + t] = hb;
        h[t] = bf2f(hb);
      }
      if (q) {
        ea[(size_t)m * ldea + g] = quant_block(h, qmax, qi);
        if (tier == V3_TIER_C) {
          int16_t* dq = (int16_t*)q + (size_t)m * ldq + g * 32;
          for (int t = 0; t < 32; t++) dq[t] = (int16_t)qi[t];
        } else {
          int8_t* dq = (int8_t*)q + (size_t)m * ldq + g * 32;
          for (int t = 0; t < 32; t++) dq[t] = (int8_t)qi[t];
        }
      }
    }
  }
  return V3_OK;
}

int v3_rm_down_partial(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M,
                       int bL, const int* blocks, int nb, int n0, int n1, int64_t* S, int ldS) {
  if (int e = check_mat(w, n0, n1)) return e;
  BlockList bl{blocks, blocks ? nb : w->G};
  for (int i = 0; i < bl.n; i++)
    if (bl.at(i) < 0 || bl.at(i) >= w->G) return V3_ERR_SHAPE;
  FpScope fp;
  int err = V3_OK;
  for (int p = n0 / 8; p < n1 / 8; p++) {
    const bool ok = tier == V3_TIER_C ? down_panel<V3_TIER_C>(w, p, bl, q, ldq, ea, ldea, M, bL, S, ldS)
                                      : down_panel<V3_TIER_D>(w, p, bl, q, ldq, ea, ldea, M, bL, S, ldS);
    if (!ok) err = V3_ERR_DOWN_OVERFLOW;
  }
  return err;
}

int v3_finish_rows(const int64_t* S, int ldS, int M, int n0, int n1, int bL, uint16_t* y, int ldy, int64_t* comb,
                   int ldc) {
  if (bL < -126 || bL > 127 || n0 < 0 || n1 < n0) return V3_ERR_SHAPE;
  FpScope fp;
  const float scale = u2f((uint32_t)(bL + 127) << 23);
  const int64_t lim = (int64_t)1 << 62;
  int err = V3_OK;
  for (int m = 0; m < M; m++)
    for (int c = n0; c < n1; c++) {
      const int64_t s = S[(size_t)m * ldS + c];
      if (s >= lim || s <= -lim) err = V3_ERR_DOWN_OVERFLOW;
      volatile float f = (float)s;  // cvtsi2ss (64-bit): one RNE rounding
      const uint16_t yb = bf16_rne(f * scale);
      if (y) y[(size_t)m * ldy + c] = yb;
      if (comb) comb[(size_t)m * ldc + c] = (int64_t)((uint64_t)comb[(size_t)m * ldc + c] + (uint64_t)fixed_one(yb, bL, err));
    }
  return err;
}

}  // extern "C"
