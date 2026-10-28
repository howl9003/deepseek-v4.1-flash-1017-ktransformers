// Generated from contract-v2-r1/native_v2.cpp (the prototype kernels, bit-exact to contract_v2.py): exports made
// static for embedding in the kt-kernel operator. Include inside a namespace after <immintrin.h>, <algorithm>,
// <cmath>, <cstdint>, <cstring>, <vector>. Regenerate with contract-v2-r1/fullbuild/gen_kernels_inc.py.
// contract-v2-r1/native_v2.cpp: integer-exact AVX2 kernels for the shared E4M3 x E2M1 contract.
// See contract_v2.py for the arithmetic contract; every routine here must be bit-identical to it.
#if defined(__clang__)
#define V2_UNROLL _Pragma("clang loop unroll(full)")
#else
#define V2_UNROLL _Pragma("GCC unroll 8")
#endif
#ifndef V2_ACC
#define V2_ACC 5  // 5: folded scales, int32 lanes (default); 1: chunk transpose + int64 shifts; 3: FP64 FMA per group, 8 lanes; 4: FP64 FMA per group, 4 lanes
#endif
static constexpr int MARGIN = 20;
#ifndef V2_TRUNC_HOOK
#define V2_TRUNC_HOOK() ((void)0)
#endif
#ifndef V2_KCHUNK
#define V2_KCHUNK 0
#endif
#ifndef V2_PREFETCH
#define V2_PREFETCH 0
#endif
#ifndef V2_DIAG_NOEPI
#define V2_DIAG_NOEPI 0
#endif
#ifndef V2_DECODE_PERM
#define V2_DECODE_PERM 1   // folded digits stored in nibble order (evens, odds): no unpacks in the weight decode
#endif
// floor(x / 2^n) for any n >= 0: C++ leaves a shift by 64 or more undefined, and x86 would mask the count.
static inline int64_t shr_floor(int64_t x, int n) { return n >= 64 ? (x < 0 ? -1 : 0) : (x >> n); }

static inline uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
static inline float frombits(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
static inline float bf(float x) { uint32_t b = bits(x); return frombits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000u); }

// E4M3 RNE encoder with saturation at 448 (as validated in the r1 prototype against the NumPy grid).
static inline uint8_t encode(float x) {
  uint32_t b = bits(x); uint8_t sign = (b >> 24) & 128; float a = std::fabs(x); int q;
  if (a >= 448.f) q = 126;
  else if (a < 0.015625f) q = int(std::nearbyint(a * 512.f));
  else { uint32_t v = bits(a); v += 0x7ffffu + ((v >> 20) & 1); q = int(v >> 20) - 960; }
  return uint8_t(q) | sign;
}
struct Tables {
  uint32_t n[256]; int8_t sign[256];
  Tables() {
    for (int c = 0; c < 256; c++) {
      int e = (c & 127) >> 3, m = c & 7;
      uint32_t v = e == 0 ? uint32_t(m) : (uint32_t(8 + m) << (e - 1));
      if ((c & 127) == 127) v = 0;
      n[c] = v; sign[c] = (c & 128) ? -1 : 1;
    }
  }
};
static const Tables T;

// x: BF16-valued FP32 (m, k). Outputs: codes (m,k) u8, ea (m,k/32) int8, hi/lo (m,k) int16 signed digits
// (|value| = (hi<<9)+lo in units of 2^-9), EA (m) int8 = max ea per row.
static inline __m128i v2_quant_narrow16(__m256i v) {
  return _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v,1));
}

static void v2_quant_avx2(const float* x, int m, int k, uint8_t* codes, int8_t* ea,
                       int16_t* hi, int16_t* lo, int8_t* EA) {
  const int G=k/32;
  const __m256 absmask=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  for (int r=0;r<m;r++) {
    int emax=-128;
    for (int g=0;g<G;g++) {
      const size_t off=(size_t)r*k+(size_t)g*32;
      __m256 v[4],maximum=_mm256_set1_ps(1e-4f);
      for (int j=0;j<4;j++) {
        v[j]=_mm256_loadu_ps(x+off+j*8);
        maximum=_mm256_max_ps(maximum,_mm256_and_ps(v[j],absmask));
      }
      __m128 h=_mm_max_ps(_mm256_castps256_ps128(maximum),_mm256_extractf128_ps(maximum,1));
      h=_mm_max_ps(h,_mm_movehl_ps(h,h));
      h=_mm_max_ss(h,_mm_movehdup_ps(h));
      const uint32_t t=bits(_mm_cvtss_f32(h)*(1.f/448.f));
      const int ex=int((t>>23)&255)-127+((t&0x7fffff)!=0);
      ea[(size_t)r*G+g]=int8_t(ex);emax=std::max(emax,ex);
      // ex is [-22,120] for finite FP32 and the original amax floor.
      // Both scale and its reciprocal are normal, exact powers of two.
      const __m256 inverse=_mm256_set1_ps(std::ldexp(1.f,-ex));
      for (int j=0;j<4;j++) {
        const __m256 z=_mm256_mul_ps(v[j],inverse);
        const __m256 a=_mm256_and_ps(z,absmask);
        const __m256i b=_mm256_castps_si256(a);
        const __m256i normal=_mm256_sub_epi32(_mm256_srli_epi32(
            _mm256_add_epi32(_mm256_add_epi32(b,_mm256_set1_epi32(0x7ffff)),
              _mm256_and_si256(_mm256_srli_epi32(b,20),_mm256_set1_epi32(1))),20),_mm256_set1_epi32(960));
        const __m256 sub=_mm256_mul_ps(a,_mm256_set1_ps(512.f));
        const __m256i rounded=_mm256_cvttps_epi32(_mm256_round_ps(sub,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
        __m256i q=_mm256_blendv_epi8(normal,rounded,_mm256_castps_si256(_mm256_cmp_ps(a,_mm256_set1_ps(.015625f),_CMP_LT_OQ)));
        q=_mm256_blendv_epi8(q,_mm256_set1_epi32(126),_mm256_castps_si256(_mm256_cmp_ps(a,_mm256_set1_ps(448.f),_CMP_GE_OQ)));
        const __m256i zb=_mm256_castps_si256(z);
        const __m256i code=_mm256_or_si256(q,_mm256_and_si256(_mm256_srli_epi32(zb,24),_mm256_set1_epi32(128)));
        _mm_storel_epi64((__m128i*)(codes+off+j*8),_mm_packus_epi16(v2_quant_narrow16(code),_mm_setzero_si128()));
        const __m256i e=_mm256_srli_epi32(q,3),mant=_mm256_and_si256(q,_mm256_set1_epi32(7));
        __m256i mag=_mm256_sllv_epi32(_mm256_add_epi32(mant,_mm256_set1_epi32(8)),_mm256_sub_epi32(e,_mm256_set1_epi32(1)));
        mag=_mm256_blendv_epi8(mag,mant,_mm256_cmpeq_epi32(e,_mm256_setzero_si256()));
        const __m256i sign=_mm256_srai_epi32(zb,31);
        const __m256i high=_mm256_srli_epi32(mag,9),low=_mm256_and_si256(mag,_mm256_set1_epi32(511));
        _mm_storeu_si128((__m128i*)(hi+off+j*8),v2_quant_narrow16(_mm256_sub_epi32(_mm256_xor_si256(high,sign),sign)));
        _mm_storeu_si128((__m128i*)(lo+off+j*8),v2_quant_narrow16(_mm256_sub_epi32(_mm256_xor_si256(low,sign),sign)));
      }
    }
    EA[r]=int8_t(emax);
  }
}


[[maybe_unused]] static void v2_quant(const float* x, int m, int k, uint8_t* codes, int8_t* ea, int16_t* hi, int16_t* lo, int8_t* EA) {
  // Default off. The caller's ScopedFpContract establishes RNE and disables
  // DAZ/FTZ, as required by both scalar and vector quantizers.
  static const bool use_avx2 = [] {
    const char* flag = std::getenv("KT_CONTRACT_V2_QUANT_AVX2");
    return flag != nullptr && std::strcmp(flag, "1") == 0;
  }();
  if (use_avx2) { v2_quant_avx2(x,m,k,codes,ea,hi,lo,EA); return; }

  const int G = k / 32;
  for (int r = 0; r < m; r++) {
    int emax = -128;
    for (int g = 0; g < G; g++) {
      const float* p = x + (size_t)r * k + (size_t)g * 32;
      float am = 1e-4f;
      for (int j = 0; j < 32; j++) am = std::max(am, std::fabs(p[j]));
      uint32_t t = bits(am * (1.f / 448.f));
      int ex = int((t >> 23) & 255) - 127 + ((t & 0x7fffff) != 0);
      float s = std::ldexp(1.f, ex);
      ea[(size_t)r * G + g] = int8_t(ex); emax = std::max(emax, ex);
      for (int j = 0; j < 32; j++) {
        size_t off = (size_t)r * k + (size_t)g * 32 + j;
        uint8_t c = encode(p[j] / s); codes[off] = c;
        int32_t mag = int32_t(T.n[c]), sg = T.sign[c];
        hi[off] = int16_t(sg * (mag >> 9)); lo[off] = int16_t(sg * (mag & 511));
      }
    }
    EA[r] = int8_t(emax);
  }
}
[[maybe_unused]] static void v2_prepare_weights(const uint8_t* ws, int n, int G, int8_t* EW) {
  for (int row = 0; row < n; row++) {
    int mx = 0;
    for (int g = 0; g < G; g++) mx = std::max(mx, int(ws[(size_t)row * G + g]));
    EW[row] = int8_t(mx - 127);
  }
}

static inline int32_t hsum32(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e)); s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
  return _mm_cvtsi128_si32(s);
}
static inline int64_t hsum64(__m256i v) {
  __m128i s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  return _mm_cvtsi128_si64(s) + _mm_extract_epi64(s, 1);
}
// Eight 8-lane vectors -> one vector of their eight lane sums (groups in order), with single-uop shuffles.
static inline __m256i sum8x8(const __m256i* v) {
  __m256i a0 = _mm256_add_epi32(_mm256_unpacklo_epi32(v[0], v[1]), _mm256_unpackhi_epi32(v[0], v[1]));
  __m256i a1 = _mm256_add_epi32(_mm256_unpacklo_epi32(v[2], v[3]), _mm256_unpackhi_epi32(v[2], v[3]));
  __m256i a2 = _mm256_add_epi32(_mm256_unpacklo_epi32(v[4], v[5]), _mm256_unpackhi_epi32(v[4], v[5]));
  __m256i a3 = _mm256_add_epi32(_mm256_unpacklo_epi32(v[6], v[7]), _mm256_unpackhi_epi32(v[6], v[7]));
  __m256i b0 = _mm256_add_epi32(_mm256_unpacklo_epi64(a0, a1), _mm256_unpackhi_epi64(a0, a1));
  __m256i b1 = _mm256_add_epi32(_mm256_unpacklo_epi64(a2, a3), _mm256_unpackhi_epi64(a2, a3));
  return _mm256_add_epi32(_mm256_permute2x128_si256(b0, b1, 0x20), _mm256_permute2x128_si256(b0, b1, 0x31));
}

// Groups [g0, g1) of R token rows against n output rows, accumulated exactly into acc[r*n+row] (int64, +=).
// Per group: two int16 planes x int16 weights -> eight int32 lane partials (|lane| < 2^24). Per chunk of eight
// groups: lane sums -> eight exact int32 group totals (< 2^27) -> int64, shifted by each group's exponent
// relative to the row base (sum_g D_g << sh_g), added into two int64 accumulators; one lane sum per row.
template <int R>
static void rows_R_range(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA, const uint8_t* w,
                         const uint8_t* ws, const int8_t* EW, int64_t* acc, int n, int k, int g0, int g1, int row0, int row1) {
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), mask = _mm_set1_epi8(15);
  const int G = k / 32;
  for (int row = row0; row < row1; row++) {
    const uint8_t* wrow = w + (size_t)row * (k / 2);
    const uint8_t* srow = ws + (size_t)row * G;
    __m256i acc0[R], acc1[R];
    __m256i base[R];
    V2_UNROLL
    for (int r = 0; r < R; r++) { acc0[r] = _mm256_setzero_si256(); acc1[r] = _mm256_setzero_si256(); base[r] = _mm256_set1_epi32(MARGIN - 127 - int(EA[r]) - int(EW[row])); }
    for (int gc = g0; gc < g1; gc += 8) {
      const int cnt = std::min(8, g1 - gc);
      __m256i v[R][8];
      for (int g = gc; g < gc + cnt; g++) {
        __m128i p = _mm_loadu_si128((const __m128i*)(wrow + (size_t)g * 16));
        __m128i ln = _mm_shuffle_epi8(lut, _mm_and_si128(p, mask)), hn = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(p, 4), mask));
        __m256i w0 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi8(ln, hn)), w1 = _mm256_cvtepi8_epi16(_mm_unpackhi_epi8(ln, hn));
        V2_UNROLL
        for (int r = 0; r < R; r++) {
          const int16_t* hr = hi + (size_t)r * k + (size_t)g * 32;
          const int16_t* lr = lo + (size_t)r * k + (size_t)g * 32;
          __m256i a = _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i*)hr)),
                                       _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i*)(hr + 16))));
          __m256i b = _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i*)lr)),
                                       _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i*)(lr + 16))));
          v[r][g - gc] = _mm256_add_epi32(_mm256_slli_epi32(a, 9), b);
        }
      }
      __m128i codes8;
      if (cnt == 8) codes8 = _mm_loadl_epi64((const __m128i*)(srow + gc));
      else { alignas(16) uint8_t cb[8]; std::memset(cb, 127, 8); std::memcpy(cb, srow + gc, cnt); codes8 = _mm_loadl_epi64((const __m128i*)cb); }
      const __m256i c32 = _mm256_cvtepu8_epi32(codes8);
      V2_UNROLL
      for (int r = 0; r < R; r++) {
        for (int g = cnt; g < 8; g++) v[r][g] = _mm256_setzero_si256();
        __m128i ea8;
        if (cnt == 8) ea8 = _mm_loadl_epi64((const __m128i*)(ea + (size_t)r * G + gc));
        else { alignas(16) int8_t eb[8]; std::memset(eb, 0, 8); std::memcpy(eb, ea + (size_t)r * G + gc, cnt); ea8 = _mm_loadl_epi64((const __m128i*)eb); }
        __m256i tot = sum8x8(v[r]);
        __m256i sh = _mm256_add_epi32(_mm256_add_epi32(_mm256_cvtepi8_epi32(ea8), c32), base[r]);
        if (_mm256_movemask_ps(_mm256_castsi256_ps(sh))) {
          // A term more than MARGIN bits below the row's largest: floor-shift it (deterministic, never on real data).
          alignas(32) int32_t tt[8], ss[8];
          _mm256_store_si256((__m256i*)tt, tot); _mm256_store_si256((__m256i*)ss, sh);
          int64_t s = 0;
          // sh <= MARGIN so the left shift is always defined; a right shift by 64 or more is undefined in C++
          // (x86 masks the count: g++ 15 on Linux produced D >> (n mod 64) and broke exactness at weight codes
          // 211..213), so it is clamped to the contract's floor value, 0 or -1.
          V2_TRUNC_HOOK();
          for (int i = 0; i < 8; i++) s += ss[i] >= 0 ? (int64_t(tt[i]) << ss[i]) : shr_floor(int64_t(tt[i]), -ss[i]);
          acc0[r] = _mm256_add_epi64(acc0[r], _mm256_set_epi64x(0, 0, 0, s));
        } else {
          __m128i th = _mm256_extracti128_si256(tot, 1), shh = _mm256_extracti128_si256(sh, 1);
          acc0[r] = _mm256_add_epi64(acc0[r], _mm256_sllv_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(tot)), _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sh))));
          acc1[r] = _mm256_add_epi64(acc1[r], _mm256_sllv_epi64(_mm256_cvtepi32_epi64(th), _mm256_cvtepi32_epi64(shh)));
        }
      }
    }
    V2_UNROLL
    for (int r = 0; r < R; r++) acc[(size_t)r * n + row] += hsum64(_mm256_add_epi64(acc0[r], acc1[r]));
  }
}


// FP64 accumulation (V2_ACC 3 and 4). A group's int32 lane partials (|lane| < 2^25 after folding to four
// lanes) times 2^sh (sh <= MARGIN = 20), summed over <= 256 groups, stay below 2^53 per lane, so every
// FP64 FMA is exact; the lane sums are converted to int64 exactly and merged like the int64 path.
static const struct Pow2Table { double v[64]; Pow2Table() { for (int i = 0; i < 64; i++) v[i] = std::ldexp(1.0, i); } } POW2;
template <int R, bool LANES4>
static void rows_R_fp64(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA, const uint8_t* w,
                        const uint8_t* ws, const int8_t* EW, int64_t* acc, int n, int k, int g0, int g1) {
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), mask = _mm_set1_epi8(15);
  const int G = k / 32;
  for (int row = 0; row < n; row++) {
    const uint8_t* wrow = w + (size_t)row * (k / 2);
    const uint8_t* srow = ws + (size_t)row * G;
    __m256d acc0[R], acc1[R];
    int base[R];
    V2_UNROLL
    for (int r = 0; r < R; r++) { acc0[r] = _mm256_setzero_pd(); acc1[r] = _mm256_setzero_pd(); base[r] = MARGIN - 127 - int(EA[r]) - int(EW[row]); }
    for (int g = g0; g < g1; g++) {
      __m128i p = _mm_loadu_si128((const __m128i*)(wrow + (size_t)g * 16));
      __m128i ln = _mm_shuffle_epi8(lut, _mm_and_si128(p, mask)), hn = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(p, 4), mask));
      __m256i w0 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi8(ln, hn)), w1 = _mm256_cvtepi8_epi16(_mm_unpackhi_epi8(ln, hn));
      const int code = int(srow[g]);
      V2_UNROLL
      for (int r = 0; r < R; r++) {
        const int16_t* hr = hi + (size_t)r * k + (size_t)g * 32;
        const int16_t* lr = lo + (size_t)r * k + (size_t)g * 32;
        __m256i a = _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i*)hr)),
                                     _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i*)(hr + 16))));
        __m256i b = _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i*)lr)),
                                     _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i*)(lr + 16))));
        __m256i v = _mm256_add_epi32(_mm256_slli_epi32(a, 9), b);
        const int sh = int(ea[(size_t)r * G + g]) + code + base[r];
        if (sh >= 0) {
          const __m256d sc = _mm256_broadcast_sd(POW2.v + sh);
          if constexpr (LANES4) {
            __m128i v4 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            acc0[r] = _mm256_fmadd_pd(_mm256_cvtepi32_pd(v4), sc, acc0[r]);
          } else {
            acc0[r] = _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(v)), sc, acc0[r]);
            acc1[r] = _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(v, 1)), sc, acc1[r]);
          }
        } else {
          // A term more than MARGIN bits below the row's largest: floor-shift it (deterministic, never on real data).
          const int64_t t = shr_floor(int64_t(hsum32(v)), -sh);   // clamped: a shift of 64+ is undefined in C++
          acc0[r] = _mm256_add_pd(acc0[r], _mm256_set_pd(0, 0, 0, double(t)));
        }
      }
    }
    V2_UNROLL
    for (int r = 0; r < R; r++) {
      alignas(32) double d[8];
      _mm256_store_pd(d, acc0[r]); _mm256_store_pd(d + 4, acc1[r]);
      int64_t s = 0;
      for (int i = 0; i < 8; i++) s += int64_t(d[i]);   // each lane is an exact integer below 2^53
      acc[(size_t)r * n + row] += s;
    }
  }
}

template <int R>
static void rows_R(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA, const uint8_t* w,
                   const uint8_t* ws, const int8_t* EW, int64_t* acc, int n, int k, int g0, int g1) {
  rows_R_range<R>(hi, lo, ea, EA, w, ws, EW, acc, n, k, g0, g1, 0, n);
}

// ---------------------------------------------------------------------------------------------------
// V2_ACC 5: scales folded into the integers. For a token whose activation exponents span SA <= 6 bits the
// digits are pre-shifted by (ea_g - EAmin): hi_f <= 448<<6, lo_f <= 511<<6, both int16. For a row whose
// weight exponents span SW bits the decoded weights are shifted by (ws_g - EWmin) during decode (w <= 12<<SW,
// int16 for SW <= 11). Then sum_g D_g 2^(ea_g+ew_g) = 2^(EAmin+EWmin) * sum_g D'_g with D'_g the dot of the
// folded operands, and the row is a plain int32 lane accumulation: no per-group shift, no transpose. The
// lane bound 4*511*12 * 2^(SA+SW) * G < 2^31 gives the budget SA + SW <= floor(log2(2^31 / (24528 G))):
// 9 for G = 160, 10 for G = 72; with K-chunking the budget covers one chunk of 128/R groups (11 for R = 4).
// Rows or token tiles outside the budget take the general kernel; both are
// exact, so the choice never changes a bit. Observed spreads: SA 2-6, SW 3-4.
static inline int fold_budget(int G) {
  int b = 0;
  while ((int64_t(24528) * G) << (b + 1) < (int64_t(1) << 31)) b++;
  return b;
}
// Folded digits for every token; SA[t] = EA - EAmin, or 127 when the token cannot be folded (SA > 6).
[[maybe_unused]] static void v2_fold(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA, int m, int k,
                 int16_t* hi_f, int16_t* lo_f, int8_t* EAmin, int8_t* SA) {
  const int G = k / 32;
  for (int t = 0; t < m; t++) {
    int emin = 127;
    for (int g = 0; g < G; g++) emin = std::min(emin, int(ea[(size_t)t * G + g]));
    const int spread = int(EA[t]) - emin;
    EAmin[t] = int8_t(emin);
    if (spread > 6) { SA[t] = 127; continue; }
    SA[t] = int8_t(spread);
    for (int g = 0; g < G; g++) {
      const int s = int(ea[(size_t)t * G + g]) - emin;
      const __m128i cnt = _mm_cvtsi32_si128(s);
#if V2_DECODE_PERM
      // Store the folded digits of a group as [even elements 0,2,..,30 | odd elements 1,3,..,31]: this is the order
      // in which the packed weights decode (low nibbles, then high nibbles), so row_fold needs no unpacks. The dot
      // product is order-independent, so the result is unchanged.
      const __m256i deint = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
      const size_t off = (size_t)t * k + (size_t)g * 32;
      auto perm = [&](const int16_t* src, int16_t* dst) {
        __m256i a = _mm256_sll_epi16(_mm256_loadu_si256((const __m256i*)(src + off)), cnt);
        __m256i b = _mm256_sll_epi16(_mm256_loadu_si256((const __m256i*)(src + off + 16)), cnt);
        a = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(a, deint), 0xD8);   // [evens of 0..15 | odds of 0..15]
        b = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(b, deint), 0xD8);   // [evens of 16..31 | odds of 16..31]
        _mm256_storeu_si256((__m256i*)(dst + off), _mm256_permute2x128_si256(a, b, 0x20));
        _mm256_storeu_si256((__m256i*)(dst + off + 16), _mm256_permute2x128_si256(a, b, 0x31));
      };
      perm(hi, hi_f); perm(lo, lo_f);
#else
      for (int j = 0; j < 32; j += 16) {
        const size_t off = (size_t)t * k + (size_t)g * 32 + j;
        _mm256_storeu_si256((__m256i*)(hi_f + off), _mm256_sll_epi16(_mm256_loadu_si256((const __m256i*)(hi + off)), cnt));
        _mm256_storeu_si256((__m256i*)(lo_f + off), _mm256_sll_epi16(_mm256_loadu_si256((const __m256i*)(lo + off)), cnt));
      }
#endif
    }
  }
}
[[maybe_unused]] static void v2_prepare_weights2(const uint8_t* ws, int n, int G, int8_t* EW, int8_t* EWmin) {
  for (int row = 0; row < n; row++) {
    int mx = 0, mn = 255;
    for (int g = 0; g < G; g++) { const int c = int(ws[(size_t)row * G + g]); mx = std::max(mx, c); mn = std::min(mn, c); }
    EW[row] = int8_t(mx - 127); EWmin[row] = int8_t(mn - 127);
  }
}
// One output row for R folded tokens: int32 lane accumulators, weights shifted at decode.
// Nibble LUTs with the weight value pre-shifted by s = 0..3 (w' << s <= 96 fits int8); for a row whose weight
// exponents span more than 3 bits the decoded int16 weights are shifted instead.
static const struct LutShifted { alignas(16) int8_t v[4][16]; LutShifted() {
  static const int8_t base[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  for (int s = 0; s < 4; s++) for (int i = 0; i < 16; i++) v[s][i] = int8_t(base[i] << s); } } LUT_SHIFTED;
template <int R, bool LUTSEL>
static inline void row_fold(const int16_t* hi_f, const int16_t* lo_f, const int8_t* EAmin, const int8_t* EA, const uint8_t* wrow,
                            const uint8_t* srow, int ewmin, int ew, int64_t* acc_out, int k, int g0, int g1) {
  const __m128i lut0 = _mm_load_si128((const __m128i*)LUT_SHIFTED.v[0]), mask = _mm_set1_epi8(15);
  __m256i ah[R], al[R];
  V2_UNROLL
  for (int r = 0; r < R; r++) { ah[r] = _mm256_setzero_si256(); al[r] = _mm256_setzero_si256(); }
  for (int g = g0; g < g1; g++) {
    __m128i p = _mm_loadu_si128((const __m128i*)(wrow + (size_t)g * 16));
    const int sc = int(srow[g]) - 127 - ewmin;
    __m256i w0, w1;
    if constexpr (LUTSEL) {
      const __m128i lut = _mm_load_si128((const __m128i*)LUT_SHIFTED.v[sc]);
      __m128i ln = _mm_shuffle_epi8(lut, _mm_and_si128(p, mask)), hn = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(p, 4), mask));
#if V2_DECODE_PERM
      w0 = _mm256_cvtepi8_epi16(ln); w1 = _mm256_cvtepi8_epi16(hn);   // even elements, odd elements (digits are stored in this order)
#else
      w0 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi8(ln, hn)); w1 = _mm256_cvtepi8_epi16(_mm_unpackhi_epi8(ln, hn));
#endif
    } else {
      __m128i ln = _mm_shuffle_epi8(lut0, _mm_and_si128(p, mask)), hn = _mm_shuffle_epi8(lut0, _mm_and_si128(_mm_srli_epi16(p, 4), mask));
      const __m128i cnt = _mm_cvtsi32_si128(sc);
#if V2_DECODE_PERM
      w0 = _mm256_sll_epi16(_mm256_cvtepi8_epi16(ln), cnt); w1 = _mm256_sll_epi16(_mm256_cvtepi8_epi16(hn), cnt);
#else
      w0 = _mm256_sll_epi16(_mm256_cvtepi8_epi16(_mm_unpacklo_epi8(ln, hn)), cnt); w1 = _mm256_sll_epi16(_mm256_cvtepi8_epi16(_mm_unpackhi_epi8(ln, hn)), cnt);
#endif
    }
    V2_UNROLL
    for (int r = 0; r < R; r++) {
      const int16_t* hr = hi_f + (size_t)r * k + (size_t)g * 32;
      const int16_t* lr = lo_f + (size_t)r * k + (size_t)g * 32;
#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
      // Zen 4+/Alder Lake+: vpdpwssd fuses the pair multiply-add with the accumulation (same integers, same result)
      ah[r] = _mm256_dpwssd_epi32(_mm256_dpwssd_epi32(ah[r], w0, _mm256_loadu_si256((const __m256i*)hr)), w1, _mm256_loadu_si256((const __m256i*)(hr + 16)));
      al[r] = _mm256_dpwssd_epi32(_mm256_dpwssd_epi32(al[r], w0, _mm256_loadu_si256((const __m256i*)lr)), w1, _mm256_loadu_si256((const __m256i*)(lr + 16)));
#else
      ah[r] = _mm256_add_epi32(ah[r], _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i*)hr)),
                                                       _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i*)(hr + 16)))));
      al[r] = _mm256_add_epi32(al[r], _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i*)lr)),
                                                       _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i*)(lr + 16)))));
#endif
    }
  }
#if V2_DIAG_NOEPI
  // diagnostic build only (wrong results): skip the per-row epilogue to measure its cost
  for (int r = 0; r < R; r++) acc_out[r] += _mm256_extract_epi32(_mm256_add_epi32(ah[r], al[r]), 0);
  return;
#endif
  V2_UNROLL
  for (int r = 0; r < R; r++) {
    // (hi plane << 9) + lo plane, combined per int64 lane first, then one horizontal sum (exact: lanes < 2^31)
    const __m256i lo64 = _mm256_add_epi64(_mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(ah[r])), 9), _mm256_cvtepi32_epi64(_mm256_castsi256_si128(al[r])));
    const __m256i hi64 = _mm256_add_epi64(_mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(ah[r], 1)), 9), _mm256_cvtepi32_epi64(_mm256_extracti128_si256(al[r], 1)));
    const int64_t A1 = hsum64(_mm256_add_epi64(lo64, hi64));
    const int shift = (int(EAmin[r]) + ewmin) - (int(EA[r]) + ew - MARGIN);  // = MARGIN - SA - SW >= 0 inside the budget
    acc_out[r] += A1 << shift;
  }
}
// General kernel restricted to one row (the transpose + int64 path), used outside the fold budget.
template <int R>
static void rows_R_range(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA, const uint8_t* w,
                         const uint8_t* ws, const int8_t* EW, int64_t* acc, int n, int k, int g0, int g1, int row0, int row1);

[[maybe_unused]] static void v2_linear_partial_fold(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA,
                                const int16_t* hi_f, const int16_t* lo_f, const int8_t* EAmin, const int8_t* SA,
                                const uint8_t* w, const uint8_t* ws, const int8_t* EW, const int8_t* EWmin,
                                int64_t* acc, int m, int n, int k, int g0, int g1) {
  const int G = k / 32;
  auto tile = [&](auto rc, int r0) {
    constexpr int R = decltype(rc)::value;
    // K-chunking (Zen 3 tuning, 26 Oct 2026): the row loop runs over GC groups at a time so the R tokens' folded
    // digits for the chunk (R * GC * 128 bytes = 16 KiB) stay in L1. Without it the digits of four tokens
    // (80 KiB) were re-read from L2 for every output row, 128 bytes per token-row-group at ~32 B/cycle, which
    // bound the kernel at batch >= 4 (4.8 cycles per token-row-group against KT's 3.6 on EPYC 7543).
    // Every chunk adds its exact int64 partial into acc, so the result is bit-identical to one pass over K
    // (integer sums commute); the int32 lane budget only has to cover one chunk, so it is wider as well.
    // Measured 26 Oct 2026 on EPYC 7543: the chunked pass turns the weight stream into 512-byte slices per row
    // and loses more to prefetch/DRAM efficiency than it gains from L1 (node bench 41 -> 30 GB/s at batch 1),
    // so chunking is opt-in (-DV2_KCHUNK=1) and the default is one sequential pass over K.
    constexpr int GC = V2_KCHUNK ? 128 / R : (1 << 30);
    const int budget = fold_budget(std::min(GC, g1 - g0));
    int sa = 0;
    for (int r = 0; r < R; r++) sa = std::max(sa, int(SA[r0 + r]));
    const size_t o = (size_t)r0 * k;
    for (int c0 = g0; c0 < g1; c0 += GC) {
      const int c1 = std::min(g1, c0 + GC);
      for (int row = 0; row < n; row++) {
#if V2_PREFETCH > 0
        // chunked pass: the weight stream is a 16*(c1-c0)-byte slice per row, so fetch the slice of a row ahead
        if (row + V2_PREFETCH < n) {
          const uint8_t* nx = w + (size_t)(row + V2_PREFETCH) * (k / 2) + (size_t)c0 * 16;
          for (int b = 0; b < (c1 - c0) * 16; b += 64) _mm_prefetch((const char*)nx + b, _MM_HINT_T0);
        }
#endif
        const int sw = int(EW[row]) - int(EWmin[row]);
        if (sa + sw <= budget) {
          int64_t out[R];
          for (int r = 0; r < R; r++) out[r] = 0;
          if (sw <= 3) row_fold<R, true>(hi_f + o, lo_f + o, EAmin + r0, EA + r0, w + (size_t)row * (k / 2), ws + (size_t)row * G, int(EWmin[row]), int(EW[row]), out, k, c0, c1);
          else row_fold<R, false>(hi_f + o, lo_f + o, EAmin + r0, EA + r0, w + (size_t)row * (k / 2), ws + (size_t)row * G, int(EWmin[row]), int(EW[row]), out, k, c0, c1);
          for (int r = 0; r < R; r++) acc[(size_t)(r0 + r) * n + row] += out[r];
        } else {
          rows_R_range<R>(hi + o, lo + o, ea + (size_t)r0 * G, EA + r0, w, ws, EW, acc + (size_t)r0 * n, n, k, c0, c1, row, row + 1);
        }
      }
    }
  };
  int r = 0;
  for (; r + 4 <= m; r += 4) tile(std::integral_constant<int, 4>{}, r);
  for (; r + 2 <= m; r += 2) tile(std::integral_constant<int, 2>{}, r);
  for (; r < m; r++) tile(std::integral_constant<int, 1>{}, r);
}

[[maybe_unused]] static void v2_linear_partial(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA, const uint8_t* w,
                           const uint8_t* ws, const int8_t* EW, int64_t* acc, int m, int n, int k, int g0, int g1) {
  const int G = k / 32;
#if V2_ACC == 1 || V2_ACC == 5
#define V2_ROWS(R) rows_R<R>
#elif V2_ACC == 3
#define V2_ROWS(R) rows_R_fp64<R, false>
#else
#define V2_ROWS(R) rows_R_fp64<R, true>
#endif
  int r = 0;
  for (; r + 4 <= m; r += 4) V2_ROWS(4)(hi + (size_t)r * k, lo + (size_t)r * k, ea + (size_t)r * G, EA + r, w, ws, EW, acc + (size_t)r * n, n, k, g0, g1);
  for (; r + 2 <= m; r += 2) V2_ROWS(2)(hi + (size_t)r * k, lo + (size_t)r * k, ea + (size_t)r * G, EA + r, w, ws, EW, acc + (size_t)r * n, n, k, g0, g1);
  for (; r < m; r++) V2_ROWS(1)(hi + (size_t)r * k, lo + (size_t)r * k, ea + (size_t)r * G, EA + r, w, ws, EW, acc + (size_t)r * n, n, k, g0, g1);
}
[[maybe_unused]] static int v2_variant() { return V2_ACC; }
[[maybe_unused]] static void v2_finish(const int64_t* acc, const int8_t* EA, const int8_t* EW, float* y, int m, int n) {
  for (int r = 0; r < m; r++)
    for (int row = 0; row < n; row++) {
      const int B = int(EA[r]) + int(EW[row]) - MARGIN;
      y[(size_t)r * n + row] = bf(std::ldexp(float(acc[(size_t)r * n + row]), B - 10));  // one RNE in the int->float cast
    }
}
[[maybe_unused]] static void v2_linear(const int16_t* hi, const int16_t* lo, const int8_t* ea, const int8_t* EA, const uint8_t* w,
                   const uint8_t* ws, const int8_t* EW, float* y, int m, int n, int k) {
  std::vector<int64_t> acc((size_t)m * n, 0);
#if V2_ACC == 5
  const int G = k / 32;
  std::vector<int16_t> hi_f((size_t)m * k), lo_f((size_t)m * k);
  std::vector<int8_t> EAmin(m), SA(m), EW2(n), EWmin(n);
  v2_fold(hi, lo, ea, EA, m, k, hi_f.data(), lo_f.data(), EAmin.data(), SA.data());
  v2_prepare_weights2(ws, n, G, EW2.data(), EWmin.data());
  v2_linear_partial_fold(hi, lo, ea, EA, hi_f.data(), lo_f.data(), EAmin.data(), SA.data(), w, ws, EW, EWmin.data(), acc.data(), m, n, k, 0, G);
#else
  v2_linear_partial(hi, lo, ea, EA, w, ws, EW, acc.data(), m, n, k, 0, k / 32);
#endif
  v2_finish(acc.data(), EA, EW, y, m, n);
}
[[maybe_unused]] static void v2_activate(const float* gu, const float* weight, const float* silu, float* h, int m, int width) {
  for (int r = 0; r < m; r++)
    for (int j = 0; j < width; j++) {
      float g = gu[(size_t)r * width * 2 + j], u = std::max(-10.f, std::min(10.f, gu[(size_t)r * width * 2 + width + j]));
      h[(size_t)r * width + j] = bf((silu[bits(g) >> 16] * u) * weight[r]);
    }
}
// experts (tokens, topk, hidden), order (tokens, topk) = slots sorted by ascending expert id, shared (tokens, hidden) or null.
[[maybe_unused]] static void v2_combine(const float* experts, const int32_t* order, const float* shared, float* y, int tokens, int topk, int hidden) {
  for (int t = 0; t < tokens; t++)
    for (int j = 0; j < hidden; j++) {
      float a = 0.f;
      for (int e = 0; e < topk; e++) a = a + experts[((size_t)t * topk + order[(size_t)t * topk + e]) * hidden + j];
      if (shared) a = a + shared[(size_t)t * hidden + j];
      y[(size_t)t * hidden + j] = bf(a);
    }
}
