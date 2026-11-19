// Shared inline pieces of the contract v3 CPU kernels (quantiser, packed-panel GEMM, epilogues, int64 down, fixed
// point). Included inside an anonymous namespace by v3_cpu.cpp (OpenMP library entry points) and v3_rm.cpp (arena /
// row-major entry points for external worker threads). Requires <immintrin.h>, <cstring>, <cstdint>, <vector>, v3_cpu.h.

constexpr int kPF = 1024;  // software prefetch distance in the weight stream (bytes)

alignas(32) const int8_t kJ[32] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                   0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

inline uint16_t bf16_rne(float f) {
  const uint32_t b = f2u(f);
  return (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}
inline float bf2f(uint16_t h) { return u2f((uint32_t)h << 16); }

inline bool mxcsr_default() { return (_mm_getcsr() & 0xFFC0u) == 0x1F80u; }  // masks set, RNE, no FTZ/DAZ

// ---------------- packing ----------------

// Byte i of chunk c (0..3) of a (panel, block) record, as the (column, k-in-block) of its low and high nibble.
inline void nib_src(int tier, int c, int i, int& col, int& klo, int& khi) {
  if (tier == V3_TIER_C) {  // int16 pairs: vector = 8 cols x 2 k; lo nibble -> pair 4c+half, hi -> pair 4c+2+half
    const int half = i >> 4, e = i & 15, o = e & 1;
    col = e >> 1;
    klo = 8 * c + 2 * half + o;
    khi = 8 * c + 4 + 2 * half + o;
  } else {  // int8 quads: vector = 8 cols x 4 k; lo nibble -> quad 2c, hi -> quad 2c+1
    const int t = i & 3;
    col = i >> 2;
    klo = 8 * c + t;
    khi = 8 * c + 4 + t;
  }
}

template <class GetCode, class GetScale>
int pack_impl(v3_mat* m, int N, int K, int tier, GetCode code, GetScale scale) {
  if (N <= 0 || K <= 0 || N % 8 || K % 32 || (tier != V3_TIER_C && tier != V3_TIER_D)) return V3_ERR_SHAPE;
  const int G = K / 32, P = N / 8;
  m->N = N; m->K = K; m->G = G; m->tier = tier;
  m->w = (uint8_t*)std::aligned_alloc(64, ((size_t)P * G * 128 + 63) / 64 * 64);
  m->s = (uint8_t*)std::aligned_alloc(64, ((size_t)P * G * 8 + 63) / 64 * 64);
  for (int p = 0; p < P; p++)
    for (int g = 0; g < G; g++) {
      uint8_t* rec = m->w + ((size_t)p * G + g) * 128;
      for (int c = 0; c < 4; c++)
        for (int i = 0; i < 32; i++) {
          int col, klo, khi;
          nib_src(tier, c, i, col, klo, khi);
          const int n = p * 8 + col;
          rec[c * 32 + i] = (uint8_t)(code(n, g * 32 + klo) | (code(n, g * 32 + khi) << 4));
        }
      for (int col = 0; col < 8; col++) m->s[((size_t)p * G + g) * 8 + col] = scale(p * 8 + col, g);
    }
  return V3_OK;
}

// ---------------- quantiser ----------------

inline float hmax8(__m256 v) {
  __m128 a = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_max_ps(a, _mm_movehl_ps(a, a));
  a = _mm_max_ss(a, _mm_shuffle_ps(a, a, 1));
  return _mm_cvtss_f32(a);
}

// One 32-block: q (int32 codes) and the exponent ea.
inline int32_t quant_block(const float* x, int qmax, int32_t* qi) {
  const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  __m256 v[4];
  for (int j = 0; j < 4; j++) v[j] = _mm256_loadu_ps(x + 8 * j);
  const __m256 mx = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(v[0], absm), _mm256_and_ps(v[1], absm)),
                                  _mm256_max_ps(_mm256_and_ps(v[2], absm), _mm256_and_ps(v[3], absm)));
  float amax = hmax8(mx);
  if (!(amax >= 1e-30f)) amax = 1e-30f;
  volatile float qm = (float)qmax;  // keep the division a real IEEE divss
  const float r = amax / qm;
  const uint32_t rb = f2u(r);
  const int32_t e = (int32_t)((rb >> 23) & 0xFF) - 127 + ((rb & 0x7FFFFF) != 0);
  const __m256 inv = _mm256_castsi256_ps(_mm256_set1_epi32((127 - e) << 23));
  const __m256i hi = _mm256_set1_epi32(qmax), lo = _mm256_set1_epi32(-qmax);
  for (int j = 0; j < 4; j++) {
    __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(v[j], inv));  // RNE under the default MXCSR
    q = _mm256_max_epi32(_mm256_min_epi32(q, hi), lo);
    _mm256_storeu_si256((__m256i*)(qi + 8 * j), q);
  }
  return e;
}

// ---------------- GEMM core ----------------

// Exact 2^(t-127) with biased t = ea + e8m0 - 1 per lane, for lanes outside the normal range. Returns false on overflow.
inline bool slow_scale(__m256i t, __m256& scale) {
  alignas(32) int32_t tv[8];
  alignas(32) uint32_t bits[8];
  _mm256_store_si256((__m256i*)tv, t);
  for (int l = 0; l < 8; l++) {
    const int n = tv[l] - 127;
    if (n > 127) return false;
    if (n >= -126) bits[l] = (uint32_t)(n + 127) << 23;
    else if (n >= -149) bits[l] = 1u << (n + 149);
    else bits[l] = 0;
  }
  scale = _mm256_castsi256_ps(_mm256_load_si256((const __m256i*)bits));
  return true;
}

// acc[r] = fma(dot[r], 2^(ea[r] + ew - 1), acc[r]) for one block. Returns false on scale overflow.
template <int R>
inline bool epilogue(const __m256i* dot, const uint8_t* s8, const int32_t* ea_g, int ldea, __m256* acc) {
  const __m256i ew = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)s8));
  const __m256i lim = _mm256_set1_epi32(253);
#pragma GCC unroll 4
  for (int r = 0; r < R; r++) {
    // scale = 2^(ea + (e8m0 - 127) - 1): biased exponent t = ea + e8m0 - 1
    const __m256i t = _mm256_add_epi32(ew, _mm256_set1_epi32(ea_g[r * ldea] - 1));
    const __m256i tm1 = _mm256_sub_epi32(t, _mm256_set1_epi32(1));                       // normal iff 0 <= tm1 <= 253
    __m256 scale;
    if (__builtin_expect(_mm256_movemask_epi8(_mm256_cmpeq_epi32(_mm256_min_epu32(tm1, lim), tm1)) == -1, 1))
      scale = _mm256_castsi256_ps(_mm256_slli_epi32(t, 23));
    else if (!slow_scale(t, scale))
      return false;
    acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot[r]), scale, acc[r]);
  }
  return true;
}

// One panel (8 columns) x R rows, tier C. q: int16 rows (stride ldq), ea rows (stride ldea). out[r*ldo + 0..7].
template <int R>
bool panel_C(const uint8_t* wp, const uint8_t* sp, int G, const int16_t* q, int ldq, const int32_t* ea, int ldea,
             float* out, int ldo) {
  const __m256i lut = _mm256_load_si256((const __m256i*)kJ);
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  __m256 acc[R];
  for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_ps();
  for (int g = 0; g < G; g++) {
    const uint8_t* rec = wp + (size_t)g * 128;
    _mm_prefetch((const char*)rec + kPF, _MM_HINT_T0);
    _mm_prefetch((const char*)rec + kPF + 64, _MM_HINT_T0);
    __m256i id[R];
    for (int r = 0; r < R; r++) id[r] = _mm256_setzero_si256();
#pragma GCC unroll 4
    for (int c = 0; c < 4; c++) {
      const __m256i B = _mm256_loadu_si256((const __m256i*)(rec + 32 * c));
      const __m256i jl = _mm256_shuffle_epi8(lut, _mm256_and_si256(B, m4));
      const __m256i jh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(B, 4), m4));
      const __m256i w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(jl));
      const __m256i w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(jl, 1));
      const __m256i w2 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(jh));
      const __m256i w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(jh, 1));
#pragma GCC unroll 4
      for (int r = 0; r < R; r++) {
        const int16_t* qr = q + (size_t)r * ldq + g * 32 + 8 * c;
        int32_t a0, a1, a2, a3;
        std::memcpy(&a0, qr + 0, 4); std::memcpy(&a1, qr + 2, 4);
        std::memcpy(&a2, qr + 4, 4); std::memcpy(&a3, qr + 6, 4);
        const __m256i s01 = _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_set1_epi32(a0)),
                                             _mm256_madd_epi16(w1, _mm256_set1_epi32(a1)));
        const __m256i s23 = _mm256_add_epi32(_mm256_madd_epi16(w2, _mm256_set1_epi32(a2)),
                                             _mm256_madd_epi16(w3, _mm256_set1_epi32(a3)));
        id[r] = _mm256_add_epi32(id[r], _mm256_add_epi32(s01, s23));
      }
    }
    if (!epilogue<R>(id, sp + (size_t)g * 8, ea + g, ldea, acc)) return false;
  }
  for (int r = 0; r < R; r++) _mm256_storeu_ps(out + (size_t)r * ldo, acc[r]);
  return true;
}

// Tier D: int8 activations, |w| (u8) x sign(a, w) (s8) on vpmaddubsw; 8 quads summed in int16 (<= 24384), then madd.
template <int R>
bool panel_D(const uint8_t* wp, const uint8_t* sp, int G, const int8_t* q, int ldq, const int32_t* ea, int ldea,
             float* out, int ldo) {
  const __m256i lut = _mm256_load_si256((const __m256i*)kJ);
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc[R];
  for (int r = 0; r < R; r++) acc[r] = _mm256_setzero_ps();
  for (int g = 0; g < G; g++) {
    const uint8_t* rec = wp + (size_t)g * 128;
    _mm_prefetch((const char*)rec + kPF, _MM_HINT_T0);
    _mm_prefetch((const char*)rec + kPF + 64, _MM_HINT_T0);
    __m256i s16[R];
    for (int r = 0; r < R; r++) s16[r] = _mm256_setzero_si256();
#pragma GCC unroll 4
    for (int c = 0; c < 4; c++) {
      const __m256i B = _mm256_loadu_si256((const __m256i*)(rec + 32 * c));
      const __m256i jl = _mm256_shuffle_epi8(lut, _mm256_and_si256(B, m4));
      const __m256i jh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(B, 4), m4));
      const __m256i al = _mm256_abs_epi8(jl), ah = _mm256_abs_epi8(jh);
#pragma GCC unroll 4
      for (int r = 0; r < R; r++) {
        const int8_t* qr = q + (size_t)r * ldq + g * 32 + 8 * c;
        int32_t a0, a1;
        std::memcpy(&a0, qr, 4); std::memcpy(&a1, qr + 4, 4);
        const __m256i p0 = _mm256_maddubs_epi16(al, _mm256_sign_epi8(_mm256_set1_epi32(a0), jl));
        const __m256i p1 = _mm256_maddubs_epi16(ah, _mm256_sign_epi8(_mm256_set1_epi32(a1), jh));
        s16[r] = _mm256_add_epi16(s16[r], _mm256_add_epi16(p0, p1));
      }
    }
    __m256i id[R];
    for (int r = 0; r < R; r++) id[r] = _mm256_madd_epi16(s16[r], ones);
    if (!epilogue<R>(id, sp + (size_t)g * 8, ea + g, ldea, acc)) return false;
  }
  for (int r = 0; r < R; r++) _mm256_storeu_ps(out + (size_t)r * ldo, acc[r]);
  return true;
}

// All rows of one panel p into out[m*ldo + 0..7].
bool panel_rows(const v3_mat* m, int p, const void* q, const int32_t* ea, int M, float* out, int ldo) {
  const int G = m->G, K = m->K;
  const uint8_t* wp = m->w + (size_t)p * G * 128;
  const uint8_t* sp = m->s + (size_t)p * G * 8;
  for (int r0 = 0; r0 < M; r0 += 4) {
    const int R = M - r0 < 4 ? M - r0 : 4;
    const int32_t* e = ea + (size_t)r0 * G;
    float* o = out + (size_t)r0 * ldo;
    bool ok;
    if (m->tier == V3_TIER_C) {
      const int16_t* qq = (const int16_t*)q + (size_t)r0 * K;
      switch (R) {
        case 1: ok = panel_C<1>(wp, sp, G, qq, K, e, G, o, ldo); break;
        case 2: ok = panel_C<2>(wp, sp, G, qq, K, e, G, o, ldo); break;
        case 3: ok = panel_C<3>(wp, sp, G, qq, K, e, G, o, ldo); break;
        default: ok = panel_C<4>(wp, sp, G, qq, K, e, G, o, ldo); break;
      }
    } else {
      const int8_t* qq = (const int8_t*)q + (size_t)r0 * K;
      switch (R) {
        case 1: ok = panel_D<1>(wp, sp, G, qq, K, e, G, o, ldo); break;
        case 2: ok = panel_D<2>(wp, sp, G, qq, K, e, G, o, ldo); break;
        case 3: ok = panel_D<3>(wp, sp, G, qq, K, e, G, o, ldo); break;
        default: ok = panel_D<4>(wp, sp, G, qq, K, e, G, o, ldo); break;
      }
    }
    if (!ok) return false;
  }
  return true;
}

// ---------------- spec r2: exact int64 down projection over any block subset ----------------

// One block's int32 dots (8 columns per row) for tier C / D. rec = the (panel, block) record, qb = row 0 at the block.
template <int R>
inline void block_dot_C(const uint8_t* rec, const int16_t* qb, int ldq, __m256i* id) {
  const __m256i lut = _mm256_load_si256((const __m256i*)kJ);
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  for (int r = 0; r < R; r++) id[r] = _mm256_setzero_si256();
#pragma GCC unroll 4
  for (int c = 0; c < 4; c++) {
    const __m256i B = _mm256_loadu_si256((const __m256i*)(rec + 32 * c));
    const __m256i jl = _mm256_shuffle_epi8(lut, _mm256_and_si256(B, m4));
    const __m256i jh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(B, 4), m4));
    const __m256i w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(jl));
    const __m256i w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(jl, 1));
    const __m256i w2 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(jh));
    const __m256i w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(jh, 1));
#pragma GCC unroll 4
    for (int r = 0; r < R; r++) {
      const int16_t* qr = qb + (size_t)r * ldq + 8 * c;
      int32_t a0, a1, a2, a3;
      std::memcpy(&a0, qr + 0, 4); std::memcpy(&a1, qr + 2, 4);
      std::memcpy(&a2, qr + 4, 4); std::memcpy(&a3, qr + 6, 4);
      const __m256i s01 = _mm256_add_epi32(_mm256_madd_epi16(w0, _mm256_set1_epi32(a0)),
                                           _mm256_madd_epi16(w1, _mm256_set1_epi32(a1)));
      const __m256i s23 = _mm256_add_epi32(_mm256_madd_epi16(w2, _mm256_set1_epi32(a2)),
                                           _mm256_madd_epi16(w3, _mm256_set1_epi32(a3)));
      id[r] = _mm256_add_epi32(id[r], _mm256_add_epi32(s01, s23));
    }
  }
}

template <int R>
inline void block_dot_D(const uint8_t* rec, const int8_t* qb, int ldq, __m256i* id) {
  const __m256i lut = _mm256_load_si256((const __m256i*)kJ);
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  __m256i s16[R];
  for (int r = 0; r < R; r++) s16[r] = _mm256_setzero_si256();
#pragma GCC unroll 4
  for (int c = 0; c < 4; c++) {
    const __m256i B = _mm256_loadu_si256((const __m256i*)(rec + 32 * c));
    const __m256i jl = _mm256_shuffle_epi8(lut, _mm256_and_si256(B, m4));
    const __m256i jh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(B, 4), m4));
    const __m256i al = _mm256_abs_epi8(jl), ah = _mm256_abs_epi8(jh);
#pragma GCC unroll 4
    for (int r = 0; r < R; r++) {
      const int8_t* qr = qb + (size_t)r * ldq + 8 * c;
      int32_t a0, a1;
      std::memcpy(&a0, qr, 4); std::memcpy(&a1, qr + 4, 4);
      const __m256i p0 = _mm256_maddubs_epi16(al, _mm256_sign_epi8(_mm256_set1_epi32(a0), jl));
      const __m256i p1 = _mm256_maddubs_epi16(ah, _mm256_sign_epi8(_mm256_set1_epi32(a1), jh));
      s16[r] = _mm256_add_epi16(s16[r], _mm256_add_epi16(p0, p1));
    }
  }
  const __m256i ones = _mm256_set1_epi16(1);
  for (int r = 0; r < R; r++) id[r] = _mm256_madd_epi16(s16[r], ones);
}

// t = floor(dot * 2^(s - bL)) as int64, s = ea + e8m0 - 128; S += t. Right shifts in int32 (vpsravd: counts > 31
// give 0 / -1), then widen, then the left shift in int64. bad: lanes with s - bL > 38 (s - bL + 24 > 62).
// flag:  only the FINAL S is range-checked (by the finish); a part's
// sum is a partial and may pass 2^62 while the total does not. Adds wrap mod 2^64 (vpaddq).
template <int R>
inline void epi_i64(const __m256i* dot, const uint8_t* s8, const int32_t* ea_g, int ldea, int bL, __m256i* S,
                    __m256i& bad, __m256i& flag) {
  const __m256i ew = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)s8));
  const __m256i zero = _mm256_setzero_si256(), lim = _mm256_set1_epi32(38);
  const __m256i C = _mm256_set1_epi64x(((int64_t)1 << 62) - 1);
#pragma GCC unroll 4
  for (int r = 0; r < R; r++) {
    const __m256i sh = _mm256_add_epi32(ew, _mm256_set1_epi32(ea_g[r * ldea] - 128 - bL));
    bad = _mm256_or_si256(bad, _mm256_cmpgt_epi32(sh, lim));
    const __m256i r32 = _mm256_srav_epi32(dot[r], _mm256_max_epi32(_mm256_sub_epi32(zero, sh), zero));
    const __m256i ls = _mm256_max_epi32(sh, zero);
    const __m256i lo = _mm256_sllv_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(r32)),
                                         _mm256_cvtepu32_epi64(_mm256_castsi256_si128(ls)));
    const __m256i hi = _mm256_sllv_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(r32, 1)),
                                         _mm256_cvtepu32_epi64(_mm256_extracti128_si256(ls, 1)));
    S[2 * r] = _mm256_add_epi64(S[2 * r], lo);
    S[2 * r + 1] = _mm256_add_epi64(S[2 * r + 1], hi);
    (void)C; (void)flag;
  }
}

// One panel x R rows over the listed blocks; out (int64, row stride ldo) is read, added to, written back.
template <int R, int TIER>
bool panel_down(const uint8_t* wp, const uint8_t* sp, const int* blocks, int nb, const void* q, int ldq,
                const int32_t* ea, int ldea, int bL, int64_t* out, int ldo) {
  __m256i S[2 * R];
  for (int r = 0; r < R; r++) {
    S[2 * r] = _mm256_loadu_si256((const __m256i*)(out + (size_t)r * ldo));
    S[2 * r + 1] = _mm256_loadu_si256((const __m256i*)(out + (size_t)r * ldo + 4));
  }
  __m256i bad = _mm256_setzero_si256(), flag = _mm256_setzero_si256();
  for (int i = 0; i < nb; i++) {
    const int g = blocks[i];
    const uint8_t* rec = wp + (size_t)g * 128;
    if (i + 1 < nb) {
      const char* nx = (const char*)(wp + (size_t)blocks[i + 1] * 128);
      _mm_prefetch(nx, _MM_HINT_T0);
      _mm_prefetch(nx + 64, _MM_HINT_T0);
    }
    __m256i id[R];
    if (TIER == V3_TIER_C)
      block_dot_C<R>(rec, (const int16_t*)q + g * 32, ldq, id);
    else
      block_dot_D<R>(rec, (const int8_t*)q + g * 32, ldq, id);
    epi_i64<R>(id, sp + (size_t)g * 8, ea + g, ldea, bL, S, bad, flag);
  }
  for (int r = 0; r < R; r++) {
    _mm256_storeu_si256((__m256i*)(out + (size_t)r * ldo), S[2 * r]);
    _mm256_storeu_si256((__m256i*)(out + (size_t)r * ldo + 4), S[2 * r + 1]);
  }
  return _mm256_movemask_epi8(bad) == 0 && _mm256_movemask_pd(_mm256_castsi256_pd(flag)) == 0;
}

// All rows of output panel p over the listed blocks, added into S (row stride ldS, column offset p*8).
bool down_rows(const v3_mat* m, int p, const int* blocks, int nb, const void* q, const int32_t* ea, int M, int bL,
               int64_t* S, int ldS) {
  const int G = m->G, K = m->K;
  const uint8_t* wp = m->w + (size_t)p * G * 128;
  const uint8_t* sp = m->s + (size_t)p * G * 8;
  bool ok = true;
  for (int r0 = 0; r0 < M; r0 += 4) {
    const int R = M - r0 < 4 ? M - r0 : 4;
    const int32_t* e = ea + (size_t)r0 * G;
    int64_t* o = S + (size_t)r0 * ldS + p * 8;
    const void* qq = m->tier == V3_TIER_C ? (const void*)((const int16_t*)q + (size_t)r0 * K)
                                          : (const void*)((const int8_t*)q + (size_t)r0 * K);
    bool k;
    if (m->tier == V3_TIER_C) {
      switch (R) {
        case 1: k = panel_down<1, V3_TIER_C>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
        case 2: k = panel_down<2, V3_TIER_C>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
        case 3: k = panel_down<3, V3_TIER_C>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
        default: k = panel_down<4, V3_TIER_C>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
      }
    } else {
      switch (R) {
        case 1: k = panel_down<1, V3_TIER_D>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
        case 2: k = panel_down<2, V3_TIER_D>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
        case 3: k = panel_down<3, V3_TIER_D>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
        default: k = panel_down<4, V3_TIER_D>(wp, sp, blocks, nb, qq, K, e, G, bL, o, ldS); break;
      }
    }
    ok = ok && k;
  }
  return ok;
}

inline int64_t fixed_one(uint16_t b, int bL, int& err) {
  const int s = b >> 15, E = (b >> 7) & 0xFF, mm = b & 0x7F;
  if (E == 0xFF) { err = V3_ERR_NONFINITE; return 0; }
  const int64_t mant = E ? (mm | 0x80) : mm;
  if (mant == 0) return 0;
  const int sh = (E ? E - 134 : -133) - bL;
  if (sh >= 0) {
    if ((64 - __builtin_clzll((unsigned long long)mant)) + sh > 62) { err = V3_ERR_FIXED_OVERFLOW; return 0; }  // |v| >= 2^62
    const int64_t v = mant << sh;
    return s ? -v : v;
  }
  const int rs = -sh;
  if (rs > 62) return s ? -1 : 0;
  return s ? -((mant + ((int64_t)1 << rs) - 1) >> rs) : (mant >> rs);  // floor
}

void quant_rows(const float* x, int M, int K, int tier, void* q, int32_t* ea) {
  const int G = K / 32, qmax = tier == V3_TIER_C ? 32767 : 127;
  alignas(32) int32_t qi[32];
  for (int mi = 0; mi < M; mi++)
    for (int g = 0; g < G; g++) {
      ea[(size_t)mi * G + g] = quant_block(x + (size_t)mi * K + g * 32, qmax, qi);
      if (tier == V3_TIER_C) {
        int16_t* d = (int16_t*)q + (size_t)mi * K + g * 32;
        for (int j = 0; j < 32; j++) d[j] = (int16_t)qi[j];
      } else {
        int8_t* d = (int8_t*)q + (size_t)mi * K + g * 32;
        for (int j = 0; j < 32; j++) d[j] = (int8_t)qi[j];
      }
    }
}
