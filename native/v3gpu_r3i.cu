// Contract v3 routed experts for SM120. Accumulate both rank partials in int64 and round once.
#include <cstdint>
#include <cuda_runtime.h>

#define DEV __device__ __forceinline__

DEV float pow2_normal(int n) { return __int_as_float((n + 127) << 23); }           // n in [-126, 127]
DEV float pow2_any(int n) {                                                        // n in [-149, 127]
  return n >= -126 ? __int_as_float((n + 127) << 23) : __int_as_float(1 << (n + 149));
}
DEV float wscale(uint32_t b) { return pow2_any((int)b - 128); }                    // 2^(ew - 1), ew = b - 127
DEV uint16_t bf16_rne(float f) {
  uint32_t b = __float_as_uint(f);
  return (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}

// one 32-bit word of packed FP4 (bytes 4i..4i+3 of a block: elements 8i..8i+7) -> J = 2*value as int8:
// jl = low nibbles (elements 8i, 8i+2, 8i+4, 8i+6 = logical 4i..4i+3), jh = high nibbles (logical 16+4i..16+4i+3)
DEV void cvtJ(uint32_t w, uint32_t& jl, uint32_t& jh) {
  const uint32_t LUT0 = 0x03020100u;  // |code| 0..3 -> 0, 1, 2, 3
  const uint32_t LUT1 = 0x0C080604u;  // |code| 4..7 -> 4, 6, 8, 12
  uint32_t ml = w & 0x07070707u, mh = (w >> 4) & 0x07070707u;
  uint32_t tl = ml | (ml >> 4), th = mh | (mh >> 4);
  uint32_t al = __byte_perm(LUT0, LUT1, (tl & 0xFFu) | ((tl >> 8) & 0xFF00u));
  uint32_t ah = __byte_perm(LUT0, LUT1, (th & 0xFFu) | ((th >> 8) & 0xFF00u));
  uint32_t nl = ((w >> 3) & 0x01010101u) * 0xFFu, nh = ((w >> 7) & 0x01010101u) * 0xFFu;
  jl = (al & ~nl) | (__vneg4(al) & nl);
  jh = (ah & ~nh) | (__vneg4(ah) & nh);
}

// ------------------------------------------------------------------------------------------------ quantiser
template <int TC>
DEV void quant_block(const uint16_t* xb, void* __restrict__ Qout, int* __restrict__ EA, size_t blk);
template <int TC>
__global__ void k_quant(const uint16_t* __restrict__ X, void* __restrict__ Qout, int* __restrict__ EA, int nblk) {
  int blk = blockIdx.x * blockDim.x + threadIdx.x;
  if (blk >= nblk) return;
  const uint4* xs = (const uint4*)(X + (size_t)blk * 32);
  uint16_t xb[32];
#pragma unroll
  for (int i = 0; i < 4; ++i) { uint4 v = xs[i]; *(uint4*)(xb + 8 * i) = v; }
  quant_block<TC>(xb, Qout, EA, blk);
}
// one 32-block of BF16 bits -> permuted codes + exponent (the Q3i definition)
template <int TC>
DEV void quant_block(const uint16_t* xb, void* __restrict__ Qout, int* __restrict__ EA, size_t blk) {
  constexpr int QMAX = TC ? 32767 : 127;
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(__uint_as_float((uint32_t)xb[j] << 16)));
  amax = fmaxf(amax, 1e-30f);
  float r = __fdiv_rn(amax, (float)QMAX);
  uint32_t rb = __float_as_uint(r);
  int e = (int)((rb >> 23) & 0xFF) - 127 + ((rb & 0x7FFFFF) != 0);
  EA[blk] = e;
  float inv = __int_as_float((127 - e) << 23);
  int q[32];
#pragma unroll
  for (int p = 0; p < 32; ++p) {
    int k = p < 16 ? 2 * p : 2 * (p - 16) + 1;
    int v = __float2int_rn(__fmul_rn(__uint_as_float((uint32_t)xb[k] << 16), inv));
    q[p] = max(-QMAX, min(QMAX, v));
  }
  if (TC) {
    uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = ((uint32_t)q[2 * i] & 0xFFFFu) | ((uint32_t)q[2 * i + 1] << 16);
    uint4* d = (uint4*)((int16_t*)Qout + (size_t)blk * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
  } else {
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      o[i] = ((uint32_t)q[4 * i] & 255u) | (((uint32_t)q[4 * i + 1] & 255u) << 8) | (((uint32_t)q[4 * i + 2] & 255u) << 16) |
             ((uint32_t)q[4 * i + 3] << 24);
    uint4* d = (uint4*)((int8_t*)Qout + (size_t)blk * 32);
    d[0] = make_uint4(o[0], o[1], o[2], o[3]);
    d[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

DEV long long fixed_of(uint32_t yb, int bL, int* flag);
// memfix r1 (staging r1.4, 9 Nov): the layer grid bL may come from device memory, so one captured streamer group graph
// serves every layer and bL is no longer part of the graph key (r1.3 captured one graph set per bL class: 5x the host and
// device graph memory). A launch with bL == V3_BL_DEVICE reads g_v3_bl, written by v3_set_bl on the same stream before
// each replay. Same value, same arithmetic, same bits. A value outside the legal grid range sets flag bit 16.
#define V3_BL_DEVICE 0x7fffffff
__device__ int g_v3_bl = V3_BL_DEVICE;
__global__ void k_set_bl(int bL) { g_v3_bl = bL; }
// r3d (#7, review H6): one bL word per streamer slot, written by k_stage_bl on the transfer stream right after the slot's
// weight copies (ordered by the slot's dma_done / compute_done events exactly like the weights), so the grid travels with
// the data and no second stream can see another layer's grid. A launch with bL = V3_BL_SLOT0 + k reads slot k's word;
// r3e (freeze F11): a word never staged (still V3_BL_DEVICE) is illegal: flag 16 and the kernel returns.
#define V3_BL_SLOT0 0x7fffff00
#define V3_BL_SLOTS 8
__device__ int g_v3_bl_slot[V3_BL_SLOTS] = {V3_BL_DEVICE, V3_BL_DEVICE, V3_BL_DEVICE, V3_BL_DEVICE,
                                            V3_BL_DEVICE, V3_BL_DEVICE, V3_BL_DEVICE, V3_BL_DEVICE};
__global__ void k_stage_bl(int k, int bL) { g_v3_bl_slot[k] = bL; }
#define V3_RESOLVE_BL(flagp) do { if (bL >= V3_BL_SLOT0 && bL < V3_BL_SLOT0 + V3_BL_SLOTS) { bL = g_v3_bl_slot[bL - V3_BL_SLOT0]; } else if (bL == V3_BL_DEVICE) { bL = g_v3_bl; } if (bL < -126 || bL > 0) { if ((flagp) != nullptr) atomicOr((flagp), 16); return; } } while (0)
// ------------------------------------------------------------------------------------------------ down epilogue (spec r2)
// t = floor(dot * 2^sh) as int64, sh = ea + ew - 1 - bL: dot << sh, or arithmetic dot >> min(-sh, 63). sh > 38 (sh + 24 > 62)
// raises the overflow flag (the oracle raises).
DEV long long down_term(int dot, int sh) {
  const long long d = dot;
  return sh >= 0 ? (long long)((unsigned long long)d << min(sh, 63)) : (d >> min(-sh, 63));
}
DEV uint16_t down_finish(long long S, int bL, int& bad) {
  bad |= (S >= (1LL << 62) || S <= -(1LL << 62)) ? 4 : 0;
  return bf16_rne(__fmul_rn(__ll2float_rn(S), pow2_normal(bL)));
}

// ------------------------------------------------------------------------------------------------ decode (CUDA cores)
// One lane owns one output column of one task (one expert, up to R rows) and runs its R chains over blocks [gb, ge):
//   DOWN = 0 (gate/up): FP32 chain in ascending block order (gb = 0, ge = G).
//   DOWN = 1 (down, spec r2): exact int64 sum of the block terms on the grid bL (any range, any order). With outS the
//            int64 partial is stored (no finish); otherwise acc = f32_rne(S) * 2^bL and the BF16 output is written.
// task t: expert texp[t], rows trows[4t .. 4t+R-1] (-1 = absent). Output out[(4t + r) * N + n].
template <int TC, int R, int DOWN>
__global__ void __launch_bounds__(256) k_decode(const void* __restrict__ Q, const int* __restrict__ EA, const uint8_t* __restrict__ W,
                                                const uint8_t* __restrict__ WS, const int* __restrict__ texp,
                                                const int* __restrict__ trows, int N, int K, int bL, int gb, int ge,
                                                uint16_t* __restrict__ out, float* __restrict__ outf, long long* __restrict__ outS,
                                                int* __restrict__ flag, const int* __restrict__ odst = nullptr,
                                                const int* __restrict__ ptok = nullptr, long long* __restrict__ presum = nullptr,
                                                int praw = 0) {
  V3_RESOLVE_BL(flag);
  extern __shared__ __align__(16) uint8_t dsmem[];
  const int G = K / 32, QB = K * (TC ? 2 : 1), t = blockIdx.y, e = texp[t];
  if (e < 0) return;   // staging r1: an inactive task of a fixed-size (CUDA-graph) task list; block-uniform, before any load
  int rows[R];
#pragma unroll
  for (int r = 0; r < R; ++r) rows[r] = trows[4 * t + r];
  uint8_t* sq = dsmem;
  float* ssa = (float*)(dsmem + (size_t)R * QB);
  int* sia = (int*)ssa;
  for (int r = 0; r < R; ++r) {
    const uint4* src = (const uint4*)((const uint8_t*)Q + (size_t)(rows[r] < 0 ? 0 : rows[r]) * QB);
    uint4* dst = (uint4*)(sq + (size_t)r * QB);
    for (int i = threadIdx.x; i < QB / 16; i += blockDim.x) dst[i] = rows[r] < 0 ? make_uint4(0, 0, 0, 0) : src[i];
  }
  for (int i = threadIdx.x; i < R * G; i += blockDim.x) {
    int r = i / G, g = i - r * G;
    if (DOWN) sia[i] = rows[r] < 0 ? 0 : EA[(size_t)rows[r] * G + g] - bL;
    else ssa[i] = rows[r] < 0 ? 0.f : pow2_normal(EA[(size_t)rows[r] * G + g]);
  }
  __syncthreads();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const size_t grp = (size_t)e * (N / 32) + n / 32;
  const uint4* wcol = (const uint4*)W + grp * G * 32 + (n & 31);              // block g at wcol[g * 32]
  const unsigned int* scol = (const unsigned int*)WS + grp * (G / 4) * 32 + (n & 31);   // quad g/4 at scol[(g/4) * 32]
  float acc[R];
  long long acc64[R];
#pragma unroll
  for (int r = 0; r < R; ++r) { acc[r] = 0.f; acc64[r] = 0; }
  int bad = 0;
  for (int g8 = gb; g8 < ge; g8 += 8) {
    uint4 wv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] = (g8 + i < ge) ? __ldg(wcol + (size_t)(g8 + i) * 32) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int g = g8 + i;
      if (g >= ge) break;
      const uint32_t sb = (__ldg(scol + (size_t)(g >> 2) * 32) >> (8 * (g & 3))) & 255u;
      const float sw = wscale(sb);
      uint32_t ww[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w}, jl[4], jh[4];
#pragma unroll
      for (int w = 0; w < 4; ++w) cvtJ(ww[w], jl[w], jh[w]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (rows[r] < 0) continue;
        int dot = 0;
        if (TC) {
          const uint4* qv = (const uint4*)(sq + (size_t)r * QB + g * 64);
          uint4 q0 = qv[0], q1 = qv[1], q2 = qv[2], q3 = qv[3];
          uint32_t qw[16] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w, q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w};
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            dot = __dp2a_lo((int)qw[2 * w], (int)jl[w], dot);
            dot = __dp2a_hi((int)qw[2 * w + 1], (int)jl[w], dot);
            dot = __dp2a_lo((int)qw[8 + 2 * w], (int)jh[w], dot);
            dot = __dp2a_hi((int)qw[8 + 2 * w + 1], (int)jh[w], dot);
          }
        } else {
          const uint4* qv = (const uint4*)(sq + (size_t)r * QB + g * 32);
          uint4 q0 = qv[0], q1 = qv[1];
          uint32_t qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            dot = __dp4a((int)qw[w], (int)jl[w], dot);
            dot = __dp4a((int)qw[4 + w], (int)jh[w], dot);
          }
        }
        if (DOWN) {
          const int sh = sia[r * G + g] + (int)sb - 128;
          bad |= sh > 38 ? 2 : 0;
          acc64[r] = (long long)((unsigned long long)acc64[r] + (unsigned long long)down_term(dot, sh));
        } else {
          const float s = __fmul_rn(ssa[r * G + g], sw);
          bad |= isinf(s) ? 1 : 0;
          acc[r] = __fmaf_rn(__int2float_rn(dot), s, acc[r]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (rows[r] < 0) continue;
    size_t o = ((size_t)4 * t + r) * N + n;
    if (DOWN) {
      if (outS) outS[o] = acc64[r];
      else if (presum && praw) {
        // spec r3: the raw exact int64 down sum of this route goes straight into the token's sum (no per-expert finish)
        // r3d (#8, review H8a): no |S| check on this route partial; only the final token sum is checked (finish)
        if (acc64[r]) atomicAdd((unsigned long long*)&presum[(size_t)ptok[4 * t + r] * N + n], (unsigned long long)acc64[r]);
      } else {
        const uint16_t yb = down_finish(acc64[r], bL, bad);
        // fused path: the row goes straight to its route slot; the presum adds F(y) on the same grid bL (exact, order-free)
        if (out) out[odst ? (size_t)odst[4 * t + r] * N + n : o] = yb;
        if (presum) {
          const long long v = fixed_of(yb, bL, flag);
          if (v) atomicAdd((unsigned long long*)&presum[(size_t)ptok[4 * t + r] * N + n], (unsigned long long)v);
        }
        if (outf) outf[o] = __fmul_rn(__ll2float_rn(acc64[r]), pow2_normal(bL));
      }
    } else {
      out[o] = bf16_rne(acc[r]);
      if (outf) outf[o] = acc[r];
    }
  }
  if (bad) atomicOr(flag, bad);
}

// ------------------------------------------------------------------------------------------------ prefill (IMMA)
DEV void mma_s8s8(int d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
DEV void mma_u8s8(int d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

// Tile BM x BN = 64 x (16 NT), BK = 128 (4 blocks), 4 warps (2 x 2), each warp 32 x 8NT = 2 m-tiles x NT n-tiles.
// NT = 4 for gate/up (FP32 chain), NT = NTD for down (int64 accumulators take twice the registers).
// tile i: expert tile_exp[i], sorted rows tile_row0[i] .. + tile_nrows[i] (<= 64) of sorted_rows (indices into Q),
// output row = tile_row0[i] + r (sorted order). Double-buffered global -> shared through registers.
// DOWN: blockIdx.z = split; blocks [splits[z], splits[z+1]) (any boundaries); with outS the int64 partial is added by
// 64-bit integer atomics (exact, order-free; the caller zeroes outS and finishes), otherwise finished in place.
#ifndef NTD_VALUE
#define NTD_VALUE 2
#endif
// down tile width: tier C NT = 2 (32 columns), tier D NT = 4 (64 columns), the faster of each in the r2 timing
constexpr int PBM = 64, PSA = 144, PSB = 80, NTD = NTD_VALUE, NTD_D = 4;
template <int NP, int BN> struct PSmem {
  uint8_t A[2][NP][PBM][PSA];
  uint8_t B[2][BN][PSB];
  float sa[2][PBM][4], sw[2][BN][4];     // FP32 path: 2^ea, 2^(ew-1); DOWN: (as int bits) ea - bL, ew - 1
  int row[PBM];
};
template <int TC, int DOWN>
__global__ void __launch_bounds__(128) k_prefill(const void* __restrict__ Q, const int* __restrict__ EA, const uint8_t* __restrict__ W,
                                                 const uint8_t* __restrict__ WS, const int* __restrict__ sorted_rows,
                                                 const int* __restrict__ tile_exp, const int* __restrict__ tile_row0,
                                                 const int* __restrict__ tile_nrows, int N, int K, int bL,
                                                 const int* __restrict__ splits, uint16_t* __restrict__ out,
                                                 float* __restrict__ outf, long long* __restrict__ outS, int* __restrict__ flag,
                                                 const int* __restrict__ otok = nullptr,
                                                 const int* __restrict__ ntact = nullptr, int ntcap = 0) {
  V3_RESOLVE_BL(flag);
  constexpr int NP = TC ? 2 : 1;
  constexpr int NT = DOWN ? (TC ? NTD : NTD_D) : 4, BN = 16 * NT;
  extern __shared__ __align__(16) uint8_t psmem[];
  PSmem<NP, BN>& S = *reinterpret_cast<PSmem<NP, BN>*>(psmem);
  auto& sA = S.A; auto& sB = S.B; auto& ssa = S.sa; auto& ssw = S.sw; auto& srow = S.row;
  // r3f: with ntact (a device-side count of the leading non-empty tiles) this block walks tiles blockIdx.y,
  // blockIdx.y + gridDim.y, ... below min(*ntact, ntcap); without it, the one tile blockIdx.y (r3e). Block-uniform.
  const int ntl = ntact ? min(*ntact, ntcap) : 0;
  for (int tile = blockIdx.y; ; tile += gridDim.y) {
  if (ntact && tile >= ntl) break;
  const int e = tile_exp[tile], row0 = tile_row0[tile], nrows = tile_nrows[tile];
  // staging r1: an empty tile of a fixed-size (CUDA-graph) tile list; block-uniform
  if (nrows <= 0 || e < 0) { if (ntact) continue; return; }
  const int n0 = blockIdx.x * BN, G = K / 32, tid = threadIdx.x;
  const int gb = DOWN ? splits[blockIdx.z] : 0, ge = DOWN ? splits[blockIdx.z + 1] : G;
  const int kt0 = gb / 4, kt1 = (ge + 3) / 4;
  const int warp = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3, wm = warp & 1, wn = warp >> 1;
  if (tid < PBM) srow[tid] = tid < nrows ? sorted_rows[row0 + tid] : -1;
  __syncthreads();

  constexpr int AV = TC ? 16 : 8;            // uint4 per row per stage
  constexpr int AI = PBM * AV / 128;         // per thread
  constexpr int BI = BN * 4 / 128;           // B uint4 per thread
  uint4 ra[AI], rb[BI];
  uint32_t rsw = 0; int rea[4] = {0, 0, 0, 0};
  auto gload = [&](int kt) {
#pragma unroll
    for (int i = 0; i < AI; ++i) {
      int idx = tid + 128 * i, r = idx / AV, c = idx - r * AV, gr = srow[r];
      ra[i] = gr < 0 ? make_uint4(0, 0, 0, 0)
                     : __ldg((const uint4*)((const uint8_t*)Q + (size_t)gr * K * (TC ? 2 : 1) + (size_t)kt * 128 * (TC ? 2 : 1)) + c);
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      int idx = tid + 128 * i, n = idx >> 2, c = idx & 3;
      const int col = n0 + n;
      rb[i] = __ldg((const uint4*)W + (((size_t)e * (N / 32) + col / 32) * G + kt * 4 + c) * 32 + (col & 31));
    }
    if (tid < BN) {
      const int col = n0 + tid;
      rsw = __ldg((const unsigned int*)WS + (((size_t)e * (N / 32) + col / 32) * (G / 4) + kt) * 32 + (col & 31));
    } else if (tid >= 64) {
      int gr = srow[tid - 64];
#pragma unroll
      for (int i = 0; i < 4; ++i) rea[i] = gr < 0 ? 0 : EA[(size_t)gr * G + kt * 4 + i];
    }
  };
  auto sstore = [&](int buf) {
#pragma unroll
    for (int i = 0; i < AI; ++i) {
      int idx = tid + 128 * i, r = idx / AV, c = idx - r * AV;
      if (TC) {   // 8 int16 -> 8 high bytes (s8) and 8 low bytes (u8)
        uint32_t v[4] = {ra[i].x, ra[i].y, ra[i].z, ra[i].w};
        uint32_t h0 = __byte_perm(v[0], v[1], 0x7531), h1 = __byte_perm(v[2], v[3], 0x7531);
        uint32_t l0 = __byte_perm(v[0], v[1], 0x6420), l1 = __byte_perm(v[2], v[3], 0x6420);
        *(uint2*)&sA[buf][1][r][c * 8] = make_uint2(h0, h1);
        *(uint2*)&sA[buf][0][r][c * 8] = make_uint2(l0, l1);
      } else {
        *(uint4*)&sA[buf][0][r][c * 16] = ra[i];
      }
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      int idx = tid + 128 * i, n = idx >> 2, c = idx & 3;
      *(uint4*)&sB[buf][n][c * 16] = rb[i];
    }
    if (tid < BN) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t b = (rsw >> (8 * i)) & 255u;
        ssw[buf][tid][i] = DOWN ? __int_as_float((int)b - 128) : wscale(b);
      }
    } else if (tid >= 64) {
      int r = tid - 64;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ssa[buf][r][i] = DOWN ? __int_as_float(srow[r] < 0 ? 0 : rea[i] - bL) : (srow[r] < 0 ? 0.f : pow2_normal(rea[i]));
    }
  };

  constexpr int MA = DOWN ? 1 : 2, MD = DOWN ? 2 : 1;   // only the used accumulator set is sized
  float acc[MA][DOWN ? 1 : NT][4];
  long long acc64[MD][DOWN ? NT : 1][4];
#pragma unroll
  for (int a = 0; a < MA; ++a)
#pragma unroll
    for (int b = 0; b < (DOWN ? 1 : NT); ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
#pragma unroll
  for (int a = 0; a < MD; ++a)
#pragma unroll
    for (int b = 0; b < (DOWN ? NT : 1); ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc64[a][b][c] = 0;
  int bad = 0;

  gload(kt0);
  sstore(0);
  __syncthreads();
  for (int kt = kt0; kt < kt1; ++kt) {
    const int buf = (kt - kt0) & 1;
    if (kt + 1 < kt1) gload(kt + 1);
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int g = kt * 4 + b;
      if (DOWN && (g < gb || g >= ge)) continue;     // uniform across the block
      uint32_t af[NP][2][4];
#pragma unroll
      for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const uint8_t* base = &sA[buf][p][wm * 32 + mt * 16][b * 32];
          af[p][mt][0] = *(const uint32_t*)(base + gq * PSA + 4 * tq);
          af[p][mt][1] = *(const uint32_t*)(base + (gq + 8) * PSA + 4 * tq);
          af[p][mt][2] = *(const uint32_t*)(base + gq * PSA + 16 + 4 * tq);
          af[p][mt][3] = *(const uint32_t*)(base + (gq + 8) * PSA + 16 + 4 * tq);
        }
      float sa[2][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        sa[mt][0] = ssa[buf][wm * 32 + mt * 16 + gq][b];
        sa[mt][1] = ssa[buf][wm * 32 + mt * 16 + gq + 8][b];
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int col = wn * (8 * NT) + nt * 8;
        uint32_t jl, jh;
        cvtJ(*(const uint32_t*)&sB[buf][col + gq][b * 16 + 4 * tq], jl, jh);
        const float sw0 = ssw[buf][col + 2 * tq][b], sw1 = ssw[buf][col + 2 * tq + 1][b];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          int d[4];
          if (TC) {
            int dh[4], dl[4];
            mma_s8s8(dh, af[1][mt], jl, jh);
            mma_u8s8(dl, af[0][mt], jl, jh);
#pragma unroll
            for (int i = 0; i < 4; ++i) d[i] = dh[i] * 256 + dl[i];
          } else {
            mma_s8s8(d, af[0][mt], jl, jh);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            if (DOWN) {
              const int sh = __float_as_int(sa[mt][i >> 1]) + __float_as_int((i & 1) ? sw1 : sw0);
              bad |= sh > 38 ? 2 : 0;
              acc64[DOWN ? mt : 0][DOWN ? nt : 0][i] = (long long)((unsigned long long)acc64[DOWN ? mt : 0][DOWN ? nt : 0][i] +
                                                                   (unsigned long long)down_term(d[i], sh));
            } else {
              const float s = __fmul_rn(sa[mt][i >> 1], (i & 1) ? sw1 : sw0);
              bad |= isinf(s) ? 1 : 0;
              acc[DOWN ? 0 : mt][DOWN ? 0 : nt][i] = __fmaf_rn(__int2float_rn(d[i]), s, acc[DOWN ? 0 : mt][DOWN ? 0 : nt][i]);
            }
          }
        }
      }
    }
    if (kt + 1 < kt1) sstore(buf ^ 1);
    __syncthreads();
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int r = wm * 32 + mt * 16 + gq + ((i >> 1) ? 8 : 0), c = n0 + wn * (8 * NT) + nt * 8 + 2 * tq + (i & 1);
        if (r < nrows) {
          size_t o = (size_t)(row0 + r) * N + c;
          if (DOWN) {
            const long long v = acc64[DOWN ? mt : 0][DOWN ? nt : 0][i];
            if (outS) {
              // r3 (otok): the int64 partial goes to the token's row (sorted row -> token), else to the sorted row
              const size_t oo = otok ? (size_t)otok[row0 + r] * N + c : o;
              if (v) atomicAdd((unsigned long long*)&outS[oo], (unsigned long long)v);
            }
            else {
              out[o] = down_finish(v, bL, bad);
              if (outf) outf[o] = __fmul_rn(__ll2float_rn(v), pow2_normal(bL));
            }
          } else {
            const float a = acc[DOWN ? 0 : mt][DOWN ? 0 : nt][i];
            out[o] = bf16_rne(a);
            if (outf) outf[o] = a;
          }
        }
      }
  if (bad) atomicOr(flag, bad);
  if (!ntact) break;
  }
}

// ------------------------------------------------------------------------------------------------ SwiGLU, combine
// gu: BF16 [rows, 2I] (gate in columns 0..I-1, up in I..2I-1); w: FP32 [rows]; silu: FP32 bits [65536]; out: BF16 [rows, I]
__global__ void k_swiglu(const uint16_t* __restrict__ gu, const float* __restrict__ w, const uint32_t* __restrict__ silu,
                         uint16_t* __restrict__ out, int rows, int I) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * I) return;
  size_t r = i / I, c = i - r * I;
  float gate = __uint_as_float((uint32_t)gu[r * 2 * I + c] << 16), up = __uint_as_float((uint32_t)gu[r * 2 * I + I + c] << 16);
  float u = fminf(fmaxf(up, -10.f), 10.f), g = fminf(gate, 10.f);
  float sg = __uint_as_float(silu[__float_as_uint(g) >> 16]);
  float h = __fmul_rn(__fmul_rn(sg, u), w[r]);
  out[i] = bf16_rne(h);
}

// r3f: k_swiglu's program over the first min(*nact, rows) rows; fixed grid, threads stride over the elements
__global__ void k_swiglu_n(const uint16_t* __restrict__ gu, const float* __restrict__ w, const uint32_t* __restrict__ silu,
                           uint16_t* __restrict__ out, int rows, int I, const long long* __restrict__ nact) {
  long long na = *nact;
  if (na > rows) na = rows;
  if (na <= 0) return;
  const size_t total = (size_t)na * I, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += stride) {
    size_t r = i / I, c = i - r * I;
    float gate = __uint_as_float((uint32_t)gu[r * 2 * I + c] << 16), up = __uint_as_float((uint32_t)gu[r * 2 * I + I + c] << 16);
    float u = fminf(fmaxf(up, -10.f), 10.f), g = fminf(gate, 10.f);
    float sg = __uint_as_float(silu[__float_as_uint(g) >> 16]);
    float h = __fmul_rn(__fmul_rn(sg, u), w[r]);
    out[i] = bf16_rne(h);
  }
}
// r3f: k_quant's program over the 32-blocks of the first min(*nact, M) rows (KB blocks a row); fixed grid, strided
template <int TC>
__global__ void k_quant_n(const uint16_t* __restrict__ X, void* __restrict__ Qout, int* __restrict__ EA, int M, int KB,
                          const long long* __restrict__ nact) {
  long long na = *nact;
  if (na > M) na = M;
  if (na <= 0) return;
  const size_t nblk = (size_t)na * KB, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t blk = blockIdx.x * (size_t)blockDim.x + threadIdx.x; blk < nblk; blk += stride) {
    const uint4* xs = (const uint4*)(X + blk * 32);
    uint16_t xb[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) { uint4 v = xs[i]; *(uint4*)(xb + 8 * i) = v; }
    quant_block<TC>(xb, Qout, EA, blk);
  }
}

// BF16 -> int64 floor(y * 2^-bL), exact
DEV long long fixed_of(uint32_t yb, int bL, int* flag) {
  uint32_t s = yb >> 15, E = (yb >> 7) & 0xFF, m = yb & 0x7F;
  long long v = 0;
  if (E != 0 || m != 0) {
    long long M = E ? 128 + m : m;
    int ex = E ? (int)E - 134 : -133;
    int sh = ex - bL;
    if (sh >= 0) {
      // r3d (#8, H8b): |v| = M * 2^sh must stay below 2^62 (M < 2^8: sh <= 54); beyond, saturate and flag
      if (sh > 54) { atomicOr(flag, 2); v = (1LL << 62) - 1; } else v = M << sh;
    } else {
      int r = -sh;
      v = s == 0 ? (r >= 62 ? 0 : (M >> r)) : (r >= 62 ? 1 : ((M + (1LL << r) - 1) >> r));
    }
    if (s) v = -v;
  }
  return v;
}
// S[tok[row], c] += F(y[row, c]) by 64-bit integer atomics: exact and independent of order
__global__ void k_scatter_fixed(const uint16_t* __restrict__ y, const long long* __restrict__ tok, long long* __restrict__ S,
                                size_t rows, int N, int bL, int* __restrict__ flag) {
  V3_RESOLVE_BL(flag);
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= rows * N) return;
  size_t r = i / N, c = i - r * N;
  long long v = fixed_of(y[i], bL, flag);
  if (v) atomicAdd((unsigned long long*)&S[(size_t)tok[r] * N + c], (unsigned long long)v);
}
// S[t, c] = sum over the token's rows (CSR rowptr/rows into y, any order) of F(y[row, c]): exact, no atomics
__global__ void k_token_sum_fixed(const uint16_t* __restrict__ y, const int* __restrict__ rowptr, const int* __restrict__ rows,
                                  long long* __restrict__ S, int ntok, int N, int bL, int* __restrict__ flag) {
  V3_RESOLVE_BL(flag);
  const int t = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= ntok || c >= N) return;
  long long acc = 0;
  for (int j = rowptr[t]; j < rowptr[t + 1]; ++j)
    acc = (long long)((unsigned long long)acc + (unsigned long long)fixed_of(y[(size_t)rows[j] * N + c], bL, flag));
  S[(size_t)t * N + c] = acc;
}
__global__ void k_to_fixed(const uint16_t* __restrict__ y, long long* __restrict__ o, size_t n, int bL, int* __restrict__ flag) {
  V3_RESOLVE_BL(flag);
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  o[i] = fixed_of(y[i], bL, flag);
}
__global__ void k_add_fixed(long long* __restrict__ acc, const long long* __restrict__ p, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) acc[i] = (long long)((unsigned long long)acc[i] + (unsigned long long)p[i]);
}
__global__ void k_fixed_to_bf16(const long long* __restrict__ S, uint16_t* __restrict__ out, size_t n, int bL, int* __restrict__ flag) {
  V3_RESOLVE_BL(flag);
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long v = S[i];
  if (v >= (1LL << 62) || v <= -(1LL << 62)) atomicOr(flag, 4);
  float z = __fmul_rn(__ll2float_rn(v), pow2_any(bL));
  out[i] = bf16_rne(z);
}
// r3e (freeze F3 / F6): the same finish per element; smax (one layer's slot) keeps max |S| (warp max, then one 64-bit
// atomicMax per warp); pathw (the all-reduced rank path word) nonzero sets flag 32. Output bits are k_fixed_to_bf16's.
__global__ void k_fixed_to_bf16_m(const long long* __restrict__ S, uint16_t* __restrict__ out, size_t n, int bL,
                                  int* __restrict__ flag, unsigned long long* __restrict__ smax,
                                  const long long* __restrict__ pathw) {
  V3_RESOLVE_BL(flag);
  if (pathw != nullptr && blockIdx.x == 0 && threadIdx.x == 0 && *pathw != 0) atomicOr(flag, 32);
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  unsigned long long a = 0;
  if (i < n) {
    const long long v = S[i];
    if (v >= (1LL << 62) || v <= -(1LL << 62)) atomicOr(flag, 4);
    a = v < 0 ? 0ull - (unsigned long long)v : (unsigned long long)v;
    out[i] = bf16_rne(__fmul_rn(__ll2float_rn(v), pow2_any(bL)));
  }
  if (smax != nullptr) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { const unsigned long long b = __shfl_xor_sync(0xffffffffu, a, o); a = a > b ? a : b; }
    if ((threadIdx.x & 31) == 0 && a != 0) atomicMax(smax, a);
  }
}


// ------------------------------------------------------------------------------------------------ fused decode (f1)
// One MoE layer's decode on one device, T <= 4 tokens x K <= 6 routes, 4 fixed-size launches:
//   1. k_fplan_quant: block 0 builds the task list (one task per expert, up to R = min(T, 4) rows; any order: the bits do
//      not depend on it), the other blocks quantise x and zero the presum.
//   2. k_decode<TC, R, 0> gate/up over the FMAXT fixed task slots (inactive slots return at once).
//   3. k_fswiglu_quant: SwiGLU (table, clamp, route weight) -> BF16 -> the quantiser, per row and 32-block (the same
//      operations as k_swiglu then k_quant, so the same bits).
//   4. k_decode<TC, R, 1> down: int64 on bL, finished rows written to out[tok * K + slot] and/or F(y) added into presum[tok].
// An active route is (token, slot) with 0 <= id < E and weight != 0; other rows of out are left untouched.
constexpr int FMAXT = 24, FMAXR = 4 * FMAXT;
struct FPlan {
  int texp[FMAXT];          // expert of each task slot, -1 = inactive
  int trows[FMAXR];         // gate/up rows (token index into x), -1 = absent
  int trows2[FMAXR];        // down rows (4t + r into the h buffers), -1 = absent
  int tdst[FMAXR];          // route slot tok * K + k
  int ttok[FMAXR];          // token
  float tw[FMAXR];          // route weight
  int ntask, pad[3];
};

template <int TC>
__global__ void k_fplan_quant(const uint16_t* __restrict__ x, const void* __restrict__ ids, int ids_i64, const float* __restrict__ wts,
                              int T, int K, int E, int H, int R, FPlan* __restrict__ plan, void* __restrict__ q, int* __restrict__ ea,
                              long long* __restrict__ presum, int* __restrict__ flag) {
  if (blockIdx.x == 0) {
    __shared__ FPlan P;
    __shared__ int cnt[FMAXT];
    for (int i = threadIdx.x; i < FMAXT; i += blockDim.x) { P.texp[i] = -1; cnt[i] = 0; }
    for (int i = threadIdx.x; i < FMAXR; i += blockDim.x) { P.trows[i] = -1; P.trows2[i] = -1; P.tdst[i] = 0; P.ttok[i] = 0; P.tw[i] = 0.f; }
    __syncthreads();
    if (threadIdx.x == 0) {
      int nt = 0, bad = 0;
      for (int i = 0; i < T * K; ++i) {
        const long long e = ids_i64 ? ((const long long*)ids)[i] : (long long)((const int*)ids)[i];
        const float w = wts[i];
        if (!(e >= 0 && e < E && w != 0.f)) continue;
        int j = 0;
        for (; j < nt; ++j) if (P.texp[j] == (int)e && cnt[j] < R) break;
        if (j == nt) {
          if (nt == FMAXT) { bad |= 8; continue; }
          P.texp[nt++] = (int)e;
        }
        const int r = cnt[j]++, s = 4 * j + r;
        P.trows[s] = i / K; P.trows2[s] = s; P.tdst[s] = i; P.ttok[s] = i / K; P.tw[s] = w;
      }
      P.ntask = nt;
      if (bad) atomicOr(flag, bad);
    }
    __syncthreads();
    const int* src = (const int*)&P;
    int* dst = (int*)plan;
    for (int i = threadIdx.x; i < (int)(sizeof(FPlan) / 4); i += blockDim.x) dst[i] = src[i];
    return;
  }
  const size_t nthr = (size_t)(gridDim.x - 1) * blockDim.x, tid = (size_t)(blockIdx.x - 1) * blockDim.x + threadIdx.x;
  const size_t nblk = (size_t)T * (H / 32);
  for (size_t b = tid; b < nblk; b += nthr) {
    const uint4* xs = (const uint4*)(x + b * 32);
    uint16_t xb[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) *(uint4*)(xb + 8 * i) = xs[i];
    quant_block<TC>(xb, q, ea, b);
  }
  if (presum)
    for (size_t i = tid; i < (size_t)T * H; i += nthr) presum[i] = 0;
}

// rows 4t + r of the gate/up output gu [FMAXR, 2I] -> h quantised into q2 [FMAXR, I], ea2 [FMAXR, I/32]
template <int TC>
__global__ void k_fswiglu_quant(const uint16_t* __restrict__ gu, const FPlan* __restrict__ plan, const uint32_t* __restrict__ silu,
                                int I, void* __restrict__ q2, int* __restrict__ ea2) {
  const int G = I / 32;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= FMAXR * G) return;
  const int s = i / G, b = i - s * G;
  if (plan->trows[s] < 0) return;
  const float w = plan->tw[s];
  const uint16_t* gate = gu + (size_t)s * 2 * I + b * 32;
  const uint16_t* up = gate + I;
  uint16_t hb[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const float gt = __uint_as_float((uint32_t)gate[j] << 16), uu = __uint_as_float((uint32_t)up[j] << 16);
    const float u = fminf(fmaxf(uu, -10.f), 10.f), g = fminf(gt, 10.f);
    const float sg = __uint_as_float(silu[__float_as_uint(g) >> 16]);
    hb[j] = bf16_rne(__fmul_rn(__fmul_rn(sg, u), w));
  }
  quant_block<TC>(hb, q2, ea2, (size_t)s * G + b);
}

// ------------------------------------------------------------------------------------------------ repack (byte permutation)
__global__ void k_repack(const uint8_t* __restrict__ W, const uint8_t* __restrict__ WS, uint8_t* __restrict__ Wr,
                         uint8_t* __restrict__ Sr, int E, int N, int G) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;   // one (e, n, g)
  if (i >= (size_t)E * N * G) return;
  size_t g = i % G, en = i / G, n = en % N, e = en / N;
  size_t grp = e * (N / 32) + n / 32;
  *(uint4*)(Wr + ((grp * G + g) * 32 + (n & 31)) * 16) = *(const uint4*)(W + i * 16);
  Sr[((grp * (G / 4) + g / 4) * 32 + (n & 31)) * 4 + (g & 3)] = WS[i];
}

// ------------------------------------------------------------------------------------------------ r3h: streamed group
// The streamed prefill's landed kt-kernel blocks straight into the v3 layout: v3_his.assemble_bank followed by k_repack,
// composed. A landing (one kt-kernel part of one device slot) holds, per projection p (0 gate, 1 up, 2 down) and expert
// slot e: raw [3][Gslot][wstride] packed FP4 (gate / up: the part's m rows of K/2 bytes; down: K rows of m/2 bytes) and
// raw_s [3][Gslot][sstride] the fp32 power-of-two group scales in the same order. Rows lo..lo+c of the part are this
// rank's rows (gate, up) / columns (down) d..d+c. The scale byte is the fp32's exponent field (bits 23..30), read as
// bytes so no alignment of the scales is assumed.
#define V3_SR_SEGS 8
struct SRSeg {
  const uint8_t* w;
  const uint8_t* s;
  long long wstride, sstride;
  int lo, c, d, m;
};
struct SRTab {
  int nseg;
  SRSeg seg[V3_SR_SEGS];
};
DEV uint8_t sr_code(const uint8_t* p) { return (uint8_t)(((p[3] & 0x7Fu) << 1) | (p[2] >> 7)); }
__global__ void k_stream_repack(SRTab t, uint8_t* __restrict__ W13r, uint8_t* __restrict__ S13r, uint8_t* __restrict__ W2r,
                                uint8_t* __restrict__ S2r, int E, int Gslot, int N, int K) {
  // 32-bit index arithmetic (the entry point refuses a group of 2^31 blocks or more); byte offsets in size_t
  const unsigned G13 = (unsigned)(K / 32), G2 = (unsigned)(N / 32), R13 = 2u * (unsigned)N, R2 = (unsigned)K;
  const unsigned n13 = (unsigned)E * R13 * G13, n2 = (unsigned)E * R2 * G2;
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n13) {   // gate / up: one (e, row n of 2N, block g of K/32), the index k_repack gives the checkpoint bank
    const unsigned g = i % G13, en = i / G13, n = en % R13, e = en / R13;
    const unsigned pi = n >= (unsigned)N ? 1u : 0u;
    const int r = (int)(n - pi * (unsigned)N);
    for (int k = 0; k < t.nseg; ++k) {
      const SRSeg& sg = t.seg[k];
      if (r < sg.d || r >= sg.d + sg.c) continue;
      const size_t row = (size_t)(sg.lo + r - sg.d), ex = (size_t)(pi * (unsigned)Gslot + e);
      const unsigned grp = e * (R13 / 32) + n / 32;
      *(uint4*)(W13r + ((size_t)(grp * G13 + g) * 32 + (n & 31)) * 16) =
          *(const uint4*)(sg.w + ex * (size_t)sg.wstride + row * (size_t)(K / 2) + (size_t)g * 16);
      S13r[((size_t)(grp * (G13 / 4) + g / 4) * 32 + (n & 31)) * 4 + (g & 3)] =
          sr_code(sg.s + ex * (size_t)sg.sstride + 4 * (row * (size_t)G13 + g));
      return;
    }
    return;
  }
  i -= n13;
  if (i >= n2) return;
  {   // down: one (e, row n of K, block g of N/32)
    const unsigned g = i % G2, en = i / G2, n = en % R2, e = en / R2;
    const int col = (int)(g * 32);
    for (int k = 0; k < t.nseg; ++k) {
      const SRSeg& sg = t.seg[k];
      if (col < sg.d || col >= sg.d + sg.c) continue;
      const size_t sb = (size_t)((unsigned)(sg.lo / 32) + g - (unsigned)(sg.d / 32)), ex = (size_t)(2u * (unsigned)Gslot + e);
      const unsigned grp = e * (R2 / 32) + n / 32;
      *(uint4*)(W2r + ((size_t)(grp * G2 + g) * 32 + (n & 31)) * 16) =
          *(const uint4*)(sg.w + ex * (size_t)sg.wstride + (size_t)n * (size_t)(sg.m / 2) + sb * 16);
      S2r[((size_t)(grp * (G2 / 4) + g / 4) * 32 + (n & 31)) * 4 + (g & 3)] =
          sr_code(sg.s + ex * (size_t)sg.sstride + 4 * ((size_t)n * (size_t)(sg.m / 32) + sb));
      return;
    }
  }
}
__global__ void k_stream_repack_b(SRTab t, uint8_t* __restrict__ W13r, uint8_t* __restrict__ S13r, uint8_t* __restrict__ W2r,
                                uint8_t* __restrict__ S2r, int E, int Gslot, int N, int K) {
  // r3i: r3h's k_stream_repack for byte scales; the only change is the scale read, landing byte j for fp32 j
  // 32-bit index arithmetic (the entry point refuses a group of 2^31 blocks or more); byte offsets in size_t
  const unsigned G13 = (unsigned)(K / 32), G2 = (unsigned)(N / 32), R13 = 2u * (unsigned)N, R2 = (unsigned)K;
  const unsigned n13 = (unsigned)E * R13 * G13, n2 = (unsigned)E * R2 * G2;
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n13) {   // gate / up: one (e, row n of 2N, block g of K/32), the index k_repack gives the checkpoint bank
    const unsigned g = i % G13, en = i / G13, n = en % R13, e = en / R13;
    const unsigned pi = n >= (unsigned)N ? 1u : 0u;
    const int r = (int)(n - pi * (unsigned)N);
    for (int k = 0; k < t.nseg; ++k) {
      const SRSeg& sg = t.seg[k];
      if (r < sg.d || r >= sg.d + sg.c) continue;
      const size_t row = (size_t)(sg.lo + r - sg.d), ex = (size_t)(pi * (unsigned)Gslot + e);
      const unsigned grp = e * (R13 / 32) + n / 32;
      *(uint4*)(W13r + ((size_t)(grp * G13 + g) * 32 + (n & 31)) * 16) =
          *(const uint4*)(sg.w + ex * (size_t)sg.wstride + row * (size_t)(K / 2) + (size_t)g * 16);
      S13r[((size_t)(grp * (G13 / 4) + g / 4) * 32 + (n & 31)) * 4 + (g & 3)] =
          sg.s[ex * (size_t)sg.sstride + (row * (size_t)G13 + g)];
      return;
    }
    return;
  }
  i -= n13;
  if (i >= n2) return;
  {   // down: one (e, row n of K, block g of N/32)
    const unsigned g = i % G2, en = i / G2, n = en % R2, e = en / R2;
    const int col = (int)(g * 32);
    for (int k = 0; k < t.nseg; ++k) {
      const SRSeg& sg = t.seg[k];
      if (col < sg.d || col >= sg.d + sg.c) continue;
      const size_t sb = (size_t)((unsigned)(sg.lo / 32) + g - (unsigned)(sg.d / 32)), ex = (size_t)(2u * (unsigned)Gslot + e);
      const unsigned grp = e * (R2 / 32) + n / 32;
      *(uint4*)(W2r + ((size_t)(grp * G2 + g) * 32 + (n & 31)) * 16) =
          *(const uint4*)(sg.w + ex * (size_t)sg.wstride + (size_t)n * (size_t)(sg.m / 2) + sb * 16);
      S2r[((size_t)(grp * (G2 / 4) + g / 4) * 32 + (n & 31)) * 4 + (g & 3)] =
          sg.s[ex * (size_t)sg.sstride + ((size_t)n * (size_t)(sg.m / 32) + sb)];
      return;
    }
  }
}
// the group's local route ids: (base <= id < base + cnt) ? id - base : -1
__global__ void k_local_ids(const int* __restrict__ ids, int* __restrict__ out, size_t n, int base, int cnt) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int v = ids[i];
  out[i] = (v >= base && v < base + cnt) ? v - base : -1;
}

// ------------------------------------------------------------------------------------------------ C entry points
#define CK_RET do { cudaError_t e_ = cudaGetLastError(); return (int)e_; } while (0)
extern "C" {
int v3_repack(const void* W, const void* WS, void* Wr, void* Sr, int E, int N, int K, void* stream) {
  if (N % 32 || K % 128) return -1;
  size_t n = (size_t)E * N * (K / 32);
  k_repack<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint8_t*)W, (const uint8_t*)WS, (uint8_t*)Wr, (uint8_t*)Sr, E, N, K / 32);
  CK_RET;
}
int v3_quant(const void* x, void* q, int* ea, int M, int K, int tierC, void* stream) {
  int nblk = M * (K / 32);
  dim3 g((nblk + 127) / 128);
  if (tierC) k_quant<1><<<g, 128, 0, (cudaStream_t)stream>>>((const uint16_t*)x, q, ea, nblk);
  else k_quant<0><<<g, 128, 0, (cudaStream_t)stream>>>((const uint16_t*)x, q, ea, nblk);
  CK_RET;
}
int v3_decode(const void* q, const int* ea, const void* W, const void* WS, const int* texp, const int* trows, int T, int R, int N,
              int K, int tierC, int down, int bL, int gb, int ge, void* out, float* outf, long long* outS, int* flag, void* stream) {
  if (K % 128 || N % 32 || R < 1 || R > 4 || gb < 0 || ge > K / 32 || gb >= ge || (!down && (gb != 0 || ge != K / 32))) return -1;
  size_t sh = (size_t)R * K * (tierC ? 2 : 1) + (size_t)R * (K / 32) * 4;
  dim3 g((N + 255) / 256, T);
  cudaStream_t s = (cudaStream_t)stream;
#define DL(TC, RR, DN) k_decode<TC, RR, DN><<<g, 256, sh, s>>>(q, ea, (const uint8_t*)W, (const uint8_t*)WS, texp, trows, N, K, bL, gb, ge, (uint16_t*)out, outf, outS, flag)
#define DR(TC, DN) do { if (R == 1) DL(TC, 1, DN); else if (R == 2) DL(TC, 2, DN); else if (R == 3) DL(TC, 3, DN); else DL(TC, 4, DN); } while (0)
  if (tierC) { if (down) DR(1, 1); else DR(1, 0); }
  else { if (down) DR(0, 1); else DR(0, 0); }
#undef DR
#undef DL
  CK_RET;
}
// down: splits = device array of nsplit+1 block boundaries; outS (zeroed by the caller) receives the exact int64 partials by
// atomics when given (then call v3_fixed_to_bf16); without outS, nsplit must be 1 (in-place finish). Gate/up: splits ignored.
static int prefill_impl(const void* q, const int* ea, const void* W, const void* WS, const int* sorted_rows, const int* tile_exp,
               const int* tile_row0, const int* tile_nrows, int ntiles, int N, int K, int tierC, int down, int bL, const int* splits,
               int nsplit, void* out, float* outf, long long* outS, int* flag, void* stream, const int* otok,
               int nty = 0, const int* ntact = nullptr) {
  const int BN = down ? (tierC ? 16 * NTD : 16 * NTD_D) : 64;
  if (K % 128 || N % BN || (down && (!splits || nsplit < 1 || (nsplit > 1 && !outS)))) return -1;
  if (ntact && (nty < 1 || ntiles < 1)) return -1;
  dim3 g(N / BN, ntact ? nty : ntiles, down ? nsplit : 1);   // r3f: with ntact, nty blocks walk the active tiles
  cudaStream_t s = (cudaStream_t)stream;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(k_prefill<1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(PSmem<2, 64>));
    cudaFuncSetAttribute(k_prefill<0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(PSmem<1, 64>));
    cudaFuncSetAttribute(k_prefill<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(PSmem<2, 16 * NTD>));
    cudaFuncSetAttribute(k_prefill<0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(PSmem<1, 16 * NTD_D>));
    attr_set = true;
  }
#define PL(TC, DN, NPP, BB) k_prefill<TC, DN><<<g, 128, sizeof(PSmem<NPP, BB>), s>>>(q, ea, (const uint8_t*)W, (const uint8_t*)WS, sorted_rows, tile_exp, tile_row0, tile_nrows, N, K, bL, splits, (uint16_t*)out, outf, outS, flag, otok, ntact, ntiles)
  if (tierC) { if (down) PL(1, 1, 2, 16 * NTD); else PL(1, 0, 2, 64); }
  else { if (down) PL(0, 1, 1, 16 * NTD_D); else PL(0, 0, 1, 64); }
#undef PL
  CK_RET;
}
int v3_set_bl(int bL, void* stream) {
  k_set_bl<<<1, 1, 0, (cudaStream_t)stream>>>(bL);
  CK_RET;
}
// r3d (#7): stage slot k's grid on `stream` (the streamer's transfer stream, after the slot's weight copies)
int v3_stage_bl(int k, int bL, void* stream) {
  if (k < 0 || k >= V3_BL_SLOTS || bL < -126 || bL > 0) return 3;
  k_stage_bl<<<1, 1, 0, (cudaStream_t)stream>>>(k, bL);
  CK_RET;
}
int v3_bl_slot_base() { return V3_BL_SLOT0; }
int v3_swiglu(const void* gu, const float* w, const void* silu, void* out, int rows, int I, void* stream) {
  size_t n = (size_t)rows * I;
  k_swiglu<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint16_t*)gu, w, (const uint32_t*)silu, (uint16_t*)out, rows, I);
  CK_RET;
}
// r3f: the same SwiGLU over the first min(*nact, rows) rows (nact: device int64), on a fixed grid of nblocks blocks
int v3_swiglu_n(const void* gu, const float* w, const void* silu, void* out, int rows, int I, const long long* nact,
                int nblocks, void* stream) {
  if (!nact || nblocks < 1 || rows < 0) return -1;
  k_swiglu_n<<<(unsigned)nblocks, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)gu, w, (const uint32_t*)silu, (uint16_t*)out, rows, I, nact);
  CK_RET;
}
// r3f: the same quantiser over the first min(*nact, M) rows (nact: device int64), on a fixed grid of nblocks blocks
int v3_quant_n(const void* x, void* q, int* ea, int M, int K, int tierC, const long long* nact, int nblocks, void* stream) {
  if (!nact || nblocks < 1 || M < 0 || K % 32) return -1;
  if (tierC) k_quant_n<1><<<(unsigned)nblocks, 128, 0, (cudaStream_t)stream>>>((const uint16_t*)x, q, ea, M, K / 32, nact);
  else k_quant_n<0><<<(unsigned)nblocks, 128, 0, (cudaStream_t)stream>>>((const uint16_t*)x, q, ea, M, K / 32, nact);
  CK_RET;
}
int v3_to_fixed(const void* y, long long* o, long long n, int bL, int* flag, void* stream) {
  k_to_fixed<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint16_t*)y, o, (size_t)n, bL, flag);
  CK_RET;
}
int v3_scatter_fixed(const void* y, const long long* tok, long long* S, long long rows, int N, int bL, int* flag, void* stream) {
  size_t n = (size_t)rows * N;
  k_scatter_fixed<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint16_t*)y, tok, S, (size_t)rows, N, bL, flag);
  CK_RET;
}
int v3_token_sum_fixed(const void* y, const int* rowptr, const int* rows, long long* S, int ntok, int N, int bL, int* flag, void* stream) {
  dim3 g((N + 255) / 256, ntok);
  k_token_sum_fixed<<<g, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)y, rowptr, rows, S, ntok, N, bL, flag);
  CK_RET;
}
int v3_add_fixed(long long* acc, const long long* p, long long n, void* stream) {
  k_add_fixed<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(acc, p, (size_t)n);
  CK_RET;
}
int v3_fixed_to_bf16(const long long* S, void* out, long long n, int bL, int* flag, void* stream) {
  k_fixed_to_bf16<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(S, (uint16_t*)out, (size_t)n, bL, flag);
  CK_RET;
}
int v3_fixed_to_bf16_m(const long long* S, void* out, long long n, int bL, int* flag, unsigned long long* smax,
                       const long long* pathw, void* stream) {
  if (n <= 0) return 0;
  k_fixed_to_bf16_m<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(S, (uint16_t*)out, (size_t)n, bL, flag,
                                                                                  smax, pathw);
  CK_RET;
}
// ---------------------------------------------------------------------------------------- fused decode entry points (f1)
const char* v3_lib_version() { return "f1"; }
long long v3_fused_ws_bytes(int H, int I) {
  size_t b = (sizeof(FPlan) + 255) / 256 * 256;
  b += (size_t)4 * H * 2 + 256;                 // q (tier C worst case)
  b += (size_t)4 * (H / 32) * 4 + 256;          // ea
  b += (size_t)FMAXR * 2 * I * 2 + 256;         // gu
  b += (size_t)FMAXR * I * 2 + 256;             // q2
  b += (size_t)FMAXR * (I / 32) * 4 + 256;      // ea2
  return (long long)b;
}
// x BF16 [T, H]; ids [T, K] (int32 or int64), wts FP32 [T, K]; repacked W13 [E, 2I, H/2], S13, W2 [E, H, I/2], S2;
// out BF16 [T, K, H] (route rows written for active routes, others untouched) and/or presum int64 [T, H] (zeroed here,
// then sum over this device's active routes of F(y) on grid bL). silu: FP32 bits [65536]. ws: v3_fused_ws_bytes bytes.
static int fused_decode_impl(const void* x, const void* ids, int ids_i64, const float* wts, int T, int K, int E, const void* W13,
                    const void* S13, const void* W2, const void* S2, int H, int I, int tierC, int bL, const void* silu, void* ws,
                    void* out, long long* presum, int* flag, void* stream, int praw) {
  if (T < 1 || T > 4 || K < 1 || T * K > FMAXT || H % 128 || I % 128 || (2 * I) % 32 || (!out && !presum)) return -1;
  cudaStream_t s = (cudaStream_t)stream;
  uint8_t* p = (uint8_t*)ws;
  auto take = [&](size_t n) { uint8_t* r = p; p += (n + 255) / 256 * 256; return (void*)r; };
  FPlan* plan = (FPlan*)take(sizeof(FPlan));
  void* q = take((size_t)4 * H * 2);
  int* ea = (int*)take((size_t)4 * (H / 32) * 4);
  uint16_t* gu = (uint16_t*)take((size_t)FMAXR * 2 * I * 2);
  void* q2 = take((size_t)FMAXR * I * 2);
  int* ea2 = (int*)take((size_t)FMAXR * (I / 32) * 4);
  const int R = T < 4 ? T : 4;
  const int qblocks = (T * (H / 32) + 127) / 128 + 1;
  if (tierC) k_fplan_quant<1><<<1 + qblocks, 128, 0, s>>>((const uint16_t*)x, ids, ids_i64, wts, T, K, E, H, R, plan, q, ea, presum, flag);
  else k_fplan_quant<0><<<1 + qblocks, 128, 0, s>>>((const uint16_t*)x, ids, ids_i64, wts, T, K, E, H, R, plan, q, ea, presum, flag);
  const int* texp = plan->texp;
  {   // gate/up
    const int N = 2 * I, KK = H;
    size_t sh = (size_t)R * KK * (tierC ? 2 : 1) + (size_t)R * (KK / 32) * 4;
    dim3 g((N + 255) / 256, FMAXT);
#define FG(TC, RR) k_decode<TC, RR, 0><<<g, 256, sh, s>>>(q, ea, (const uint8_t*)W13, (const uint8_t*)S13, texp, plan->trows, N, KK, 0, 0, KK / 32, gu, nullptr, nullptr, flag)
    if (tierC) { if (R == 1) FG(1, 1); else if (R == 2) FG(1, 2); else if (R == 3) FG(1, 3); else FG(1, 4); }
    else { if (R == 1) FG(0, 1); else if (R == 2) FG(0, 2); else if (R == 3) FG(0, 3); else FG(0, 4); }
#undef FG
  }
  {   // SwiGLU + quantiser
    const int n = FMAXR * (I / 32);
    if (tierC) k_fswiglu_quant<1><<<(n + 127) / 128, 128, 0, s>>>(gu, plan, (const uint32_t*)silu, I, q2, ea2);
    else k_fswiglu_quant<0><<<(n + 127) / 128, 128, 0, s>>>(gu, plan, (const uint32_t*)silu, I, q2, ea2);
  }
  {   // down
    const int N = H, KK = I;
    size_t sh = (size_t)R * KK * (tierC ? 2 : 1) + (size_t)R * (KK / 32) * 4;
    dim3 g((N + 255) / 256, FMAXT);
#define FD(TC, RR) k_decode<TC, RR, 1><<<g, 256, sh, s>>>(q2, ea2, (const uint8_t*)W2, (const uint8_t*)S2, texp, plan->trows2, N, KK, bL, 0, KK / 32, (uint16_t*)out, nullptr, nullptr, flag, plan->tdst, plan->ttok, presum, praw)
    if (tierC) { if (R == 1) FD(1, 1); else if (R == 2) FD(1, 2); else if (R == 3) FD(1, 3); else FD(1, 4); }
    else { if (R == 1) FD(0, 1); else if (R == 2) FD(0, 2); else if (R == 3) FD(0, 3); else FD(0, 4); }
#undef FD
  }
  CK_RET;
}
// the plan alone (for tests): the FPlan struct is copied to plan_out (sizeof = v3_fused_plan_bytes)
long long v3_fused_plan_bytes() { return (long long)sizeof(FPlan); }
int v3_fused_decode(const void* x, const void* ids, int ids_i64, const float* wts, int T, int K, int E, const void* W13,
                    const void* S13, const void* W2, const void* S2, int H, int I, int tierC, int bL, const void* silu, void* ws,
                    void* out, long long* presum, int* flag, void* stream) {
  return fused_decode_impl(x, ids, ids_i64, wts, T, K, E, W13, S13, W2, S2, H, I, tierC, bL, silu, ws, out, presum, flag, stream, 0);
}
// spec r3: presum (int64 [T, H], zeroed here) receives the raw exact int64 down sums of this device's active routes; out
// may be null. The caller adds every other source's partial exactly and finishes once with v3_fixed_to_bf16.
int v3_fused_decode_r3(const void* x, const void* ids, int ids_i64, const float* wts, int T, int K, int E, const void* W13,
                       const void* S13, const void* W2, const void* S2, int H, int I, int tierC, int bL, const void* silu, void* ws,
                       long long* presum, int* flag, void* stream) {
  if (!presum) return -1;
  return fused_decode_impl(x, ids, ids_i64, wts, T, K, E, W13, S13, W2, S2, H, I, tierC, bL, silu, ws, nullptr, presum, flag, stream, 1);
}
int v3_prefill(const void* q, const int* ea, const void* W, const void* WS, const int* sorted_rows, const int* tile_exp,
               const int* tile_row0, const int* tile_nrows, int ntiles, int N, int K, int tierC, int down, int bL, const int* splits,
               int nsplit, void* out, float* outf, long long* outS, int* flag, void* stream) {
  return prefill_impl(q, ea, W, WS, sorted_rows, tile_exp, tile_row0, tile_nrows, ntiles, N, K, tierC, down, bL, splits, nsplit,
                      out, outf, outS, flag, stream, nullptr);
}
// spec r3 prefill down: outS int64 [tokens, N] (zeroed by the caller) receives the raw int64 partials by atomics at the
// token of each sorted row (otok[sorted output row]); any split, any order. Gate/up: use v3_prefill.
int v3_prefill_r3(const void* q, const int* ea, const void* W, const void* WS, const int* sorted_rows, const int* tile_exp,
                  const int* tile_row0, const int* tile_nrows, int ntiles, int N, int K, int tierC, int bL, const int* splits,
                  int nsplit, long long* outS, const int* otok, int* flag, void* stream) {
  if (!outS || !otok) return -1;
  return prefill_impl(q, ea, W, WS, sorted_rows, tile_exp, tile_row0, tile_nrows, ntiles, N, K, tierC, 1, bL, splits, nsplit,
                      nullptr, nullptr, outS, flag, stream, otok);
}
// r3f / r3g: v3_prefill (gate/up) and v3_prefill_r3 (down) with a device-side count of the leading non-empty tiles (ntact:
// device int32, clamped to ntiles): nty blocks in y walk those tiles. Same per-tile program, same bits.
int v3_prefill_n(const void* q, const int* ea, const void* W, const void* WS, const int* sorted_rows, const int* tile_exp,
                 const int* tile_row0, const int* tile_nrows, int ntiles, int N, int K, int tierC, void* out, int* flag,
                 int nty, const int* ntact, void* stream) {
  if (!ntact) return -1;
  return prefill_impl(q, ea, W, WS, sorted_rows, tile_exp, tile_row0, tile_nrows, ntiles, N, K, tierC, 0, 0, nullptr, 1,
                      out, nullptr, nullptr, flag, stream, nullptr, nty, ntact);
}
int v3_prefill_r3_n(const void* q, const int* ea, const void* W, const void* WS, const int* sorted_rows, const int* tile_exp,
                    const int* tile_row0, const int* tile_nrows, int ntiles, int N, int K, int tierC, int bL, const int* splits,
                    int nsplit, long long* outS, const int* otok, int* flag, int nty, const int* ntact, void* stream) {
  if (!outS || !otok || !ntact) return -1;
  return prefill_impl(q, ea, W, WS, sorted_rows, tile_exp, tile_row0, tile_nrows, ntiles, N, K, tierC, 1, bL, splits, nsplit,
                      nullptr, nullptr, outS, flag, stream, otok, nty, ntact);
}
// r3h: the streamed group's landings -> the v3 bank in one kernel (see k_stream_repack). seg4: nseg x (lo, c, d, m); raw,
// raws: nseg landing pointers; wstride, sstride: bytes per expert slot of each landing. Refused (nothing launched) unless
// the segments are on 32-blocks, tile [0, N) exactly, fit their landings, and every weight block is 16-aligned.
int v3_stream_repack(int nseg, const int* seg4, const void* const* raw, const void* const* raws, const long long* wstride,
                     const long long* sstride, int Gslot, int E, int N, int K, void* W13r, void* S13r, void* W2r, void* S2r,
                     void* stream) {
  if (nseg < 1 || nseg > V3_SR_SEGS || !seg4 || !raw || !raws || !wstride || !sstride) return -1;
  if (E < 1 || E > Gslot || N <= 0 || K <= 0 || N % 128 || K % 128 || N / 32 > 4096) return -1;
  if (!W13r || !S13r || !W2r || !S2r || ((size_t)W13r & 15) || ((size_t)W2r & 15)) return -3;
  SRTab t;
  t.nseg = nseg;
  unsigned char cover[4096];
  for (int u = 0; u < N / 32; ++u) cover[u] = 0;
  for (int k = 0; k < nseg; ++k) {
    SRSeg& s = t.seg[k];
    s.lo = seg4[4 * k];
    s.c = seg4[4 * k + 1];
    s.d = seg4[4 * k + 2];
    s.m = seg4[4 * k + 3];
    s.w = (const uint8_t*)raw[k];
    s.s = (const uint8_t*)raws[k];
    s.wstride = wstride[k];
    s.sstride = sstride[k];
    if (s.lo < 0 || s.c <= 0 || s.d < 0 || s.m <= 0 || s.lo % 32 || s.c % 32 || s.d % 32 || s.m % 32) return -2;
    if (s.lo + s.c > s.m || s.d + s.c > N) return -2;
    if (!s.w || !s.s || ((size_t)s.w & 15) || (s.wstride & 15)) return -3;
    if (s.wstride < (long long)s.m * (K / 2) || s.sstride < 4LL * s.m * (K / 32)) return -3;
    for (int u = s.d / 32; u < (s.d + s.c) / 32; ++u) {
      if (cover[u]) return -4;
      cover[u] = 1;
    }
  }
  for (int u = 0; u < N / 32; ++u)
    if (!cover[u]) return -4;
  for (int k = nseg; k < V3_SR_SEGS; ++k) t.seg[k] = t.seg[0];
  const size_t n = (size_t)E * (2 * (size_t)N * (size_t)(K / 32) + (size_t)K * (size_t)(N / 32));
  if (n >= ((size_t)1 << 31)) return -1;   // the kernel indexes blocks in 32 bits
  k_stream_repack<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(t, (uint8_t*)W13r, (uint8_t*)S13r, (uint8_t*)W2r,
                                                                               (uint8_t*)S2r, E, Gslot, N, K);
  CK_RET;
}
int v3_local_ids(const int* ids, int* out, long long n, int base, int cnt, void* stream) {
  if (!ids || !out || n < 0) return -1;
  if (n == 0) return 0;
  k_local_ids<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(ids, out, (size_t)n, base, cnt);
  CK_RET;
}
// r3i: v3_stream_repack for byte scales (KT_ARENA_SCALE_BYTES=1): each landing's scale arena holds one E8M0 byte a
// group scale (sstride bytes per expert slot); everything else as v3_stream_repack.
int v3_stream_repack_b(int nseg, const int* seg4, const void* const* raw, const void* const* raws, const long long* wstride,
                     const long long* sstride, int Gslot, int E, int N, int K, void* W13r, void* S13r, void* W2r, void* S2r,
                     void* stream) {
  if (nseg < 1 || nseg > V3_SR_SEGS || !seg4 || !raw || !raws || !wstride || !sstride) return -1;
  if (E < 1 || E > Gslot || N <= 0 || K <= 0 || N % 128 || K % 128 || N / 32 > 4096) return -1;
  if (!W13r || !S13r || !W2r || !S2r || ((size_t)W13r & 15) || ((size_t)W2r & 15)) return -3;
  SRTab t;
  t.nseg = nseg;
  unsigned char cover[4096];
  for (int u = 0; u < N / 32; ++u) cover[u] = 0;
  for (int k = 0; k < nseg; ++k) {
    SRSeg& s = t.seg[k];
    s.lo = seg4[4 * k];
    s.c = seg4[4 * k + 1];
    s.d = seg4[4 * k + 2];
    s.m = seg4[4 * k + 3];
    s.w = (const uint8_t*)raw[k];
    s.s = (const uint8_t*)raws[k];
    s.wstride = wstride[k];
    s.sstride = sstride[k];
    if (s.lo < 0 || s.c <= 0 || s.d < 0 || s.m <= 0 || s.lo % 32 || s.c % 32 || s.d % 32 || s.m % 32) return -2;
    if (s.lo + s.c > s.m || s.d + s.c > N) return -2;
    if (!s.w || !s.s || ((size_t)s.w & 15) || (s.wstride & 15)) return -3;
    if (s.wstride < (long long)s.m * (K / 2) || s.sstride < 1LL * s.m * (K / 32)) return -3;
    for (int u = s.d / 32; u < (s.d + s.c) / 32; ++u) {
      if (cover[u]) return -4;
      cover[u] = 1;
    }
  }
  for (int u = 0; u < N / 32; ++u)
    if (!cover[u]) return -4;
  for (int k = nseg; k < V3_SR_SEGS; ++k) t.seg[k] = t.seg[0];
  const size_t n = (size_t)E * (2 * (size_t)N * (size_t)(K / 32) + (size_t)K * (size_t)(N / 32));
  if (n >= ((size_t)1 << 31)) return -1;   // the kernel indexes blocks in 32 bits
  k_stream_repack_b<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(t, (uint8_t*)W13r, (uint8_t*)S13r, (uint8_t*)W2r,
                                                                               (uint8_t*)S2r, E, Gslot, N, K);
  CK_RET;
}
const char* v3_lib_version_r3() { return "r3i"; }
}
