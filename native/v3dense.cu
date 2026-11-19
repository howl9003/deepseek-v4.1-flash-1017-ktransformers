// v3dense r3: batch-invariant block-FP8 linear for SM120 (fixed row geometry).
//
// Per-row contract (identical for every M, every kernel, every tile shape; unchanged from r1/r2):
//   q, ea   = reference act_quant of the BF16 row (block 32, UE8M0 scale, amax floor 1e-4; kernel.py:41-92 form)
//   for each output column n, with weight codes W[n, :] (E4M3) and UE8M0 scale exponents eb[n/32, g]:
//     partitions p = 0..P-1 cover k32 blocks [b_p, b_{p+1}), b_p = (p * G) / P, G = K / 32
//     acc_p = +0;  for g in partition p ascending:  acc_p = fmaf(QMMA_k32(q[g], W[n, g]; C = 0), 2^(ea[g] + eb[n/32, g]), acc_p)
//     total = acc_0;  for p = 1..P-1:  total = total + acc_p            (sequential, IEEE RNE, no FMA)
//     out = bf16_rne(total)   (or FP32 total for row-parallel partials)
//   P is a pure function of (N, K, batch) chosen by the caller (v3dense.py), never of the device.
//
// r3 weight layout ("frag"): the weights are repacked once at load (v3dense.pack_frag, lossless, in place) so that
// the 16-column x 32-k tile (c, g) is 512 contiguous bytes, tile (c, g) at byte ((c * G) + g) * 512, and lane
// l = 4 * gq + t holds bytes 16 l .. 16 l + 15 = [row 16c+gq, k 32g+8t .. 32g+8t+7][row 16c+gq+8, same k]. Every lane
// then issues one coalesced 16-byte load per tile, and the bytes land in the m16n8k32 A fragment exactly as before:
// a = {v.x, v.z, v.y, v.w}. The instruction inputs per output element are unchanged, so the bits are unchanged.
// The prefill kernel reads the same layout (its shared-memory fill unpacks the tile into the old row-major staging).
//
// Both kernels issue mma.sync.m16n8k32 E4M3xE4M3 with the weights as A (16 output columns) and the activations as B
// (8 tokens). QMMA from C = 0 is row/column independent and deterministic (Q1d, D2).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xcompiler -fPIC -shared
//        -o libv3dense.so v3dense.cu       (no --use_fast_math, no -ftz)
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define V3D_MAXP 16

__device__ __forceinline__ int fast_log2_ceil(float x) {
  uint32_t b = __float_as_uint(x);
  int ex = (b >> 23) & 0xFF;
  uint32_t man = b & 0x7FFFFF;
  return ex - 127 + (man != 0 ? 1 : 0);
}

__device__ __forceinline__ float pow2i(int e) {
  if (e >= -126) return __int_as_float((e + 127) << 23);
  if (e >= -149) return __int_as_float(1u << (e + 149));
  return 0.0f;
}

__device__ __forceinline__ void mma_e4m3(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}

// streaming 16-byte weight load: read-only, no L1 allocation, 256-byte L2 prefetch
__device__ __forceinline__ uint4 ld_stream16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// programmatic dependent launch (PDL): the quantiser lets the GEMM grid start early; the GEMM waits for the quantiser's
// completion (and memory flush) right before it first reads Q / EA. Both are no-ops when no PDL edge exists.
// coherent loads of data written by the PDL predecessor (Q, EA): asm volatile, so they are never hoisted above
// griddepcontrol.wait (an __ldg / ld.global.nc may be scheduled across it and read stale bytes)
__device__ __forceinline__ uint2 ld_dep8(const void* p) {
  uint2 v;
  asm volatile("ld.global.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_dep16(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int ld_dep4(const int* p) {
  int v;
  asm volatile("ld.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

// P is a power of two (2^lgP), so the partition bound is a shift: no runtime integer division, whose GPU expansion
// uses an F2I.FTZ instruction.
__device__ __forceinline__ int part_lo(int p, int G, int lgP) { return (p * G) >> lgP; }

// ------------------------------------------------------------------ activation quantiser (reference act_quant)
// X: bf16 [M, K] with row stride ldx (elements); Q: e4m3 [M, K] (row stride K); EA: int32 [M, K/32]
__global__ void k_v3d_quant(const uint16_t* __restrict__ X, int ldx, uint8_t* __restrict__ Q, int* __restrict__ EA,
                            int M, int K) {
  pdl_trigger();
  int G = K / 32;
  const int g = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;  // grid (ceil(G / 128), M): no division
  if (g >= G) return;
  const uint16_t* x = X + (long)m * ldx + g * 32;
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    v[j] = __uint_as_float((uint32_t)x[j] << 16);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = fmaxf(amax, 1e-4f);
  int e = fast_log2_ceil(__fmul_rn(amax, 1.0f / 448.0f));
  // x / 2^e == x * 2^-e bit for bit for every x: an exact power-of-two scaling followed by one RNE rounding either way.
  const float inv = __int_as_float((127 - e) << 23);
  EA[(long)m * G + g] = e;
  uint8_t* q = Q + (long)m * K + g * 32;
#pragma unroll
  for (int j = 0; j < 32; j += 2) {
    float y0 = fminf(fmaxf(__fmul_rn(v[j], inv), -448.f), 448.f);
    float y1 = fminf(fmaxf(__fmul_rn(v[j + 1], inv), -448.f), 448.f);
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(y1), "f"(y0));
    *(unsigned short*)(q + j) = r;  // low byte = y0 (element j), high byte = y1 (element j + 1)
  }
}

// ------------------------------------------------------------------ shared argument block
struct V3DArgs {
  const uint8_t* Q;   // activations e4m3, [batch][M][K] with strides
  const int* EA;      // activation exponents, [batch][M][G]
  const uint8_t* W;   // weights e4m3, frag layout [batch][N/16][G][512 B]
  const uint8_t* WS;  // weight scales e8m0 bytes [batch][N/32][G] (or FP32 powers of two)
  void* OUT;          // [batch][M][ldo] bf16 or f32
  int M, N, K, P, out_f32, ldo, ldq, lde;  // ldq/lde: row strides of Q (bytes) and EA (ints)
  int lgP;     // P = 2^lgP
  int ws_f32;  // weight scales as FP32 powers of two (else E8M0 bytes)
  long sQb, sEb, sWb, sSb, sOb;  // per-batch strides (elements of each array)
};

__device__ __forceinline__ int wexp(const V3DArgs& a, const uint8_t* WS, long idx) {
  if (a.ws_f32) return (int)((__float_as_uint(__ldg(((const float*)WS) + idx)) >> 23) & 0xFF) - 127;
  return (int)__ldg(WS + idx) - 127;
}

// ------------------------------------------------------------------ decode kernel (M <= 8)
// grid (N / (16 * T), 1, batch); block = P warps. Warp p computes partition p for T consecutive 16-column tiles.
// U k32 blocks per stage, two stages in registers: stage s+1's weight (and activation) loads are in flight while stage
// s's MMAs and FMAs run.
template <int T>
__global__ void __launch_bounds__(32 * V3D_MAXP) k_v3d_decode(V3DArgs a) {
  constexpr int U = 4 / T;
  extern __shared__ __align__(16) float sm[];  // fold [P][T][4][32] floats
  const int lane = threadIdx.x & 31, p = threadIdx.x >> 5, gq = lane >> 2, t = lane & 3;
  const int G = a.K >> 5, b = blockIdx.z;
  const int ct0 = blockIdx.x * T;
  const uint8_t* Q = a.Q + b * a.sQb;
  const int* EA = a.EA + b * a.sEb;
  const uint4* Wp = (const uint4*)(a.W + b * a.sWb) + lane;
  const uint8_t* WS = a.WS + b * a.sSb * (a.ws_f32 ? 4 : 1);
  const int tok0 = 2 * t, tok1 = 2 * t + 1;
  const bool tokv = gq < a.M;
  const int g0 = part_lo(p, G, a.lgP), g1 = part_lo(p + 1, G, a.lgP);

  float acc[T][4];
#pragma unroll
  for (int i = 0; i < T; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;

  uint4 wA[U][T], wB[U][T];
  int sA[U][T], sB[U][T];
  uint2 xA[U], xB[U];
  int a0A[U], a1A[U], a0B[U], a1B[U];

  auto loadW = [&](uint4 (&w)[U][T], int (&s)[U][T], int gs) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int g = gs + u;
      const bool ok = g < g1;
#pragma unroll
      for (int i = 0; i < T; ++i) {
        const int c = ct0 + i;
        w[u][i] = ok ? ld_stream16(Wp + ((long)c * G + g) * 32) : make_uint4(0u, 0u, 0u, 0u);
        s[u][i] = ok ? wexp(a, WS, (long)(c >> 1) * G + g) : 0;
      }
    }
  };
  auto loadX = [&](uint2 (&x)[U], int (&e0)[U], int (&e1)[U], int gs) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int g = gs + u;
      const bool ok = g < g1;
      x[u] = (ok && tokv) ? ld_dep8(Q + (long)gq * a.ldq + g * 32 + 8 * t) : make_uint2(0u, 0u);
      e0[u] = (ok && tok0 < a.M) ? ld_dep4(EA + (long)tok0 * a.lde + g) : 0;
      e1[u] = (ok && tok1 < a.M) ? ld_dep4(EA + (long)tok1 * a.lde + g) : 0;
    }
  };
  auto compute = [&](const uint4 (&w)[U][T], const int (&s)[U][T], const uint2 (&x)[U], const int (&e0)[U],
                     const int (&e1)[U], int gs) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (gs + u < g1) {
        const uint32_t bf[2] = {x[u].x, x[u].y};
#pragma unroll
        for (int i = 0; i < T; ++i) {
          const uint32_t af[4] = {w[u][i].x, w[u][i].z, w[u][i].y, w[u][i].w};
          float d[4];
          mma_e4m3(d, af, bf);
          const int eb = s[u][i];
          acc[i][0] = __fmaf_rn(d[0], pow2i(e0[u] + eb), acc[i][0]);
          acc[i][1] = __fmaf_rn(d[1], pow2i(e1[u] + eb), acc[i][1]);
          acc[i][2] = __fmaf_rn(d[2], pow2i(e0[u] + eb), acc[i][2]);
          acc[i][3] = __fmaf_rn(d[3], pow2i(e1[u] + eb), acc[i][3]);
        }
      }
    }
  };

  loadW(wA, sA, g0);  // the first weights stream in while the quantiser (PDL predecessor) finishes
  pdl_wait();
  loadX(xA, a0A, a1A, g0);
#pragma unroll 1
  for (int gs = g0; gs < g1; gs += 2 * U) {
    loadW(wB, sB, gs + U);
    loadX(xB, a0B, a1B, gs + U);
    compute(wA, sA, xA, a0A, a1A, gs);
    loadW(wA, sA, gs + 2 * U);
    loadX(xA, a0A, a1A, gs + 2 * U);
    compute(wB, sB, xB, a0B, a1B, gs + U);
  }

#pragma unroll
  for (int i = 0; i < T; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) sm[((p * T + i) * 4 + j) * 32 + lane] = acc[i][j];
  __syncthreads();
  // fold: one thread per (tile i, reg j, lane) in a fixed sequential partition order
  // no unrolling: an unrolled trip count with the runtime stride blockDim.x is an integer division (F2I.FTZ)
#pragma unroll 1
  for (int idx = threadIdx.x; idx < T * 4 * 32; idx += blockDim.x) {
    const int i = idx >> 7, j = (idx >> 5) & 3, ln = idx & 31;
    float tot = sm[((0 * T + i) * 4 + j) * 32 + ln];
#pragma unroll 1
    for (int q = 1; q < a.P; ++q) tot = __fadd_rn(tot, sm[((q * T + i) * 4 + j) * 32 + ln]);
    const int g2 = ln >> 2, t2 = ln & 3;
    const int col = (ct0 + i) * 16 + g2 + ((j & 2) ? 8 : 0);
    const int tok = 2 * t2 + (j & 1);
    if (tok < a.M) {
      long o = b * a.sOb + (long)tok * a.ldo + col;
      if (a.out_f32) ((float*)a.OUT)[o] = tot;
      else ((__nv_bfloat16*)a.OUT)[o] = __float2bfloat16_rn(tot);
    }
  }
}

// ------------------------------------------------------------------ prefill kernel (any M)
// CTA tile: 128 output columns x 64 tokens, 8 warps as 4 (columns) x 2 (tokens); warp tile 32 columns (2 m16) x 32
// tokens (4 n8). K is staged KB k32 blocks at a time through shared memory (row stride 96 bytes: conflict-free 8-byte
// fragment reads). The weights come from the frag layout and are unpacked into the same row-major staging as r1/r2.
// The partition boundaries are applied per k32 block exactly as in the decode kernel.
#define PF_BN 128
#define PF_BM 64
#define PF_KB 2
#define PF_RS (PF_KB * 32 + 32)
__global__ void __launch_bounds__(256) k_v3d_prefill(V3DArgs a) {
  __shared__ __align__(16) uint8_t sW[2][PF_BN * PF_RS];
  __shared__ __align__(16) uint8_t sX[2][PF_BM * PF_RS];
  __shared__ int sEA[2][PF_BM][PF_KB];
  __shared__ int sEB[2][PF_BN / 32][PF_KB];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, t = lane & 3;
  const int wc = warp & 3, wt = warp >> 2;
  const int G = a.K >> 5, b = blockIdx.z;
  const int ncol0 = blockIdx.x * PF_BN, ntok0 = blockIdx.y * PF_BM;
  const uint8_t* Q = a.Q + b * a.sQb;
  const int* EA = a.EA + b * a.sEb;
  const uint4* Wp = (const uint4*)(a.W + b * a.sWb);
  const uint8_t* WS = a.WS + b * a.sSb * (a.ws_f32 ? 4 : 1);
  const int nstage = (G + PF_KB - 1) / PF_KB;

  auto load_stage = [&](int s, int buf) {
    const int gb = s * PF_KB;
    const int kb = min(PF_KB, G - gb);
    // weights: 8 column tiles x KB k32 blocks, 32 lanes of 16 bytes each, unpacked to rows
    for (int c = tid; c < (PF_BN / 16) * PF_KB * 32; c += 256) {
      const int l = c & 31, tk = c >> 5;
      const int ci = tk / PF_KB, k = tk % PF_KB;  // compile-time power-of-two constants: shifts
      const int col = ncol0 + 16 * ci;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (k < kb && col < a.N) v = __ldg(Wp + ((long)(col >> 4) * G + gb + k) * 32 + l);
      const int r = 16 * ci + (l >> 2), t2 = l & 3;
      *(uint2*)(&sW[buf][r * PF_RS + k * 32 + 8 * t2]) = make_uint2(v.x, v.y);
      *(uint2*)(&sW[buf][(r + 8) * PF_RS + k * 32 + 8 * t2]) = make_uint2(v.z, v.w);
    }
    for (int c = tid; c < PF_BM * PF_KB * 2; c += 256) {
      const int r = c / (PF_KB * 2), ch = c % (PF_KB * 2);
      uint4 v = make_uint4(0, 0, 0, 0);
      if (ch < kb * 2 && ntok0 + r < a.M) v = ld_dep16(Q + (long)(ntok0 + r) * a.ldq + gb * 32 + ch * 16);
      *(uint4*)(&sX[buf][r * PF_RS + ch * 16]) = v;
    }
    for (int c = tid; c < PF_BM * PF_KB; c += 256) {
      const int r = c / PF_KB, k = c % PF_KB;
      sEA[buf][r][k] = (k < kb && ntok0 + r < a.M) ? ld_dep4(EA + (long)(ntok0 + r) * a.lde + gb + k) : 0;
    }
    for (int c = tid; c < (PF_BN / 32) * PF_KB; c += 256) {
      const int r = c / PF_KB, k = c % PF_KB;
      const int cb = (ncol0 >> 5) + r;
      sEB[buf][r][k] = (k < kb && cb * 32 < a.N) ? wexp(a, WS, (long)cb * G + gb + k) : 0;
    }
  };

  float cur[2][4][4], tot[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) { cur[i][n][j] = 0.0f; tot[i][n][j] = 0.0f; }
  int p = 0, pend = part_lo(1, G, a.lgP);

  pdl_wait();
  load_stage(0, 0);
  __syncthreads();
  for (int s = 0; s < nstage; ++s) {
    const int buf = s & 1;
    if (s + 1 < nstage) load_stage(s + 1, buf ^ 1);
    const int gb = s * PF_KB, kb = min(PF_KB, G - gb);
    for (int k = 0; k < kb; ++k) {
      const int g = gb + k;
      if (g == pend) {  // close partition p
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              tot[i][n][j] = (p == 0) ? cur[i][n][j] : __fadd_rn(tot[i][n][j], cur[i][n][j]);
              cur[i][n][j] = 0.0f;
            }
        ++p;
        pend = part_lo(p + 1, G, a.lgP);
      }
      uint32_t af[2][4], bf[4][2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = wc * 32 + 16 * i + gq;
        uint2 w0 = *(const uint2*)(&sW[buf][r * PF_RS + k * 32 + 8 * t]);
        uint2 w1 = *(const uint2*)(&sW[buf][(r + 8) * PF_RS + k * 32 + 8 * t]);
        af[i][0] = w0.x; af[i][1] = w1.x; af[i][2] = w0.y; af[i][3] = w1.y;
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int r = wt * 32 + 8 * n + gq;
        uint2 x = *(const uint2*)(&sX[buf][r * PF_RS + k * 32 + 8 * t]);
        bf[n][0] = x.x; bf[n][1] = x.y;
      }
      const int eb = sEB[buf][wc][k];
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int e0 = sEA[buf][wt * 32 + 8 * n + 2 * t][k] + eb;
        const int e1 = sEA[buf][wt * 32 + 8 * n + 2 * t + 1][k] + eb;
        const float s0 = pow2i(e0), s1 = pow2i(e1);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          float d[4];
          mma_e4m3(d, af[i], bf[n]);
          cur[i][n][0] = __fmaf_rn(d[0], s0, cur[i][n][0]);
          cur[i][n][1] = __fmaf_rn(d[1], s1, cur[i][n][1]);
          cur[i][n][2] = __fmaf_rn(d[2], s0, cur[i][n][2]);
          cur[i][n][3] = __fmaf_rn(d[3], s1, cur[i][n][3]);
        }
      }
    }
    __syncthreads();
  }
  // close the last partition
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) tot[i][n][j] = (p == 0) ? cur[i][n][j] : __fadd_rn(tot[i][n][j], cur[i][n][j]);
  // store
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int col = ncol0 + wc * 32 + 16 * i + gq + ((j & 2) ? 8 : 0);
        const int tok = ntok0 + wt * 32 + 8 * n + 2 * t + (j & 1);
        if (col < a.N && tok < a.M) {
          long o = b * a.sOb + (long)tok * a.ldo + col;
          if (a.out_f32) ((float*)a.OUT)[o] = tot[i][n][j];
          else ((__nv_bfloat16*)a.OUT)[o] = __float2bfloat16_rn(tot[i][n][j]);
        }
      }
}

// ------------------------------------------------------------------ row-parallel finish: out = bf16(p0 + p1)
__global__ void k_v3d_add2(const float* __restrict__ A, const float* __restrict__ B, __nv_bfloat16* __restrict__ O,
                           long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) O[i] = __float2bfloat16_rn(__fadd_rn(A[i], B[i]));
}

template <typename K>
static cudaError_t launch_pdl(K kern, dim3 grid, dim3 block, size_t smem, cudaStream_t st, int pdl, const V3DArgs& a) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kern, a);
}

template <int T>
static cudaError_t launch_decode_t(const V3DArgs& a, int batch, cudaStream_t st, int pdl) {
  dim3 grid(a.N / (16 * T), 1, batch), block(32 * a.P);
  size_t smem = (size_t)a.P * T * 4 * 32 * sizeof(float);
  return launch_pdl(k_v3d_decode<T>, grid, block, smem, st, pdl, a);
}

static int g_pdl = 1;  // v3d_set_pdl(0) disables the PDL launch attribute (bits are identical either way)

// ------------------------------------------------------------------ C API
extern "C" {

int v3d_version() { return 3; }

void v3d_set_pdl(int on) { g_pdl = on ? 1 : 0; }

int v3d_quant(const void* X, int ldx, void* Q, void* EA, int M, int K, void* stream) {
  if (K % 32) return 1;
  for (int m0 = 0; m0 < M; m0 += 65535) {  // gridDim.y <= 65535
    const int mc = M - m0 < 65535 ? M - m0 : 65535;
    dim3 grid((K / 32 + 127) / 128, mc);
    k_v3d_quant<<<grid, 128, 0, (cudaStream_t)stream>>>((const uint16_t*)X + (long)m0 * ldx, ldx,
                                                         (uint8_t*)Q + (long)m0 * K, (int*)EA + (long)m0 * (K / 32), mc, K);
  }
  return (int)cudaGetLastError();
}

// W must be in the frag layout (v3dense.pack_frag). mode: 0 = auto (decode for M <= 8), 1 = force decode (M <= 8),
// 2 = force prefill.  tdec: decode column tiles per warp (1, 2 or 4; bit-neutral).
int v3d_gemm(const void* Q, const void* EA, const void* W, const void* WS, void* OUT, int M, int N, int K, int P,
             int out_f32, int ldo, int ldq, int lde, int ws_f32, int batch, long sQb, long sEb, long sWb, long sSb, long sOb, int mode, int tdec,
             void* stream) {
  if (K % 32 || N % 32 || P < 1 || P > V3D_MAXP || P > K / 32 || (P & (P - 1))) return 1;
  int lgP = 0;
  while ((1 << lgP) < P) ++lgP;
  if (M == 0 || batch == 0) return 0;
  if (ldq % 16 || ldq < K || lde < K / 32) return 5;
  if (sWb % 512) return 6;
  V3DArgs a{(const uint8_t*)Q, (const int*)EA, (const uint8_t*)W, (const uint8_t*)WS, OUT, M, N, K, P, out_f32, ldo,
            ldq, lde, lgP, ws_f32, sQb, sEb, sWb, sSb, sOb};
  cudaStream_t st = (cudaStream_t)stream;
  bool dec = (mode == 1) || (mode == 0 && M <= 8);
  cudaError_t e;
  if (dec) {
    if (M > 8) return 2;
    if (tdec != 1 && tdec != 2 && tdec != 4) return 3;
    if (N % (16 * tdec)) return 4;
    if (tdec == 1) e = launch_decode_t<1>(a, batch, st, g_pdl);
    else if (tdec == 2) e = launch_decode_t<2>(a, batch, st, g_pdl);
    else e = launch_decode_t<4>(a, batch, st, g_pdl);
  } else {
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM, batch);
    e = launch_pdl(k_v3d_prefill, grid, dim3(256), 0, st, g_pdl, a);
  }
  if (e != cudaSuccess) return 100 + (int)e;
  return (int)cudaGetLastError();
}

int v3d_add2(const void* A, const void* B, void* O, long n, void* stream) {
  if (n == 0) return 0;
  k_v3d_add2<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const float*)A, (const float*)B,
                                                                            (__nv_bfloat16*)O, n);
  return (int)cudaGetLastError();
}
}
