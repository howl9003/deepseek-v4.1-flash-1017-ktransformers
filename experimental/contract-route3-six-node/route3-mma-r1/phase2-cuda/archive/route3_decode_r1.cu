// route3_decode.cu - Route 3 phase 2: dense FP8 decode kernel (M <= 16) reproducing the accepted prefill kernel's
// per-row sequence bit for bit (see route3_sequence.h), at weight-bandwidth speed.
//
// Why the Triton geometries lost (ROUTE3_PHASE1_RESULTS_FOR_CLAUDE.md, ROUTE3_COMPILER_ANALYSIS_r1.json): no spills,
// native QMMA, enough bytes in flight, but per 32-wide K step seven block barriers and about two dozen shared loads
// around two MMAs. This kernel amortises that: one cp.async group and one block barrier per 8 K blocks, one
// ldmatrix.x4 for the A fragment and one for the B fragment per block per warp, two MMAs, eight FMUL/FFMA.
//
// Contract (must not change without re-acceptance): one zero-C m16n8k32 e4m3 MMA per 32-wide K block in natural k
// order; epilogue t = mul.rn.f32(D, As[m,t]); acc = fma.rn.f32(t, Bs[n/32,t], acc); SPLIT_K = 4 fixed partitions;
// partials [4, M, N] float32 reduced by the unchanged Triton _reduce_block_fp8_split_k.
//
// Layouts (as the Triton call site passes them): A [M, K] e4m3 contiguous; As [M, K/32] fp32 contiguous;
// B [N, K] e4m3 contiguous (k innermost); Bs [N/32, K/32] fp32 contiguous; P [4, M, N] fp32 contiguous.
// Requirements: 1 <= M <= 16, K % 32 == 0, N % 32 == 0. Scales must be exact powers of two (validated in-kernel,
// error flag readable with route3_error_flag()).
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "route3_sequence.h"

namespace route3 {

__device__ unsigned int g_error = 0u;   // bit 0: invalid activation scale, bit 1: invalid weight scale

struct __align__(16) Stage {
  uint8_t A[TILE_M][CHUNK_BYTES];                  // 4096 B, 16-byte chunks swizzled by row
  uint8_t B[WARPS][COLS_PER_WARP][CHUNK_BYTES];    // 8192 B
  float   As[TILE_M][BPC];                         // 512 B
  float   Bs[BPC];                                 // 32 B
};
static_assert(sizeof(Stage) % 16 == 0, "stage must stay 16-byte aligned");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int PENDING> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// One k32 e4m3 MMA from a zero accumulator: the pinned per-block instruction.
__device__ __forceinline__ void mma_k32_zero_c(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
        "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}
// Exact power of two: a normal float32 with a zero mantissa, or a subnormal with exactly one mantissa bit.
__device__ __forceinline__ bool pow2_scale(float s) {
  const uint32_t u = __float_as_uint(s);
  const uint32_t e = (u >> 23) & 0xFFu, m = u & 0x7FFFFFu;
  if (u & 0x80000000u) return false;
  if (e == 0xFFu) return false;
  if (e == 0u) return __popc(m) == 1;
  return m == 0u;
}

// Issue the asynchronous copies of one chunk (blocks t_first .. t_first+nb-1) into a stage.
__device__ __forceinline__ void load_chunk(Stage* st, const uint8_t* __restrict__ A, const float* __restrict__ As,
                                           const uint8_t* __restrict__ B, const float* __restrict__ Bs,
                                           int M, int K, int ldAs, int ldBs, int nblock_row, int col_base,
                                           int t_first, int nb, int tid) {
  const int kbyte0 = t_first * BLOCK_K;
  // A rows (rows >= M and chunks beyond the live blocks are zero-filled)
  for (int i = tid; i < TILE_M * CHUNKS16; i += THREADS) {
    const int row = i / CHUNKS16, ch = i % CHUNKS16;
    const bool live = (row < M) && (ch < 2 * nb);
    const uint8_t* src = A + static_cast<size_t>(live ? row : 0) * K + kbyte0 + (live ? ch * 16 : 0);
    cp_async16(smem_u32(&st->A[row][swz(row, ch) * 16]), src, live ? 16 : 0);
  }
  // B rows: WARPS x 16 output columns, k innermost
  for (int i = tid; i < WARPS * COLS_PER_WARP * CHUNKS16; i += THREADS) {
    const int w = i / (COLS_PER_WARP * CHUNKS16);
    const int r = (i / CHUNKS16) % COLS_PER_WARP, ch = i % CHUNKS16;
    const bool live = (ch < 2 * nb);
    const int col = col_base + w * COLS_PER_WARP + r;
    const uint8_t* src = B + static_cast<size_t>(col) * K + kbyte0 + (live ? ch * 16 : 0);
    cp_async16(smem_u32(&st->B[w][r][swz(r, ch) * 16]), src, live ? 16 : 0);
  }
  // activation scales [row][block] and the CTA's weight scales [block]
  for (int i = tid; i < TILE_M * BPC; i += THREADS) {
    const int row = i / BPC, b = i % BPC;
    const bool live = (row < M) && (b < nb);
    const float* src = As + static_cast<size_t>(live ? row : 0) * ldAs + t_first + (live ? b : 0);
    cp_async4(smem_u32(&st->As[row][b]), src, live ? 4 : 0);
  }
  if (tid < BPC) {
    const bool live = (tid < nb);
    const float* src = Bs + static_cast<size_t>(nblock_row) * ldBs + t_first + (live ? tid : 0);
    cp_async4(smem_u32(&st->Bs[tid]), src, live ? 4 : 0);
  }
}

__global__ void __launch_bounds__(THREADS)
route3_decode_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As,
                     const uint8_t* __restrict__ B, const float* __restrict__ Bs,
                     float* __restrict__ P, int M, int N, int K, int ldAs, int ldBs) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Stage* stages = reinterpret_cast<Stage*>(smem_raw);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tig = lane & 3;
  const int split = blockIdx.y;
  const int col_base = blockIdx.x * BN;
  const int nblock_row = col_base / SCALE_GROUP_N;          // BN == 32: one weight-scale block per CTA

  const int T = tiles_total(K), tps = tiles_per_split(K);
  const int t0 = split * tps;
  const int t1 = min(t0 + tps, T);
  const int nblk = max(t1 - t0, 0);
  const int nchunk = (nblk + BPC - 1) / BPC;

  float acc[2][4];
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

  // prologue
#pragma unroll
  for (int c = 0; c < STAGES - 1; ++c) {
    if (c < nchunk)
      load_chunk(&stages[c], A, As, B, Bs, M, K, ldAs, ldBs, nblock_row, col_base, t0 + c * BPC,
                 min(BPC, nblk - c * BPC), tid);
    cp_commit();
  }

  // ldmatrix lane addressing. A tiles: (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15,
  // k 16-31) -> registers a0-3, a4-7, a8-11, a12-15. B tiles: (cols 0-7, k 0-15), (cols 0-7, k 16-31),
  // (cols 8-15, k 0-15), (cols 8-15, k 16-31) -> b0/b1 of n-tile 0, b0/b1 of n-tile 1.
  const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_ch = (lane >> 4) & 1;
  const int b_row = (lane & 7) + 8 * ((lane >> 4) & 1), b_ch = (lane >> 3) & 1;

  for (int c = 0; c < nchunk; ++c) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int cn = c + STAGES - 1;
      if (cn < nchunk)
        load_chunk(&stages[cn % STAGES], A, As, B, Bs, M, K, ldAs, ldBs, nblock_row, col_base, t0 + cn * BPC,
                   min(BPC, nblk - cn * BPC), tid);
      cp_commit();
    }
    const Stage& st = stages[c % STAGES];
    const int nb = min(BPC, nblk - c * BPC);

    // scale validation for this chunk (cheap; failures set the flag, never change the arithmetic)
    for (int i = tid; i < TILE_M * BPC; i += THREADS) {
      const int row = i / BPC, b = i % BPC;
      if (row < M && b < nb && !pow2_scale(st.As[row][b])) atomicOr(&g_error, 1u);
    }
    if (tid < nb && !pow2_scale(st.Bs[tid])) atomicOr(&g_error, 2u);

#pragma unroll
    for (int b = 0; b < BPC; ++b) {
      if (b < nb) {
        uint32_t a[4], bb[4];
        ldmatrix_x4(smem_u32(&st.A[a_row][swz(a_row, 2 * b + a_ch) * 16]), a[0], a[1], a[2], a[3]);
        ldmatrix_x4(smem_u32(&st.B[warp][b_row][swz(b_row, 2 * b + b_ch) * 16]), bb[0], bb[1], bb[2], bb[3]);
        float d0[4], d1[4];
        mma_k32_zero_c(d0, a, bb[0], bb[1]);
        mma_k32_zero_c(d1, a, bb[2], bb[3]);
        const float as_lo = st.As[g][b], as_hi = st.As[g + 8][b];
        const float bs = st.Bs[b];
        // pinned epilogue: t = mul.rn(D, As); acc = fma.rn(t, Bs, acc)
        acc[0][0] = __fmaf_rn(__fmul_rn(d0[0], as_lo), bs, acc[0][0]);
        acc[0][1] = __fmaf_rn(__fmul_rn(d0[1], as_lo), bs, acc[0][1]);
        acc[0][2] = __fmaf_rn(__fmul_rn(d0[2], as_hi), bs, acc[0][2]);
        acc[0][3] = __fmaf_rn(__fmul_rn(d0[3], as_hi), bs, acc[0][3]);
        acc[1][0] = __fmaf_rn(__fmul_rn(d1[0], as_lo), bs, acc[1][0]);
        acc[1][1] = __fmaf_rn(__fmul_rn(d1[1], as_lo), bs, acc[1][1]);
        acc[1][2] = __fmaf_rn(__fmul_rn(d1[2], as_hi), bs, acc[1][2]);
        acc[1][3] = __fmaf_rn(__fmul_rn(d1[3], as_hi), bs, acc[1][3]);
      }
    }
  }

  // store this split's partials for the live rows (C fragment: c0,c1 row g; c2,c3 row g+8; cols tig*2, +1)
  float* Pq = P + static_cast<size_t>(split) * M * N;
  const int col0 = col_base + warp * COLS_PER_WARP;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int col = col0 + 8 * j + tig * 2;
    if (g < M) {
      Pq[static_cast<size_t>(g) * N + col] = acc[j][0];
      Pq[static_cast<size_t>(g) * N + col + 1] = acc[j][1];
    }
    if (g + 8 < M) {
      Pq[static_cast<size_t>(g + 8) * N + col] = acc[j][2];
      Pq[static_cast<size_t>(g + 8) * N + col + 1] = acc[j][3];
    }
  }
}

}  // namespace route3

// ------------------------------------------------------------------------------------------ torch binding

static void route3_decode_fp8(torch::Tensor A, torch::Tensor As, torch::Tensor B, torch::Tensor Bs, torch::Tensor P) {
  using namespace route3;
  TORCH_CHECK(A.is_cuda() && As.is_cuda() && B.is_cuda() && Bs.is_cuda() && P.is_cuda(), "route3: CUDA tensors required");
  TORCH_CHECK(A.scalar_type() == at::kFloat8_e4m3fn && B.scalar_type() == at::kFloat8_e4m3fn, "route3: A and B must be float8_e4m3fn");
  TORCH_CHECK(As.scalar_type() == at::kFloat && Bs.scalar_type() == at::kFloat && P.scalar_type() == at::kFloat, "route3: scales and partials must be float32");
  TORCH_CHECK(A.is_contiguous() && As.is_contiguous() && B.is_contiguous() && Bs.is_contiguous() && P.is_contiguous(), "route3: contiguous tensors required");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && As.dim() == 2 && Bs.dim() == 2 && P.dim() == 3, "route3: ranks");
  const int M = static_cast<int>(A.size(0)), K = static_cast<int>(A.size(1)), N = static_cast<int>(B.size(0));
  TORCH_CHECK(B.size(1) == K, "route3: K mismatch");
  TORCH_CHECK(M >= 1 && M <= TILE_M, "route3: 1 <= M <= 16 (larger batches take the unchanged prefill kernel)");
  TORCH_CHECK(K % BLOCK_K == 0 && N % BN == 0, "route3: K % 32 == 0 and N % 32 == 0");
  TORCH_CHECK(As.size(0) == M && As.size(1) == K / BLOCK_K, "route3: As must be [M, K/32]");
  TORCH_CHECK(Bs.size(0) == N / SCALE_GROUP_N && Bs.size(1) == K / BLOCK_K, "route3: Bs must be [N/32, K/32]");
  TORCH_CHECK(P.size(0) == SPLIT_K && P.size(1) == M && P.size(2) == N, "route3: P must be [4, M, N]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(A.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(B.data_ptr()) % 16 == 0,
              "route3: A and B must be 16-byte aligned (cp.async 16-byte copies)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(As.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(Bs.data_ptr()) % 4 == 0,
              "route3: scale tensors must be 4-byte aligned");
  const c10::cuda::CUDAGuard guard(A.device());
  const dim3 grid(N / BN, SPLIT_K), block(THREADS);
  const size_t smem = sizeof(Stage) * STAGES;
  route3_decode_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(A.data_ptr()), As.data_ptr<float>(),
      reinterpret_cast<const uint8_t*>(B.data_ptr()), Bs.data_ptr<float>(), P.data_ptr<float>(),
      M, N, K, K / BLOCK_K, K / BLOCK_K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Read (and optionally clear) the scale-validation flag. Synchronous; call between runs, never inside a capture.
static int64_t route3_error_flag(bool reset) {
  unsigned int v = 0u;
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(&v, route3::g_error, sizeof(v)));
  if (reset) {
    const unsigned int z = 0u;
    C10_CUDA_CHECK(cudaMemcpyToSymbol(route3::g_error, &z, sizeof(z)));
  }
  return static_cast<int64_t>(v);
}

static int64_t route3_smem_bytes() { return static_cast<int64_t>(sizeof(route3::Stage) * route3::STAGES); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_fp8", &route3_decode_fp8, "Route 3 dense FP8 decode partials (M<=16): A, As, B, Bs, P[4,M,N]");
  m.def("error_flag", &route3_error_flag, "scale validation flag (bit0 As, bit1 Bs); reset=True clears");
  m.def("smem_bytes", &route3_smem_bytes, "dynamic shared memory per CTA");
}
