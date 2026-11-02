// wo_a_parts.cu - the grouped FP8 wo_a projection (fixed_wo_a_fp8._parts) as a CUDA kernel, decode rows only (Claude, 1 Nov 2026).
// route3_decode.cu with a group axis and the epilogue of _parts. Same per-element sequence as the Triton kernel:
//
//   K cut into 32-wide blocks t = 0..T-1, SPLIT_K fixed partitions, split s owns [s*ceil(T/S), min((s+1)*ceil(T/S), T)).
//   Per block one mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 from a zero accumulator, k in natural order.
//   Per block, per element: acc = add.rn.f32(acc, mul.rn.f32(mul.rn.f32(D, sa), sb)), no .ftz, no fma (PTX of _parts, 1 Nov 2026).
//   Each split's accumulator starts at +0 and is stored to P [S, M, G, N] float32; the Triton _merge is unchanged.
//
// Layouts as _rows passes them: A [M*G, K] e4m3, row m group g at (m*G+g); As [M*G, K/32] fp32; B [G, N, K] e4m3;
// Bs [G, N/32, K/32] fp32; P [S, M, G, N] fp32, all contiguous. CTA (x, y, z) = (N block, split, group). The arithmetic does
// not assume power-of-two scales, so unlike route 3 it carries no scale check, as _parts carries none.
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef WO_A_BPC
#define WO_A_BPC 8
#endif
#ifndef WO_A_STAGES
#define WO_A_STAGES 3
#endif

namespace wo_a {

constexpr int BLOCK_K = 32, SPLIT_K = 4, TILE_M = 16, SCALE_GROUP_N = 32;
constexpr int WARPS = 2, THREADS = 32 * WARPS, COLS_PER_WARP = 16, BN = WARPS * COLS_PER_WARP;
constexpr int BPC = WO_A_BPC, CHUNK_BYTES = BPC * BLOCK_K, CHUNKS16 = CHUNK_BYTES / 16, STAGES = WO_A_STAGES;

inline __host__ __device__ int tiles_total(int K) { return (K + BLOCK_K - 1) / BLOCK_K; }
inline __host__ __device__ int tiles_per_split(int K) { return (tiles_total(K) + SPLIT_K - 1) / SPLIT_K; }
inline __device__ int swz(int row, int chunk16) { return chunk16 ^ (row & 7); }

struct __align__(16) Stage {
  uint8_t A[TILE_M][CHUNK_BYTES];
  uint8_t B[WARPS][COLS_PER_WARP][CHUNK_BYTES];
  float   As[TILE_M][BPC];
  float   Bs[BPC];
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
__device__ __forceinline__ void mma_k32_zero_c(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
        "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}
// The pinned epilogue of _parts
__device__ __forceinline__ float step(float acc, float d, float sa, float sb) {
  return __fadd_rn(acc, __fmul_rn(__fmul_rn(d, sa), sb));
}

__device__ __forceinline__ void load_chunk(Stage* st, const uint8_t* __restrict__ A, const float* __restrict__ As,
                                           const uint8_t* __restrict__ B, const float* __restrict__ Bs,
                                           int M, int NG, int N, int K, int ldS, int group, int col_base,
                                           int t_first, int nb, int tid) {
  const int kbyte0 = t_first * BLOCK_K;
  const size_t b_base = static_cast<size_t>(group) * N;
  for (int i = tid; i < TILE_M * CHUNKS16; i += THREADS) {
    const int row = i / CHUNKS16, ch = i % CHUNKS16;
    const bool live = (row < M) && (ch < 2 * nb);
    const uint8_t* src = A + (static_cast<size_t>(live ? row : 0) * NG + group) * K + kbyte0 + (live ? ch * 16 : 0);
    cp_async16(smem_u32(&st->A[row][swz(row, ch) * 16]), src, live ? 16 : 0);
  }
  for (int i = tid; i < WARPS * COLS_PER_WARP * CHUNKS16; i += THREADS) {
    const int w = i / (COLS_PER_WARP * CHUNKS16);
    const int r = (i / CHUNKS16) % COLS_PER_WARP, ch = i % CHUNKS16;
    const bool live = (ch < 2 * nb);
    const int col = col_base + w * COLS_PER_WARP + r;
    const uint8_t* src = B + (b_base + col) * K + kbyte0 + (live ? ch * 16 : 0);
    cp_async16(smem_u32(&st->B[w][r][swz(r, ch) * 16]), src, live ? 16 : 0);
  }
  for (int i = tid; i < TILE_M * BPC; i += THREADS) {
    const int row = i / BPC, b = i % BPC;
    const bool live = (row < M) && (b < nb);
    const float* src = As + (static_cast<size_t>(live ? row : 0) * NG + group) * ldS + t_first + (live ? b : 0);
    cp_async4(smem_u32(&st->As[row][b]), src, live ? 4 : 0);
  }
  if (tid < BPC) {
    const bool live = (tid < nb);
    const size_t row = static_cast<size_t>(group) * (N / SCALE_GROUP_N) + col_base / SCALE_GROUP_N;
    const float* src = Bs + row * ldS + t_first + (live ? tid : 0);
    cp_async4(smem_u32(&st->Bs[tid]), src, live ? 4 : 0);
  }
}

__global__ void __launch_bounds__(THREADS)
wo_a_parts_kernel(const uint8_t* __restrict__ A, const float* __restrict__ As,
                  const uint8_t* __restrict__ B, const float* __restrict__ Bs,
                  float* __restrict__ P, int M, int NG, int N, int K, int ldS) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Stage* stages = reinterpret_cast<Stage*>(smem_raw);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tig = lane & 3;
  const int col_base = blockIdx.x * BN, split = blockIdx.y, group = blockIdx.z;

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

#pragma unroll
  for (int c = 0; c < STAGES - 1; ++c) {
    if (c < nchunk)
      load_chunk(&stages[c], A, As, B, Bs, M, NG, N, K, ldS, group, col_base, t0 + c * BPC, min(BPC, nblk - c * BPC), tid);
    cp_commit();
  }

  // ldmatrix lanes as route 3: A (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15, k 16-31);
  // B (cols 0-7, k 0-15), (cols 0-7, k 16-31), (cols 8-15, k 0-15), (cols 8-15, k 16-31)
  const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1), a_ch = (lane >> 4) & 1;
  const int b_row = (lane & 7) + 8 * ((lane >> 4) & 1), b_ch = (lane >> 3) & 1;

  for (int c = 0; c < nchunk; ++c) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int cn = c + STAGES - 1;
      if (cn < nchunk)
        load_chunk(&stages[cn % STAGES], A, As, B, Bs, M, NG, N, K, ldS, group, col_base, t0 + cn * BPC, min(BPC, nblk - cn * BPC), tid);
      cp_commit();
    }
    const Stage& st = stages[c % STAGES];
    const int nb = min(BPC, nblk - c * BPC);

#pragma unroll
    for (int b = 0; b < BPC; ++b) {
      if (b < nb) {
        uint32_t a[4], bb[4];
        ldmatrix_x4(smem_u32(&st.A[a_row][swz(a_row, 2 * b + a_ch) * 16]), a[0], a[1], a[2], a[3]);
        ldmatrix_x4(smem_u32(&st.B[warp][b_row][swz(b_row, 2 * b + b_ch) * 16]), bb[0], bb[1], bb[2], bb[3]);
        float d0[4], d1[4];
        mma_k32_zero_c(d0, a, bb[0], bb[1]);
        mma_k32_zero_c(d1, a, bb[2], bb[3]);
        const float as_lo = st.As[g][b], as_hi = st.As[g + 8][b], bs = st.Bs[b];
        acc[0][0] = step(acc[0][0], d0[0], as_lo, bs);
        acc[0][1] = step(acc[0][1], d0[1], as_lo, bs);
        acc[0][2] = step(acc[0][2], d0[2], as_hi, bs);
        acc[0][3] = step(acc[0][3], d0[3], as_hi, bs);
        acc[1][0] = step(acc[1][0], d1[0], as_lo, bs);
        acc[1][1] = step(acc[1][1], d1[1], as_lo, bs);
        acc[1][2] = step(acc[1][2], d1[2], as_hi, bs);
        acc[1][3] = step(acc[1][3], d1[3], as_hi, bs);
      }
    }
  }

  // C fragment: c0,c1 row g; c2,c3 row g+8; columns tig*2, +1
  const int col0 = col_base + warp * COLS_PER_WARP;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int col = col0 + 8 * j + tig * 2;
    if (g < M) {
      float* out = P + ((static_cast<size_t>(split) * M + g) * NG + group) * N + col;
      out[0] = acc[j][0];
      out[1] = acc[j][1];
    }
    if (g + 8 < M) {
      float* out = P + ((static_cast<size_t>(split) * M + g + 8) * NG + group) * N + col;
      out[0] = acc[j][2];
      out[1] = acc[j][3];
    }
  }
}

}  // namespace wo_a

static void wo_a_parts(torch::Tensor A, torch::Tensor As, torch::Tensor B, torch::Tensor Bs, torch::Tensor P) {
  using namespace wo_a;
  TORCH_CHECK(A.is_cuda() && As.is_cuda() && B.is_cuda() && Bs.is_cuda() && P.is_cuda(), "wo_a_parts: CUDA tensors required");
  TORCH_CHECK(A.scalar_type() == at::kFloat8_e4m3fn && B.scalar_type() == at::kFloat8_e4m3fn, "wo_a_parts: A and B must be float8_e4m3fn");
  TORCH_CHECK(As.scalar_type() == at::kFloat && Bs.scalar_type() == at::kFloat && P.scalar_type() == at::kFloat, "wo_a_parts: scales and partials must be float32");
  TORCH_CHECK(A.is_contiguous() && As.is_contiguous() && B.is_contiguous() && Bs.is_contiguous() && P.is_contiguous(), "wo_a_parts: contiguous tensors required");
  TORCH_CHECK(A.dim() == 2 && As.dim() == 2 && B.dim() == 3 && Bs.dim() == 3 && P.dim() == 4, "wo_a_parts: ranks");
  const int NG = static_cast<int>(B.size(0)), N = static_cast<int>(B.size(1)), K = static_cast<int>(B.size(2));
  const int M = static_cast<int>(A.size(0)) / NG;
  TORCH_CHECK(M >= 1 && M <= TILE_M && A.size(0) == M * NG && A.size(1) == K, "wo_a_parts: A must be [M*G, K] with 1 <= M <= 16");
  TORCH_CHECK(K % BLOCK_K == 0 && N % BN == 0, "wo_a_parts: K % 32 == 0 and N % 32 == 0");
  TORCH_CHECK(As.size(0) == M * NG && As.size(1) == K / BLOCK_K, "wo_a_parts: As must be [M*G, K/32]");
  TORCH_CHECK(Bs.size(0) == NG && Bs.size(1) == N / SCALE_GROUP_N && Bs.size(2) == K / BLOCK_K, "wo_a_parts: Bs must be [G, N/32, K/32]");
  TORCH_CHECK(P.size(0) == SPLIT_K && P.size(1) == M && P.size(2) == NG && P.size(3) == N, "wo_a_parts: P must be [4, M, G, N]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(A.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(B.data_ptr()) % 16 == 0, "wo_a_parts: A and B must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(As.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(Bs.data_ptr()) % 4 == 0, "wo_a_parts: scales must be 4-byte aligned");
  const c10::cuda::CUDAGuard guard(A.device());
  const dim3 grid(N / BN, SPLIT_K, NG), block(THREADS);
  wo_a_parts_kernel<<<grid, block, sizeof(Stage) * STAGES, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(A.data_ptr()), As.data_ptr<float>(),
      reinterpret_cast<const uint8_t*>(B.data_ptr()), Bs.data_ptr<float>(), P.data_ptr<float>(), M, NG, N, K, K / BLOCK_K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static int64_t wo_a_smem_bytes() { return static_cast<int64_t>(sizeof(wo_a::Stage) * wo_a::STAGES); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("parts", &wo_a_parts, "Grouped FP8 wo_a split partials (M<=16): A, As, B, Bs, P[4,M,G,N]");
  m.def("smem_bytes", &wo_a_smem_bytes, "dynamic shared memory per CTA");
}
