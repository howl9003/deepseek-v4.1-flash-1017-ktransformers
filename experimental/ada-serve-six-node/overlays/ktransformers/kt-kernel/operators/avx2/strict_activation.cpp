// operators/avx2/strict_activation.cpp
// The AVX2 SwiGLU activation compiled without fast-math: IEEE division, no vrcpps rewrite,
// no multiply/add contraction. Used by KT_MOE_FP32_INTERMEDIATES (moe_base.hpp
// apply_activation_fp32). The vector arithmetic is avx2::act_fn unchanged; only the code
// generation differs from the legacy translation unit, which the project builds with
// -ffast-math (CMakeLists.txt line 172).
//
// CMake (add next to the other avx2 sources of the extension target):
//   set_source_files_properties(operators/avx2/strict_activation.cpp PROPERTIES
//       COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-reciprocal-math")
// Build guard: objdump of this object contains vdivps and no vrcpps.
#include <immintrin.h>
#include "avx2_bf16_utils.hpp"

__attribute__((noinline)) __m256 kt_strict_act_fn(__m256 gate, __m256 up, float swiglu_limit, float swiglu_alpha) {
  return avx2::act_fn(gate, up, swiglu_limit, swiglu_alpha);
}
