// Contract v3 CPU kernels, arena / row-major input mode: row-wise decode (concept-r1/CPU_KERNEL_COMPUTE_r1.md).
// Same C ABI and the same bits as v3_rm.h's gate / up and down entry points; faster because the 8 x 8 register
// transpose of every (panel, block pair) is gone. Each weight row is decoded where it lies (low nibbles = even k,
// high nibbles = odd k), multiplied by activations that the call de-interleaves once, and the 8 rows' block dots are
// reduced to the served layout (one int32 per column) by horizontal adds. The epilogues are v3_common.inc's own, so
// the FP32 chain order, its rounding and the int64 sums are those of v3_rm.cpp for every activation the quantiser can
// produce (tier D |q| <= 127; at q = -128, which the quantiser never emits, v3_rm.cpp's sign trick is wrong and this
// file is not).
// Off unless KT_V3_RW=1 (read once): with it off, or for tier C, or for a down call with a block list, every entry
// point is the v3_rm one.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "v3_rm.h"
#ifdef __cplusplus
extern "C" {
#endif
int v3_rw_enabled(void);
void v3_rw_set_enabled(int on);   // tests; overrides the environment
// pathb-cpu-r1: the reduction, 1 (default: the 3bb076a kernel) or 2 (KT_V3_RW_RED=2: no vphadd, activations as memory
// operands, the down shift spread computed once a call). Same bits either way; read only while KT_V3_RW is on.
int v3_rw_red(void);
void v3_rw_set_red(int red);      // tests; overrides the environment
int v3_rw_gemm_bf16(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                    int n1, uint16_t* out, int ldo);
int v3_rw_gemm_acc(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                   int n1, float* acc, int ldacc);
int v3_rw_down_partial(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M,
                       int bL, const int* blocks, int nb, int n0, int n1, int64_t* S, int ldS);
#ifdef __cplusplus
}
#endif
// A translation unit that includes this header after v3_rm.h calls the row-wise entry points through the old names.
#ifndef V3_RW_NO_REDIRECT
#define v3_rm_gemm_bf16 v3_rw_gemm_bf16
#define v3_rm_gemm_acc v3_rw_gemm_acc
#define v3_rm_down_partial v3_rw_down_partial
#endif
