"""Experimental fixed-order FP8 family. Isolated; no serving dispatcher change.

Each output has the same tree of 32 products, activation then weight scale,
tree of eight groups, then increasing 256-column chunks at every M. Row and
output blocking affect reuse only. No tensor-core padding or split scratch.
This is a NEW arithmetic order, not a bit-identical old-kernel replacement.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sum32(v, RM: tl.constexpr, BN: tl.constexpr):
    # Explicit adjacent-pair tree; do not delegate association to a reduction.
    a, b = tl.split(tl.reshape(v, (RM, BN, 8, 16, 2)))
    a, b = tl.split(tl.reshape(a + b, (RM, BN, 8, 8, 2)))
    a, b = tl.split(tl.reshape(a + b, (RM, BN, 8, 4, 2)))
    a, b = tl.split(tl.reshape(a + b, (RM, BN, 8, 2, 2)))
    a, b = tl.split(tl.reshape(a + b, (RM, BN, 8, 2)))
    return a + b


@triton.jit
def _kernel(A, B, AS, BS, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SAR: tl.constexpr, SAG: tl.constexpr, SBR: tl.constexpr,
            SBG: tl.constexpr, RM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * RM + tl.arange(0, RM)
    outs = tl.program_id(1) * BN + tl.arange(0, BN)
    off = tl.arange(0, 256)
    acc = tl.full((RM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, 256)):
        kk = start * 256 + off
        bv = tl.load(B + outs[:, None] * K + kk[None, :],
                     (outs[:, None] < N) & (kk[None, :] < K), other=0.0).to(tl.float32)
        groups = start * 8 + tl.arange(0, 8)
        sb = tl.load(BS + (outs[:, None] // 32) * SBR + groups[None, :] * SBG,
                     (outs[:, None] < N) & (groups[None, :] < K // 32), other=0)
        # Keep one weight tile live and reuse it across rows, instead of
        # materializing the RM x BN x K product volume simultaneously.
        for mi in tl.static_range(RM):
            row = tl.program_id(0) * RM + mi
            av = tl.load(A + row * K + kk, (row < M) & (kk < K), other=0.0).to(tl.float32)
            prod = bv * av[None, :]
            dots = tl.reshape(_sum32(tl.reshape(prod, (1, BN, 8, 32)), 1, BN), (BN, 8))
            sa = tl.load(AS + row * SAR + groups * SAG,
                         (row < M) & (groups < K // 32), other=0)
            scaled = (dots * sa[None, :]) * sb
            a, b = tl.split(tl.reshape(scaled, (BN, 4, 2)))
            a, b = tl.split(tl.reshape(a + b, (BN, 2, 2)))
            a, b = tl.split(tl.reshape(a + b, (BN, 2)))
            acc = tl.where(tl.arange(0, RM)[:, None] == mi, acc + (a + b)[None, :], acc)
    tl.store(Y + rows[:, None] * N + outs[None, :], acc,
             (rows[:, None] < M) & (outs[None, :] < N))


def linear(a, b, a_scale, b_scale, *, output_dtype=torch.bfloat16,
           out=None, rows_per_program=None, outputs_per_program=16):
    """Already-quantized 32x32 block FP8 projection; quantization timed separately.

    Only the actual checkpoint's floating scale layout is supported. Explicitly
    reject unsupported layouts rather than silently reinterpret packed scales.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('expected compatible 2D activation and weight tensors')
    m, k = a.shape; n = b.shape[0]
    if k == 0 or k % 32 or n == 0:
        raise ValueError('K must be a positive multiple of32 and N positive')
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise TypeError('expected E4M3FN inputs')
    if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        raise TypeError('expected floating block scales')
    if a_scale.shape != (m, k // 32) or b_scale.shape != ((n + 31) // 32, k // 32):
        raise ValueError('expected32-column activation and32x32weight scales')
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError('expected contiguous activations and weights')
    if not a.is_cuda or any(t.device != a.device for t in (b, a_scale, b_scale)):
        raise ValueError('expected one CUDA device')
    if output_dtype not in (torch.bfloat16, torch.float32, torch.float16):
        raise TypeError('unsupported output dtype')
    rm = (1 if m <= 4 else 8) if rows_per_program is None else rows_per_program
    bn = outputs_per_program
    if rm not in (1, 2, 4, 8) or bn not in (1, 2, 4, 8, 16, 32):
        raise ValueError('unsupported tile')
    if out is not None and (out.shape != (m,n) or out.dtype != output_dtype or out.device != a.device or not out.is_contiguous()):
        raise ValueError('invalid output buffer')
    y = torch.empty((m, n), dtype=output_dtype, device=a.device) if out is None else out
    if m:
        _kernel[(triton.cdiv(m, rm), triton.cdiv(n, bn))](
            a, b, a_scale, b_scale, y, m, n, k,
            *a_scale.stride(), *b_scale.stride(), rm, bn,
            num_warps=4, enable_fp_fusion=False)
    return y
