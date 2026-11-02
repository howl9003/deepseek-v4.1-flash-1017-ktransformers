"""Isolated exact DP4A projection: one real route, no padded tensor-core rows.

PTX specification: https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp4a
Three signed byte planes and groupwise int64 shifts preserve contract v2.
No serving dispatch imports this experiment.
"""
import triton as tr
import triton.language as tl
from .kernels import MARGIN


@tr.jit
def _pack_weights(packed):
    # Generic PRMT selectors are four nibbles. A selector's high bit asks
    # for sign replication. Both tables have nonnegative bytes, so that
    # mode supplies zero for the unwanted sign without a branch or load.
    positive = tl.inline_asm_elementwise(
        "prmt.b32 $0, 0x03020100, 0x0c080604, $1;",
        constraints="=r,r",args=[packed],dtype=tl.int32,is_pure=True,pack=1)
    negative7 = tl.inline_asm_elementwise(
        "prmt.b32 $0, 0x7d7e7f00, 0x74787a7c, $1;",
        constraints="=r,r",args=[packed ^ 0x8888],dtype=tl.int32,is_pure=True,pack=1)
    # Negative nonzero entries116..127 have bit6 set; positive entries0..12
    # do not. Restore bit7 without cross-byte addition/carry. Both zeros
    # decode to zero, including the negative-zero nibble.
    return positive | negative7 | ((negative7 & 0x40404040) << 1)


@tr.jit
def _dot4(a, b):
    return tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',
        constraints='=r,r,r', args=[a, b], dtype=tl.int32, is_pure=True, pack=1)


@tr.jit
def _projection(A0, A1, A2, EAG, EAT, W, WS, EW, IDS, Y,
                COUNT: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                TOPK: tl.constexpr, FIRST: tl.constexpr,
                BN: tl.constexpr, SPLITS: tl.constexpr, BG: tl.constexpr):
    route = tl.program_id(0).to(tl.int64)
    eid = tl.load(IDS + route).to(tl.int64)
    split = tl.program_id(2)
    if eid >= 0:
        row = route // TOPK if FIRST else route
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        G: tl.constexpr = K // 32
        per_split: tl.constexpr = (G + SPLITS - 1) // SPLITS
        packed_index = tl.arange(0, BG * 8)
        local_group = packed_index // 8
        group = split * per_split + local_group
        valid_k = (local_group < per_split) & (group < G)
        k4 = group * 8 + (packed_index & 7)
        # W has been viewed as int16: each load contains four FP4 nibbles.
        packed = tl.load(W + eid * N * (K // 4) + cols[:, None].to(tl.int64) * (K // 4) + k4[None, :],
                         (cols[:, None] < N) & valid_k[None, :], 0).to(tl.int32) & 65535
        wb = _pack_weights(packed)
        a0 = tl.load(A0 + row * (K // 4) + k4, valid_k, 0)
        a1 = tl.load(A1 + row * (K // 4) + k4, valid_k, 0)
        a2 = tl.load(A2 + row * (K // 4) + k4, valid_k, 0)
        v0 = _dot4(a0[None, :], wb)
        v1 = _dot4(a1[None, :], wb)
        v2 = _dot4(a2[None, :], wb)
        value = (v2 << 12) + (v1 << 6) + v0
        # Every group dot fits int32: 32 * 229376 * 12 < 2**27.
        dot = tl.sum(tl.reshape(value, (BN, BG, 8)), 2).to(tl.int64)
        groups = split * per_split + tl.arange(0, BG)
        valid_g = (tl.arange(0, BG) < per_split) & (groups < G)
        eat = tl.load(EAT + row).to(tl.int32)
        eag = tl.load(EAG + row * G + groups, valid_g, 0).to(tl.int32)
        ewc = tl.load(EW + eid * N + cols, cols < N, 0).to(tl.int32)
        wsg = tl.load(WS + eid * N * G + cols[:, None].to(tl.int64) * G + groups[None, :],
                      (cols[:, None] < N) & valid_g[None, :], 127).to(tl.int32)
        shift = eag[None, :] - eat + (wsg - 127 - ewc[:, None]) + MARGIN
        scaled = tl.where(shift >= 0, dot << tl.maximum(shift, 0), dot >> tl.minimum(0 - shift, 63))
        total = tl.sum(tl.where(valid_g[None, :], scaled, 0), 1)
        tl.store(Y + (split.to(tl.int64) * COUNT + route) * N + cols, total, cols < N)


def project(planes, ea, eat, weight, scales, ew, active, partials,
            count, n, k, topk, first, splits, bn=8):
    import torch
    bg = tr.next_power_of_2(tr.cdiv(k // 32, splits))
    _projection[(count, tr.cdiv(n, bn), splits)](
        *[p.view(torch.int32) for p in planes], ea, eat, weight.view(torch.int16), scales,
        ew, active, partials, count, n, k, topk, first, bn, splits, bg,
        enable_fp_fusion=False, num_warps=4)
