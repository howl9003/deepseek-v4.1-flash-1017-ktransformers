"""SM89 dual-scope gather in three stages, bit for bit the unsplit kernel (Claude, 1 Nov 2026).

The unsplit kernel walks the top-k in blocks of 64 tokens on 4 CTAs, about 4 of the card's 80 SMs. Its online softmax is order-bearing, but only through the running maximum and the FP32
folds, not through the MMAs. So:
  1. scores: every (head block, token block) on its own CTA: the eight q x k dots, the block's maximum.
  2. probabilities: every (head block, token block) on its own CTA, with the running maximum read back from stage 1: alpha, p, the row sum of p.
  3. output: every (head block, tile) on its own CTA, folding its acc over the token blocks in order, with l and m folded beside it, then the output scale.
Every element goes through the same operations in the same order as the unsplit kernel: the same dots on the same shapes, the same exp2, the same acc * alpha + dot in token-block order.
SWA-only layers have an empty extra scope. The LSE is not produced: the SM89 call site reads only the output.
"""
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.sm89_fp4_tile import FP4_BYTES_PER_TOKEN, FP4_DATA_BYTES, FP4_SCALE_BYTES, fp4_tile
from sglang.kernels.ops.attention.sm89_v41_tile import V41_BYTES_PER_TOKEN, V41_DATA_BYTES, V41_SCALE_BYTES, v41_tile

_KV_FP8_TY = tl.float8e4nv
BLOCK_H, BLOCK_N, NUM_WARPS = 16, 64, 8
_D = 512


@triton.jit
def _block_valid(kv, idx_base, offs_n, n_limit, topk_len, block_size, stride_kv, HAS_LEN: tl.constexpr, KV_FP4: tl.constexpr, KV_V41: tl.constexpr):
    """One token block's addressing, as the unsplit kernel's loop head."""
    mask_n = offs_n < n_limit
    indices = tl.load(idx_base + offs_n, mask=mask_n, other=-1)
    is_invalid = indices == -1
    if HAS_LEN:
        is_invalid = is_invalid | (offs_n >= topk_len)
    valid = mask_n & ~is_invalid
    indices_clamped = tl.maximum(indices, 0)
    block_idx = indices_clamped // block_size
    offset_in_block = indices_clamped % block_size
    kv_block_base = kv + block_idx.to(tl.int64) * tl.cast(stride_kv, tl.int64)
    offset_64 = offset_in_block.to(tl.int64)
    if KV_V41:
        data_off, scale_off = offset_64 * V41_DATA_BYTES, block_size * V41_DATA_BYTES + offset_64 * V41_SCALE_BYTES
    elif KV_FP4:
        data_off, scale_off = offset_64 * FP4_DATA_BYTES, block_size * FP4_DATA_BYTES + offset_64 * FP4_SCALE_BYTES
    else:
        data_off, scale_off = offset_64 * 576, block_size * 576 + offset_64 * 8
    return kv_block_base, data_off, scale_off, valid


@triton.jit
def _tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, tile, TILE_SIZE: tl.constexpr):
    """One FP8 tile, dequantised to BF16 as _process_kv_block_aggressive does."""
    scale_u8 = tl.load(kv_block_base + scale_off + tile, mask=valid, other=127).to(tl.uint8)
    base = kv_block_base[:, None] + data_off[:, None]
    nope_u8 = tl.load(base + tile * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0)
    scale = tl.math.exp2(scale_u8.to(tl.float32) - 127.0).to(tl.bfloat16)
    nope = nope_u8.to(_KV_FP8_TY, bitcast=True)
    kv = (nope.to(tl.bfloat16) * scale[:, None]).to(tl.bfloat16)
    return tl.where(valid_2d, kv, 0.0)


@triton.jit
def _tile_rope(kv_block_base, data_off, valid_2d, offs_tile, D_NOPE: tl.constexpr):
    base = kv_block_base[:, None] + data_off[:, None]
    rope_ptrs = base + D_NOPE + offs_tile[None, :] * 2
    rope_lo = tl.load(rope_ptrs, mask=valid_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
    kv = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
    return tl.where(valid_2d, kv, 0.0)


@triton.jit
def _score_block(q_base, kv, idx_base, n_start, n_limit, topk_len, block_size, stride_kv, mask_h, offs_tile, mx_ptrs, qk_base, sm_scale, stride_q_d, HAS_LEN: tl.constexpr,
                 BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, TILE_SIZE: tl.constexpr, D_NOPE: tl.constexpr, KV_FP4: tl.constexpr, KV_V41: tl.constexpr):
    NEG_INF = float("-inf")
    kv_block_base, data_off, scale_off, valid = _block_valid(kv, idx_base, n_start + tl.arange(0, BLOCK_N), n_limit, topk_len, block_size, stride_kv, HAS_LEN, KV_FP4, KV_V41)
    valid_2d = valid[:, None]
    qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
    q_0 = tl.load(q_base + offs_tile[None, :] * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_1 = tl.load(q_base + (TILE_SIZE + offs_tile[None, :]) * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_2 = tl.load(q_base + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_3 = tl.load(q_base + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_4 = tl.load(q_base + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_5 = tl.load(q_base + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_6 = tl.load(q_base + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    q_7 = tl.load(q_base + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d, mask=mask_h[:, None], other=0.0).to(tl.bfloat16)
    if KV_V41:
        qk += tl.dot(q_0, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 0))).to(tl.float32)
        qk += tl.dot(q_1, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 1))).to(tl.float32)
        qk += tl.dot(q_2, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 2))).to(tl.float32)
        qk += tl.dot(q_3, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 3))).to(tl.float32)
        qk += tl.dot(q_4, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 4))).to(tl.float32)
        qk += tl.dot(q_5, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 5))).to(tl.float32)
        qk += tl.dot(q_6, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 6))).to(tl.float32)
        qk += tl.dot(q_7, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, 7))).to(tl.float32)
    elif KV_FP4:
        qk += tl.dot(q_0, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 0))).to(tl.float32)
        qk += tl.dot(q_1, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 1))).to(tl.float32)
        qk += tl.dot(q_2, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 2))).to(tl.float32)
        qk += tl.dot(q_3, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 3))).to(tl.float32)
        qk += tl.dot(q_4, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 4))).to(tl.float32)
        qk += tl.dot(q_5, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 5))).to(tl.float32)
        qk += tl.dot(q_6, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 6))).to(tl.float32)
        qk += tl.dot(q_7, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, 7))).to(tl.float32)
    else:
        qk += tl.dot(q_0, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, 0, TILE_SIZE))).to(tl.float32)
        qk += tl.dot(q_1, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, 1, TILE_SIZE))).to(tl.float32)
        qk += tl.dot(q_2, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, 2, TILE_SIZE))).to(tl.float32)
        qk += tl.dot(q_3, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, 3, TILE_SIZE))).to(tl.float32)
        qk += tl.dot(q_4, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, 4, TILE_SIZE))).to(tl.float32)
        qk += tl.dot(q_5, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, 5, TILE_SIZE))).to(tl.float32)
        qk += tl.dot(q_6, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, 6, TILE_SIZE))).to(tl.float32)
        qk += tl.dot(q_7, tl.trans(_tile_rope(kv_block_base, data_off, valid_2d, offs_tile, D_NOPE))).to(tl.float32)
    qk = qk * sm_scale
    qk = tl.where(valid[None, :], qk, NEG_INF)
    m_ij = tl.max(qk, axis=1)
    tl.store(mx_ptrs, m_ij, mask=mask_h)
    tl.store(qk_base, qk, mask=mask_h[:, None])


@triton.jit
def _scores_kernel(Q, KV_Main, Idx_Main, Len_Main, KV_Extra, Idx_Extra, Len_Extra, QK, MaxV, sm_scale, topk_main, block_size_main, topk_extra, block_size_extra, blocks_main, h_q,
                   stride_q_t, stride_q_h, stride_q_d, stride_kv_main, stride_kv_extra, stride_idx_main_t, stride_idx_extra_t, stride_qk_t, stride_qk_b, stride_mx_t, stride_mx_b,
                   HAS_LEN_MAIN: tl.constexpr, HAS_LEN_EXTRA: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, EXTRA_FP4: tl.constexpr, MAIN_V41: tl.constexpr):
    TILE_SIZE: tl.constexpr = 64
    D_NOPE: tl.constexpr = 448
    NEG_INF = float("-inf")
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_tile = tl.arange(0, TILE_SIZE)
    mx_ptrs = MaxV + pid_t * stride_mx_t + pid_b * stride_mx_b + offs_h
    tl.store(mx_ptrs, tl.full([BLOCK_H], NEG_INF, dtype=tl.float32), mask=mask_h)
    q_base = Q + pid_t.to(tl.int64) * tl.cast(stride_q_t, tl.int64) + offs_h[:, None] * stride_q_h
    qk_base = QK + pid_t * stride_qk_t + pid_b * stride_qk_b + offs_h[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    topk_len_main = 0
    topk_len_extra = 0
    if HAS_LEN_MAIN:
        topk_len_main = tl.load(Len_Main + pid_t)
    if HAS_LEN_EXTRA:
        topk_len_extra = tl.load(Len_Extra + pid_t)
    if pid_b < blocks_main:
        n_start = pid_b * BLOCK_N
        if not HAS_LEN_MAIN or n_start < topk_len_main:
            _score_block(q_base, KV_Main, Idx_Main + pid_t * stride_idx_main_t, n_start, topk_main, topk_len_main, block_size_main, stride_kv_main, mask_h, offs_tile, mx_ptrs, qk_base,
                         sm_scale, stride_q_d, HAS_LEN_MAIN, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, False, MAIN_V41)
    else:
        n_start = (pid_b - blocks_main) * BLOCK_N
        if not HAS_LEN_EXTRA or n_start < topk_len_extra:
            _score_block(q_base, KV_Extra, Idx_Extra + pid_t * stride_idx_extra_t, n_start, topk_extra, topk_len_extra, block_size_extra, stride_kv_extra, mask_h, offs_tile, mx_ptrs, qk_base,
                         sm_scale, stride_q_d, HAS_LEN_EXTRA, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, EXTRA_FP4, False)


@triton.jit
def _probabilities_kernel(QK, MaxV, PB, Alpha, SumP, h_q, stride_qk_t, stride_qk_b, stride_mx_t, stride_mx_b, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr):
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    for b in range(0, pid_b):
        m_i = tl.maximum(m_i, tl.load(MaxV + pid_t * stride_mx_t + b * stride_mx_b + offs_h, mask=mask_h, other=NEG_INF))
    m_ij = tl.load(MaxV + pid_t * stride_mx_t + pid_b * stride_mx_b + offs_h, mask=mask_h, other=NEG_INF)
    # A block no token of which ran leaves acc, l and m alone, as the unsplit kernel's skip: alpha 1, l unchanged, p unused
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
    offs_n = tl.arange(0, BLOCK_N)
    row = offs_h[:, None] * BLOCK_N + offs_n[None, :]
    qk = tl.load(QK + pid_t * stride_qk_t + pid_b * stride_qk_b + row, mask=mask_h[:, None], other=NEG_INF)
    p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
    # The row sum is FP32 and order-bearing, and on another layout the BF16 cast compiles to the packed cvt.rn.bf16x2.f32, which rounds differently from the unsplit kernel's
    # cvt.rn.bf16.f32. Both must run on the MMA layout, so p takes it from a dot of zeros, through a select that always picks p: NaN and all, its bits do not change.
    zero = tl.dot(tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.bfloat16), tl.zeros([BLOCK_N, 64], dtype=tl.bfloat16)).to(tl.float32)
    p_mma = tl.where(zero != 0.0, zero, p)
    sum_p = tl.sum(p_mma, axis=1)
    p_bf16 = p_mma.to(tl.bfloat16)
    tl.store(PB + pid_t * stride_qk_t + pid_b * stride_qk_b + row, p_bf16, mask=mask_h[:, None])
    tl.store(Alpha + pid_t * stride_mx_t + pid_b * stride_mx_b + offs_h, alpha, mask=mask_h)
    tl.store(SumP + pid_t * stride_mx_t + pid_b * stride_mx_b + offs_h, sum_p, mask=mask_h)


@triton.jit
def _output_kernel(KV_Main, Idx_Main, Len_Main, KV_Extra, Idx_Extra, Len_Extra, PB, Alpha, SumP, MaxV, AttnSink, Output, topk_main, block_size_main, topk_extra, block_size_extra,
                   blocks_main, blocks_extra, h_q, stride_kv_main, stride_kv_extra, stride_idx_main_t, stride_idx_extra_t, stride_qk_t, stride_qk_b, stride_mx_t, stride_mx_b,
                   stride_o_t, stride_o_h, stride_o_d, HAS_LEN_MAIN: tl.constexpr, HAS_LEN_EXTRA: tl.constexpr, HAS_ATTN_SINK: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, EXTRA_FP4: tl.constexpr, MAIN_V41: tl.constexpr):
    TILE_SIZE: tl.constexpr = 64
    D_NOPE: tl.constexpr = 448
    LOG2E: tl.constexpr = 1.4426950408889634
    NEG_INF = float("-inf")
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q
    offs_tile = tl.arange(0, TILE_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    row = offs_h[:, None] * BLOCK_N + offs_n[None, :]
    topk_len_main = 0
    topk_len_extra = 0
    if HAS_LEN_MAIN:
        topk_len_main = tl.load(Len_Main + pid_t)
    if HAS_LEN_EXTRA:
        topk_len_extra = tl.load(Len_Extra + pid_t)
    for b in range(0, blocks_main):
        n_start = b * BLOCK_N
        should_compute = not HAS_LEN_MAIN or n_start < topk_len_main
        if should_compute:
            kv_block_base, data_off, scale_off, valid = _block_valid(KV_Main, Idx_Main + pid_t * stride_idx_main_t, n_start + offs_n, topk_main, topk_len_main,
                                                                     block_size_main, stride_kv_main, HAS_LEN_MAIN, False, MAIN_V41)
            valid_2d = valid[:, None]
            # No branch here: Triton picks the dot's kWidth from the narrowest type it can trace back, and a branch hides the uint8 loads. kWidth 2 splits K differently across the MMAs.
            if MAIN_V41:
                kv = v41_tile(kv_block_base, data_off, scale_off, valid_2d, pid_d)
            else:
                kv = tl.where(pid_d < 7, _tile_nope(kv_block_base, data_off, scale_off, valid & (pid_d < 7), valid_2d & (pid_d < 7), offs_tile, tl.minimum(pid_d, 6), TILE_SIZE),
                              _tile_rope(kv_block_base, data_off, valid_2d & (pid_d == 7), offs_tile, D_NOPE))
            alpha = tl.load(Alpha + pid_t * stride_mx_t + b * stride_mx_b + offs_h, mask=mask_h, other=0.0)
            sum_p = tl.load(SumP + pid_t * stride_mx_t + b * stride_mx_b + offs_h, mask=mask_h, other=0.0)
            p_bf16 = tl.load(PB + pid_t * stride_qk_t + b * stride_qk_b + row, mask=mask_h[:, None], other=0.0)
            m_i = tl.maximum(m_i, tl.load(MaxV + pid_t * stride_mx_t + b * stride_mx_b + offs_h, mask=mask_h, other=NEG_INF))
            l_i = alpha * l_i + sum_p
            acc = acc * alpha[:, None] + tl.dot(p_bf16, kv).to(tl.float32)
    for e in range(0, blocks_extra):
        n_start = e * BLOCK_N
        b = blocks_main + e
        should_compute = not HAS_LEN_EXTRA or n_start < topk_len_extra
        if should_compute:
            kv_block_base, data_off, scale_off, valid = _block_valid(KV_Extra, Idx_Extra + pid_t * stride_idx_extra_t, n_start + offs_n, topk_extra, topk_len_extra,
                                                                     block_size_extra, stride_kv_extra, HAS_LEN_EXTRA, EXTRA_FP4, False)
            valid_2d = valid[:, None]
            # No branch here: Triton picks the dot's kWidth from the narrowest type it can trace back, and a branch hides the uint8 loads. kWidth 2 splits K differently across the MMAs.
            if EXTRA_FP4:
                kv = fp4_tile(kv_block_base, data_off, scale_off, valid_2d, pid_d)
            else:
                kv = tl.where(pid_d < 7, _tile_nope(kv_block_base, data_off, scale_off, valid & (pid_d < 7), valid_2d & (pid_d < 7), offs_tile, tl.minimum(pid_d, 6), TILE_SIZE),
                              _tile_rope(kv_block_base, data_off, valid_2d & (pid_d == 7), offs_tile, D_NOPE))
            alpha = tl.load(Alpha + pid_t * stride_mx_t + b * stride_mx_b + offs_h, mask=mask_h, other=0.0)
            sum_p = tl.load(SumP + pid_t * stride_mx_t + b * stride_mx_b + offs_h, mask=mask_h, other=0.0)
            p_bf16 = tl.load(PB + pid_t * stride_qk_t + b * stride_qk_b + row, mask=mask_h[:, None], other=0.0)
            m_i = tl.maximum(m_i, tl.load(MaxV + pid_t * stride_mx_t + b * stride_mx_b + offs_h, mask=mask_h, other=NEG_INF))
            l_i = alpha * l_i + sum_p
            acc = acc * alpha[:, None] + tl.dot(p_bf16, kv).to(tl.float32)
    is_lonely_q = l_i == 0.0
    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)
    acc = tl.where(is_lonely_q[:, None], 0.0, acc * output_scale[:, None])
    o = acc.to(tl.bfloat16)
    o_base = Output + pid_t.to(tl.int64) * tl.cast(stride_o_t, tl.int64)
    tl.store(o_base + offs_h[:, None] * stride_o_h + (pid_d * TILE_SIZE + offs_tile[None, :]) * stride_o_d, o, mask=mask_h[:, None])


def decode(q3, k_cache, idx_main, topk_length, attn_sink, softmax_scale, extra_k_cache, idx_extra, extra_topk_length):
    """[rows, H, 512] bf16 query; the output [rows, H, 512] bf16 of the unsplit kernel. rows small: the grids are per row."""
    rows, h_q, _ = q3.shape
    device = q3.device
    kv_main = k_cache.view(torch.uint8)
    topk_main = idx_main.shape[1]
    has_extra = extra_k_cache is not None and idx_extra is not None
    if has_extra:
        kv_extra, topk_extra = extra_k_cache.view(torch.uint8), idx_extra.shape[1]
    else:
        kv_extra, idx_extra, topk_extra, extra_topk_length = kv_main, idx_main, 0, None
    blocks_main, blocks_extra = triton.cdiv(topk_main, BLOCK_N), triton.cdiv(topk_extra, BLOCK_N)
    blocks = blocks_main + blocks_extra
    hb = triton.cdiv(h_q, BLOCK_H)
    qk = torch.empty(rows, blocks, h_q, BLOCK_N, dtype=torch.float32, device=device)
    pb = torch.empty(rows, blocks, h_q, BLOCK_N, dtype=torch.bfloat16, device=device)
    mx = torch.empty(rows, blocks, h_q, dtype=torch.float32, device=device)
    alpha = torch.empty(rows, blocks, h_q, dtype=torch.float32, device=device)
    sum_p = torch.empty(rows, blocks, h_q, dtype=torch.float32, device=device)
    out = torch.empty(rows, h_q, _D, dtype=torch.bfloat16, device=device)
    tl_main = topk_length.reshape(-1).contiguous() if topk_length is not None else mx[0, 0, :1]
    tl_extra = extra_topk_length.reshape(-1).contiguous() if extra_topk_length is not None else mx[0, 0, :1]
    sink = attn_sink.contiguous() if attn_sink is not None else mx[0, 0, :]
    extra_fp4 = has_extra and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN
    flags = dict(HAS_LEN_MAIN=topk_length is not None, HAS_LEN_EXTRA=extra_topk_length is not None, EXTRA_FP4=extra_fp4, MAIN_V41=k_cache.shape[-1] == V41_BYTES_PER_TOKEN)
    cfg = dict(BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, num_warps=NUM_WARPS, num_stages=1)
    head = (kv_main, idx_main, tl_main, kv_extra, idx_extra, tl_extra)
    st_qk = (qk.stride(0), qk.stride(1))
    st_mx = (mx.stride(0), mx.stride(1))
    _scores_kernel[(hb, rows, blocks)](q3, *head, qk, mx, softmax_scale, topk_main, kv_main.shape[1], topk_extra, kv_extra.shape[1], blocks_main, h_q,
                                       q3.stride(0), q3.stride(1), q3.stride(2), kv_main.stride(0), kv_extra.stride(0), idx_main.stride(0), idx_extra.stride(0), *st_qk, *st_mx, **flags, **cfg)
    _probabilities_kernel[(hb, rows, blocks)](qk, mx, pb, alpha, sum_p, h_q, *st_qk, *st_mx, BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, num_warps=NUM_WARPS, num_stages=1)
    _output_kernel[(hb, rows, 8)](*head, pb, alpha, sum_p, mx, sink, out, topk_main, kv_main.shape[1], topk_extra, kv_extra.shape[1], blocks_main, blocks_extra, h_q,
                                  kv_main.stride(0), kv_extra.stride(0), idx_main.stride(0), idx_extra.stride(0), *st_qk, *st_mx, out.stride(0), out.stride(1), out.stride(2),
                                  **flags, HAS_ATTN_SINK=attn_sink is not None, **cfg)
    return out
