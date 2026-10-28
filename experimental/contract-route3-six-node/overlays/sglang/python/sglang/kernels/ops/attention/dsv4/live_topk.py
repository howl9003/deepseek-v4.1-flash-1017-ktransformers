"""Live-length top-k for the non-paged (SM89) DeepSeek V4.1 decode indexer.

Opt-in with KT_IDX_LIVE_TOPK=1; only used on compute capability 8.9 and for graph
capacities up to KT_IDX_LIVE_TOPK_MAX_L (default 65536). Origin: the Strata gpu_glue
bundle (2026-10-22); it applies the live-length bound of the SM89 recipe's
sparse-valid-bound logits kernel to the selection that follows it.

The decode graph scores every indexer layer over the capture capacity ``L`` (16385
compressed positions at ratio 1 for a 16K pool) and then runs ``torch.topk`` (one
CTA, radix select over all L columns: 36-64 us), ``sort`` of the picks (19 us) and a
chain of small gathers, although only the first ``lens[b]`` columns are visible and
the rest are ``-inf``. These kernels do the same selection on the live columns only,
reading the device-side lengths, so a captured graph stays replayable.

Exactness (the output equals the torch path element for element):

* torch's CUDA top-k (sbtopk and mbtopk gatherTopK) selects every element whose
  order key is strictly above the k-th key, then the elements equal to the k-th key
  in ascending scan order until k are taken. The order key is
  ``TopKTypeConfig<float>::convert`` (sign-flipped bits, NaN largest), reproduced in
  ``_ord_key``. The k-th key is found here by an MSB-first radix select over the
  live columns; columns past ``lens`` hold -inf, which can only be selected when
  fewer than k finite columns exist, and the torch path then maps them to -1.
* write_decode sorts the picked positions ascending before mapping them to pool
  slots; the emission pass below walks positions in ascending order, so its output
  is already in that order.
* topk_among_blocks scans candidates in the published block order (block rank,
  then offset): the tie rule follows that order, so the among-blocks kernel ranks
  ties in flat (block-rank) order, marks the picks, and then emits in position
  order like the full path. It reads only the first ``cdiv(lens, block_size)``
  published blocks: select_candidate_block_ids publishes its finite blocks first
  (block amax descending, -1 after them), and every live block is finite unless a
  logit is NaN.
* ``NPASS=2`` resolves the key on its top 16 bits. That is exact for the indexer's
  scores because fp4_index_logits_decode rounds every logit through bf16 (the low
  16 float bits are zero, so the low 16 key bits are a function of the sign).
  ``NPASS=4`` (KT_IDX_LIVE_TOPK_FULLKEY=1) resolves all 32 bits for any fp32 input.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

_BLOCK = 2048
_SLOT_BLOCK = 1024
# fp4_index_logits_decode tiles positions by 64 and loads the whole live tile.
_LOGITS_TILE = 64

_ENABLED = os.environ.get("KT_IDX_LIVE_TOPK", "0") == "1"
# One CTA per row: past this capacity torch's multi-block top-k is the better
# choice (a 1M-token pool has L ~ 1M at ratio 1), so the gate is static per graph.
_MAX_L = int(os.environ.get("KT_IDX_LIVE_TOPK_MAX_L", "65536"))
_NPASS = 4 if os.environ.get("KT_IDX_LIVE_TOPK_FULLKEY", "0") == "1" else 2


@functools.lru_cache(maxsize=None)
def _is_sm89(device_index) -> bool:
    return torch.cuda.get_device_capability(device_index) == (8, 9)


def live_topk_enabled(lmax: int, device: torch.device) -> bool:
    """Static per captured graph: the env switch, SM89, and the capacity bound."""
    if not _ENABLED or device.type != "cuda" or lmax > _MAX_L:
        return False
    return _is_sm89(device.index)


@triton.jit
def _ord_key(v):
    # at::native::TopKTypeConfig<float>::convert
    x = v.to(tl.uint32, bitcast=True)
    neg = (x >> 31) != 0
    k = tl.where(neg, x ^ 0xFFFFFFFF, x | 0x80000000)
    return tl.where(v == v, k, tl.full(k.shape, 0xFFFFFFFF, tl.uint32))


@triton.jit
def _live_slots_kernel(
    r2t_ptr,
    req_ptr,
    lens_ptr,
    slots_ptr,
    L,
    stride_r2t,
    stride_s,
    ratio,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    # slots[b, j] = req_to_token[req[b], j * ratio] // ratio for j < lens[b], 0 up to the
    # end of the last live TILE (what the logits kernel loads); nothing past it.
    b = tl.program_id(0)
    blk = tl.program_id(1)
    n = tl.minimum(tl.load(lens_ptr + b).to(tl.int64), L)
    n_round = tl.minimum((n + TILE - 1) // TILE * TILE, L)
    offs = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if blk * BLOCK < n_round:
        row = tl.load(req_ptr + b).to(tl.int64)
        live = offs < n
        a = tl.load(r2t_ptr + row * stride_r2t + offs * ratio, mask=live, other=0).to(tl.int64)
        q = a // ratio
        # torch floor_divide; a is a pool slot (>= 0), kept exact for any sign
        q = tl.where(((a % ratio) != 0) & (a < 0), q - 1, q)
        q = tl.where(live, q, 0)
        tl.store(slots_ptr + b * stride_s + offs, q, mask=offs < n_round)


@triton.jit
def _emit_sequential(slots_ptr, out_ptr, b, n, K, stride_sl, stride_o, BLOCK: tl.constexpr):
    # n <= K: every live position is picked; ascending, then -1.
    offs0 = tl.arange(0, BLOCK)
    for start in range(0, K, BLOCK):
        j = start + offs0
        s = tl.load(slots_ptr + b * stride_sl + j, mask=j < n, other=0)
        v = tl.where(j < n, s.to(tl.int32), -1)
        tl.store(out_ptr + b * stride_o + j, v, mask=j < K)


@triton.jit
def _live_topk_full_kernel(
    scores_ptr,
    slots_ptr,
    lens_ptr,
    out_ptr,
    L,
    K,
    stride_sc,
    stride_sl,
    stride_o,
    NPASS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.minimum(tl.load(lens_ptr + b).to(tl.int32), L)
    offs0 = tl.arange(0, BLOCK)
    bins = tl.arange(0, 256)
    if n <= K:
        _emit_sequential(slots_ptr, out_ptr, b, n, K, stride_sl, stride_o, BLOCK)
    else:
        need = K + 0 * n
        prefix = tl.full([], 0, tl.uint32)
        for p in tl.static_range(NPASS):
            shift = 24 - 8 * p
            h = tl.zeros([256], dtype=tl.int32)
            for start in range(0, n, BLOCK):
                j = start + offs0
                m = j < n
                v = tl.load(scores_ptr + b * stride_sc + j, mask=m, other=0.0)
                key = _ord_key(v)
                if p > 0:
                    m = m & ((key >> (shift + 8)) == prefix)
                d = ((key >> shift) & 0xFF).to(tl.int32)
                h += tl.histogram(d, 256, mask=m)
            if p == 0:
                need = tl.minimum(need, tl.sum(h, 0))
            c = tl.cumsum(h, 0, reverse=True)
            dsel = tl.max(tl.where(c >= need, bins, -1), 0)
            gt = tl.sum(tl.where(bins > dsel, h, 0), 0)
            need = need - gt
            prefix = (prefix << 8) | dsel.to(tl.uint32)
        TOPSHIFT: tl.constexpr = 32 - 8 * NPASS
        sel_carry = 0 * n
        tie_carry = 0 * n
        for start in range(0, n, BLOCK):
            j = start + offs0
            m = j < n
            v = tl.load(scores_ptr + b * stride_sc + j, mask=m, other=0.0)
            kp = _ord_key(v) >> TOPSHIFT
            gtm = m & (kp > prefix)
            eqm = m & (kp == prefix)
            eqi = eqm.to(tl.int32)
            rank = tie_carry + tl.cumsum(eqi, 0) - eqi
            sel = gtm | (eqm & (rank < need))
            seli = sel.to(tl.int32)
            dst = sel_carry + tl.cumsum(seli, 0) - seli
            s = tl.load(slots_ptr + b * stride_sl + j, mask=sel, other=0)
            tl.store(out_ptr + b * stride_o + dst, s.to(tl.int32), mask=sel)
            sel_carry += tl.sum(seli, 0)
            tie_carry += tl.sum(eqi, 0)
        # fewer than K finite picks cannot happen here (n > K live columns), but keep the
        # torch layout: -1 after the picks up to K
        for start in range(0, K, BLOCK):
            j = start + offs0
            tl.store(out_ptr + b * stride_o + j, tl.full([BLOCK], -1, tl.int32),
                     mask=(j >= sel_carry) & (j < K))


@triton.jit
def _live_topk_blocks_kernel(
    scores_ptr,
    slots_ptr,
    lens_ptr,
    blocks_ptr,
    flag_ptr,
    out_ptr,
    L,
    K,
    NB,
    stride_sc,
    stride_sl,
    stride_bl,
    stride_fl,
    stride_o,
    BS: tl.constexpr,
    NPASS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.minimum(tl.load(lens_ptr + b).to(tl.int32), L)
    offs0 = tl.arange(0, BLOCK)
    bins = tl.arange(0, 256)
    nbv = tl.minimum(NB, (n + BS - 1) // BS)
    all_live_published = False
    if n <= K:
        # NaN/-inf block maxima are removed by the producer. Only take the short
        # path when the actual published blocks cover every live position.
        blk = tl.load(blocks_ptr + b * stride_bl + offs0, mask=offs0 < nbv, other=-1)
        valid = (offs0 < nbv) & (blk >= 0) & (blk * BS < n)
        covered = tl.sum(tl.where(valid, tl.minimum(BS, n - blk * BS), 0), 0)
        all_live_published = covered == n
    if all_live_published:
        _emit_sequential(slots_ptr, out_ptr, b, n, K, stride_sl, stride_o, BLOCK)
    else:
        # The source publishes its finite blocks first (block amax descending, -1 past
        # the finite ones), and the finite blocks are the cdiv(n, BS) live ones.
        F = nbv * BS
        need = K + 0 * n
        prefix = tl.full([], 0, tl.uint32)
        for p in tl.static_range(NPASS):
            shift = 24 - 8 * p
            h = tl.zeros([256], dtype=tl.int32)
            for start in range(0, F, BLOCK):
                f = start + offs0
                inb = f < F
                blk = tl.load(blocks_ptr + b * stride_bl + f // BS, mask=inb, other=-1)
                pos = blk * BS + f % BS
                m = inb & (blk >= 0) & (pos < n)
                v = tl.load(scores_ptr + b * stride_sc + pos, mask=m, other=0.0)
                key = _ord_key(v)
                if p > 0:
                    m = m & ((key >> (shift + 8)) == prefix)
                d = ((key >> shift) & 0xFF).to(tl.int32)
                h += tl.histogram(d, 256, mask=m)
            if p == 0:
                need = tl.minimum(need, tl.sum(h, 0))
            c = tl.cumsum(h, 0, reverse=True)
            dsel = tl.max(tl.where(c >= need, bins, -1), 0)
            gt = tl.sum(tl.where(bins > dsel, h, 0), 0)
            need = need - gt
            prefix = (prefix << 8) | dsel.to(tl.uint32)
        TOPSHIFT: tl.constexpr = 32 - 8 * NPASS
        # clear the live flags, mark the picks in flat order, emit in position order
        for start in range(0, n, BLOCK):
            j = start + offs0
            tl.store(flag_ptr + b * stride_fl + j, tl.zeros([BLOCK], tl.int8), mask=j < n)
        tl.debug_barrier()
        tie_carry = 0 * n
        for start in range(0, F, BLOCK):
            f = start + offs0
            inb = f < F
            blk = tl.load(blocks_ptr + b * stride_bl + f // BS, mask=inb, other=-1)
            pos = blk * BS + f % BS
            m = inb & (blk >= 0) & (pos < n)
            v = tl.load(scores_ptr + b * stride_sc + pos, mask=m, other=0.0)
            kp = _ord_key(v) >> TOPSHIFT
            gtm = m & (kp > prefix)
            eqm = m & (kp == prefix)
            eqi = eqm.to(tl.int32)
            rank = tie_carry + tl.cumsum(eqi, 0) - eqi
            sel = gtm | (eqm & (rank < need))
            tl.store(flag_ptr + b * stride_fl + pos, tl.full([BLOCK], 1, tl.int8), mask=sel)
            tie_carry += tl.sum(eqi, 0)
        tl.debug_barrier()
        sel_carry = 0 * n
        for start in range(0, n, BLOCK):
            j = start + offs0
            m = j < n
            sel = (tl.load(flag_ptr + b * stride_fl + j, mask=m, other=0) != 0) & m
            seli = sel.to(tl.int32)
            dst = sel_carry + tl.cumsum(seli, 0) - seli
            s = tl.load(slots_ptr + b * stride_sl + j, mask=sel, other=0)
            tl.store(out_ptr + b * stride_o + dst, s.to(tl.int32), mask=sel)
            sel_carry += tl.sum(seli, 0)
        for start in range(0, K, BLOCK):
            j = start + offs0
            tl.store(out_ptr + b * stride_o + j, tl.full([BLOCK], -1, tl.int32),
                     mask=(j >= sel_carry) & (j < K))


def live_decode_slots(
    req_to_token: torch.Tensor, req: torch.Tensor, lens: torch.Tensor, ratio: int, lmax: int
) -> torch.Tensor:
    """[bs, lmax] int64 pool slots, valid on [0, lens) and zero to the end of the last
    live 64-tile; the rest is never read (fp4_index_logits_decode skips those tiles)."""
    bs = req.shape[0]
    slots = torch.empty((bs, lmax), dtype=torch.int64, device=req.device)
    if bs == 0 or lmax == 0:
        return slots
    assert req_to_token.stride(1) == 1, req_to_token.stride()
    req, lens = req.contiguous(), lens.contiguous()
    grid = (bs, triton.cdiv(lmax, _SLOT_BLOCK))
    _live_slots_kernel[grid](
        req_to_token,
        req,
        lens,
        slots,
        lmax,
        req_to_token.stride(0),
        slots.stride(0),
        ratio,
        BLOCK=_SLOT_BLOCK,
        TILE=_LOGITS_TILE,
        num_warps=4,
    )
    return slots


def live_topk_full(
    scores: torch.Tensor, slots: torch.Tensor, lens: torch.Tensor, k: int, out: torch.Tensor
) -> None:
    """``write_decode(scores.topk(k, sorted=False).indices)`` on the live columns:
    out[:bs, :k] int32 = slots of the picks, ascending position, -1 padded."""
    bs, L = scores.shape
    if bs == 0:
        return
    assert scores.dtype == torch.float32 and scores.stride(1) == 1 and slots.stride(1) == 1
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[1] >= k
    lens = lens.contiguous()
    _live_topk_full_kernel[(bs,)](
        scores,
        slots,
        lens,
        out,
        L,
        k,
        scores.stride(0),
        slots.stride(0),
        out.stride(0),
        NPASS=_NPASS,
        BLOCK=_BLOCK,
        num_warps=8,
    )


def live_topk_among_blocks(
    scores: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    blocks: torch.Tensor,
    k: int,
    block_size: int,
    out: torch.Tensor,
) -> None:
    """``write_decode(topk_among_blocks(...).masked_fill(<0, lmax))`` on the live
    columns, ties ranked in the published block order as torch does."""
    bs, L = scores.shape
    if bs == 0:
        return
    assert blocks.shape[0] == bs and blocks.dtype == torch.int32 and blocks.stride(1) == 1
    assert scores.dtype == torch.float32 and scores.stride(1) == 1 and slots.stride(1) == 1
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[1] >= k
    nb = blocks.shape[1]
    # n <= k is handled as "all live positions"; that needs every live block published
    assert nb * block_size >= k, (nb, block_size, k)
    lens = lens.contiguous()
    flags = torch.empty((bs, L), dtype=torch.int8, device=scores.device)
    _live_topk_blocks_kernel[(bs,)](
        scores,
        slots,
        lens,
        blocks,
        flags,
        out,
        L,
        k,
        nb,
        scores.stride(0),
        slots.stride(0),
        blocks.stride(0),
        flags.stride(0),
        out.stride(0),
        BS=block_size,
        NPASS=_NPASS,
        BLOCK=_BLOCK,
        num_warps=8,
    )
