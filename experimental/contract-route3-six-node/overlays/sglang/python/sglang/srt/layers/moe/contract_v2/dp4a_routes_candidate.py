"""Unvalidated, isolated decode-only contract-v2 exact int64 split-K experiment.

Exact integer arithmetic and rounding boundaries follow kernels.py. GPU alignment
replaces host reads of unique experts. Weight exponent metadata is recomputed on
the GPU on every replay so raw DMA/VMM slot replacement cannot leave it stale.
Quantization keeps only the three integer planes used by the SM89 int8 route.
"""
import torch
import triton as tr
import triton.language as tl
from .kernels import _bf, _finish, MARGIN
from .dp4a_projection_r1 import project


@tr.jit
def _quant_planes(X, EA, P0, P1, P2, NTAB):
    block = tl.program_id(0)
    j = tl.arange(0, 32)
    v = tl.load(X + block * 32 + j).to(tl.float32)
    am = tl.maximum(tl.max(tl.abs(v), 0), 1.e-4)
    b = (am * (1. / 448.)).to(tl.uint32, bitcast=True)
    ex = ((b >> 23) & 255).to(tl.int32) + ((b & 0x7fffff) != 0).to(tl.int32) - 127
    scale = ((ex + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    z = v / scale
    a = tl.abs(z); ab = a.to(tl.uint32, bitcast=True)
    normal = ((ab + 0x7ffff + ((ab >> 20) & 1)) >> 20).to(tl.int32) - 960
    sub = a * 512.; flo = tl.floor(sub); frac = sub - flo
    rounded = flo.to(tl.int32) + ((frac > .5) | ((frac == .5) & ((flo.to(tl.int32) & 1) != 0))).to(tl.int32)
    code = tl.where(a >= 448., 126, tl.where(a < .015625, rounded, normal))
    sign = ((z.to(tl.uint32, bitcast=True) >> 31) & 1).to(tl.int32)
    code = code | (sign << 7)
    n = tl.load(NTAB + code); sg = 1 - 2 * sign
    tl.store(P0 + block * 32 + j, (sg * (n & 63)).to(tl.int8))
    tl.store(P1 + block * 32 + j, (sg * ((n >> 6) & 63)).to(tl.int8))
    tl.store(P2 + block * 32 + j, (sg * (n >> 12)).to(tl.int8))
    tl.store(EA + block, ex.to(tl.int8))


@tr.jit
def _active_weight_metadata(WS, EW, BAD, EXPERTS, NTP,
                            N: tl.constexpr, G: tl.constexpr, BM: tl.constexpr,
                            BN: tl.constexpr, BG: tl.constexpr):
    group = tl.program_id(0)
    if group * BM < tl.load(NTP):
        eid = tl.load(EXPERTS + group)
        previous = tl.load(EXPERTS + group - 1, group > 0, -2)
        # Alignment sorts blocks by expert. Only its first block writes the
        # metadata: no duplicate stores, and no read of unbacked VMM slots.
        if eid >= 0 and eid != previous:
            col = tl.program_id(1) * BN + tl.arange(0, BN)
            g = tl.arange(0, BG)
            mask = (col[:, None] < N) & (g[None, :] < G)
            code = tl.load(WS + (eid.to(tl.int64) * N + col[:, None]) * G + g[None, :], mask, 32).to(tl.int32)
            bad = tl.sum(tl.sum(((code < 32) | (code == 255)).to(tl.int32), 1), 0)
            if bad != 0:
                tl.atomic_or(BAD, 1)
            value = tl.max(code, 1) - 127
            tl.store(EW + eid.to(tl.int64) * N + col, value.to(tl.int8), col < N)


@tr.jit
def _grouped_projection_split(A0, A1, A2, EAG, EAT, W, WS, EW, WLUT8,
                        SORTED, EXPERTS, NTP, Y,
                        COUNT: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        TOPK: tl.constexpr, FIRST: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr):
    split = tl.program_id(2)
    group = tl.program_id(0)
    if group * BM < tl.load(NTP):
        eid = tl.load(EXPERTS + group)
        if eid >= 0:
            pos = group * BM + tl.arange(0, BM)
            route = tl.load(SORTED + pos).to(tl.int64)
            rm = route < COUNT
            row = route // TOPK if FIRST else route
            cols = tl.program_id(1) * BN + tl.arange(0, BN); cm = cols < N
            j = tl.arange(0, 32); G: tl.constexpr = K // 32
            eat = tl.load(EAT + row, rm, 0).to(tl.int32)
            ewc = tl.load(EW + eid * N + cols, cm, 0).to(tl.int32)
            acc = tl.zeros((BM, BN), tl.int64)
            per_split: tl.constexpr = (G + SPLITS - 1) // SPLITS
            for g in range(split * per_split, tl.minimum((split + 1) * per_split, G)):
                kk = g * 32 + j
                byte = tl.load(W + eid.to(tl.int64) * N * (K // 2) + cols[None, :] * (K // 2) + kk[:, None] // 2, cm[None, :], 0).to(tl.int32)
                code = (byte >> ((kk[:, None] & 1) * 4)) & 15
                wb = tl.load(WLUT8 + code)
                a0 = tl.load(A0 + row[:, None] * K + kk[None, :], rm[:, None], 0)
                a1 = tl.load(A1 + row[:, None] * K + kk[None, :], rm[:, None], 0)
                a2 = tl.load(A2 + row[:, None] * K + kk[None, :], rm[:, None], 0)
                d0 = tl.dot(a0, wb, out_dtype=tl.int32)
                d1 = tl.dot(a1, wb, out_dtype=tl.int32)
                d2 = tl.dot(a2, wb, out_dtype=tl.int32)
                dot = ((d2 << 12) + (d1 << 6) + d0).to(tl.int64)
                eag = tl.load(EAG + row * G + g, rm, 0).to(tl.int32)
                wsg = tl.load(WS + eid.to(tl.int64) * N * G + cols * G + g, cm, 127).to(tl.int32)
                shift = (eag[:, None] - eat[:, None]) + (wsg[None, :] - 127 - ewc[None, :]) + MARGIN
                acc += tl.where(shift >= 0, dot << tl.maximum(shift, 0), dot >> tl.minimum(0 - shift, 63))
            # Raw int64 partials: no floating-point conversion before the final merge.
            tl.store(Y + (split.to(tl.int64) * COUNT + route[:, None]) * N + cols[None, :], acc, rm[:, None] & cm[None, :])


@tr.jit
def _merge_projection(P, EAT, EW, SORTED, EXPERTS, NTP, Y,
                      COUNT: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr,
                      FIRST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                      SPLITS: tl.constexpr):
    group = tl.program_id(0)
    if group * BM < tl.load(NTP):
        eid = tl.load(EXPERTS + group)
        if eid >= 0:
            route = tl.load(SORTED + group * BM + tl.arange(0, BM)).to(tl.int64)
            cols = tl.program_id(1) * BN + tl.arange(0, BN)
            valid = (route[:, None] < COUNT) & (cols[None, :] < N)
            row = route // TOPK if FIRST else route
            acc = tl.zeros((BM, BN), tl.int64)
            for split in tl.static_range(SPLITS):
                acc += tl.load(P + (split * COUNT + route[:, None]) * N + cols[None, :], valid, 0)
            eat = tl.load(EAT + row, route < COUNT, 0).to(tl.int32)
            ewc = tl.load(EW + eid.to(tl.int64) * N + cols, cols < N, 0).to(tl.int32)
            tl.store(Y + route[:, None] * N + cols[None, :], _finish(acc, eat, ewc), valid)


@tr.jit
def _activate_routes(GU, IDS, WEIGHTS, TABLE, H, I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    eid = tl.load(IDS + row); weight = tl.load(WEIGHTS + row)
    value = tl.zeros((BLOCK,), tl.float32)
    if eid >= 0:
        gate = tl.load(GU + row * (2 * I) + col, col < I, 0)
        up = tl.load(GU + row * (2 * I) + I + col, col < I, 0)
        si = tl.load(TABLE + (gate.to(tl.uint32, bitcast=True) >> 16))
        value = _bf((si * tl.minimum(10., tl.maximum(-10., up))) * weight)
    tl.store(H + row * I + col, value, col < I)


def quant_planes(x, gpu):
    m, k = x.shape
    planes = [torch.empty((m, k), dtype=torch.int8, device=x.device) for _ in range(3)]
    ea = torch.empty((m, k // 32), dtype=torch.int8, device=x.device)
    _quant_planes[(m * k // 32,)](x, ea, *planes, gpu.ntab, enable_fp_fusion=False, num_warps=1)
    return planes, ea, ea.max(dim=1).values


def expert_routes_splitk(x, w13, s13, w2, s2, ids, weights, *, out=None, tile=16, splits=4, partials=False):
    from .moe import _adapter, _as_bytes
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError('contiguous CUDA BF16 activations required')
    m, hidden = x.shape; topk = ids.shape[1]
    if not 1 <= m <= 8 or splits not in (1, 2, 4, 8) or hidden > 5120 or topk > 6:
        raise ValueError("bounded decode experiment requires <=8 rows, <=6 routes, K<=5120 and splits1/2/4/8")
    w13, s13, w2, s2 = [_as_bytes(t, name) for t, name in zip((w13, s13, w2, s2), ('w13', 's13', 'w2', 's2'))]
    experts, twice_i, hh = w13.shape; intermediate = twice_i // 2
    if intermediate > 5120 or hidden % 32 or intermediate % 32:
        raise ValueError("bounded group32 geometry required")
    if not m or not topk or tile not in (16, 32, 64) or hh * 2 != hidden:
        raise ValueError('invalid dimensions/tile')
    if ids.shape != weights.shape or ids.shape[0] != m or weights.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('invalid routing inputs')
    if s13.shape != (experts, twice_i, hidden // 32) or w2.shape != (experts, hidden, intermediate // 2) or s2.shape != (experts, hidden, intermediate // 32):
        raise ValueError('weight/scale shapes differ')
    if any(not t.is_contiguous() or t.device != x.device for t in (w13, s13, w2, s2, ids, weights)):
        raise ValueError('contiguous tensors on one device required')
    if out is None:out = torch.zeros((m, topk, hidden), dtype=torch.bfloat16, device=x.device)
    if out.shape != (m, topk, hidden) or out.dtype != torch.bfloat16 or not out.is_contiguous() or out.device != x.device:
        raise ValueError('invalid per-route output')
    gpu = _adapter(x.device)
    if gpu.mode != 'int8':raise ValueError('static grouped path currently implements int8 contract only')
    torch._assert_async(torch.isfinite(weights).all(), 'non-finite route weights')
    torch._assert_async(torch.isfinite(x).all(), 'non-finite activations')
    active = torch.where((ids >= 0) & (ids < experts) & (weights != 0), ids, -1).contiguous()
    sorted_ids, expert_ids, ntp = moe_align_block_size(active, tile, experts)
    count = m * topk; blocks = tr.cdiv(sorted_ids.numel(), tile)
    # Raw arena DMA/VMM writes do not increment Tensor._version reliably.
    # Refresh only routed experts: shadow tensors include unbacked virtual slots.
    ew13 = torch.empty((experts, twice_i), dtype=torch.int8, device=x.device)
    ew2 = torch.empty((experts, hidden), dtype=torch.int8, device=x.device)
    bad = torch.zeros((), dtype=torch.int32, device=x.device)
    _active_weight_metadata[(blocks, tr.cdiv(twice_i, 16))](s13, ew13, bad, expert_ids, ntp, twice_i, hidden // 32, tile, 16, tr.next_power_of_2(hidden // 32), num_warps=4)
    _active_weight_metadata[(blocks, tr.cdiv(hidden, 16))](s2, ew2, bad, expert_ids, ntp, hidden, intermediate // 32, tile, 16, tr.next_power_of_2(intermediate // 32), num_warps=4)
    torch._assert_async(bad == 0, 'active expert scales outside contract')
    planes, ea, eat = quant_planes(x, gpu)
    gu = torch.empty((count, twice_i), dtype=torch.float32, device=x.device)
    p13 = torch.empty((splits, count, twice_i), dtype=torch.int64, device=x.device)
    project(planes, ea, eat, w13, s13, ew13, active, p13, count, twice_i, hidden, topk, True, splits)
    _merge_projection[(blocks, tr.cdiv(twice_i, 64))](p13, eat, ew13, sorted_ids, expert_ids, ntp, gu, count, twice_i, topk, True, tile, 64, splits, enable_fp_fusion=False, num_warps=4)
    h = torch.empty((count, intermediate), dtype=torch.float32, device=x.device)
    _activate_routes[(count, tr.cdiv(intermediate, 128))](gu, active, weights, gpu.silu, h, intermediate, 128, enable_fp_fusion=False)
    planes2, ea2, eat2 = quant_planes(h, gpu)
    p2 = torch.empty((splits, count, hidden), dtype=torch.int64, device=x.device)
    project(planes2, ea2, eat2, w2, s2, ew2, active, p2, count, hidden, intermediate, topk, False, splits)
    _merge_projection[(blocks, tr.cdiv(hidden, 64))](p2, eat2, ew2, sorted_ids, expert_ids, ntp, out, count, hidden, topk, False, tile, 64, splits, enable_fp_fusion=False, num_warps=4)
    return (out, {'gate_up': p13, 'down': p2, 'active': active}) if partials else out
