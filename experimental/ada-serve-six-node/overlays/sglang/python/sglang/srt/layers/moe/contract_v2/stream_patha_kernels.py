"""Triton glue kernels of the path A streamed-prefill expert path (stream_patha.py). Imported only with the selector
on, never at module import of stream_graph_candidate.py; torch and triton are required here.

Every kernel is a row program: its first program id is a row of the group's sorted route table, and a program whose
row is at or beyond the live row count (read from the device, ROW_OFF[n]) exits before any load or store, so the
static workspaces (sized at the plan's maximum block) cost nothing beyond the rows the group actually has. The path A
prep kernel takes the same device count (r3-P3 `rows=`), so no kernel of the path processes a padded row. No kernel
here does arithmetic that the contract does not already do:

  _quant_codes_rows   the contract quantiser (the lines of static_routes_candidate._quant_planes: amax floor 1e-4,
                      bit-inspection ceil of amax/448, E4M3 round to nearest even), storing the E4M3 code bytes and
                      the group exponents instead of the three int8 planes; the path A prep kernel reads the codes
  _weight_exponents   E8M0 scale codes -> int8 exponents (code - 127) and the per-row maximum; the finish-rule
                      admission bound (code >= 53, CONTRACT_PINS_r1.md section 1) raises BAD
  _row_offsets        moe_align_block_size's per-block expert ids -> the kernel's row-offset table [n + 1] (64-row
                      padded, experts ascending; the inactive routes are sorted last and never live), BAD when the
                      live rows exceed the workspace or an expert exceeds the grid bound
  _gather_rows        token rows of the per-layer planes / metadata / exponents -> the sorted route rows (a copy;
                      padding rows become the zero row: planes 0, ameta {0, 1, 0, 0}, ea 0)
  _activate_rows      the contract's SwiGLU step on the kernel's BF16 gate/up rows: static_routes_candidate
                      ._activate_routes with the BF16 operands read as the FP32 values they are
  _scatter_rows       the kernel's BF16 down rows -> routes[token, slot] of the real routes (a copy)
"""
import torch
import triton as tr
import triton.language as tl

MIN_SCALE_CODE = 53          # finish-rule admission of the expert kernel (weight exponent >= -74)


@tr.jit
def _bf(x):
    b = x.to(tl.uint32, bitcast=True)
    return ((b + 32767 + ((b >> 16) & 1)) & 0xffff0000).to(tl.float32, bitcast=True)


@tr.jit
def _quant_codes_rows(X, CODES, EA, LIVE, G: tl.constexpr, BQ: tl.constexpr, GUARD: tl.constexpr):
    # X bf16 [rows, 32 G]; CODES u8 [rows, 32 G]; EA i8 [rows, G]. The quantiser lines are those of
    # static_routes_candidate._quant_planes, vectorised over BQ groups of one row (axis 1 is the group's 32 lanes).
    row = tl.program_id(0).to(tl.int64)
    if GUARD:
        live = tl.load(LIVE).to(tl.int64)
    else:
        live = row + 1
    if row < live:
        j = tl.arange(0, 32)
        for g0 in range(0, G, BQ):
            gq = g0 + tl.arange(0, BQ)
            gm = gq < G
            off = row * (G * 32) + gq[:, None] * 32 + j[None, :]
            v = tl.load(X + off, gm[:, None], 0.).to(tl.float32)
            am = tl.maximum(tl.max(tl.abs(v), 1), 1.e-4)
            b = (am * (1. / 448.)).to(tl.uint32, bitcast=True)
            ex = ((b >> 23) & 255).to(tl.int32) + ((b & 0x7fffff) != 0).to(tl.int32) - 127
            scale = ((ex + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
            z = v / scale[:, None]
            a = tl.abs(z); ab = a.to(tl.uint32, bitcast=True)
            normal = ((ab + 0x7ffff + ((ab >> 20) & 1)) >> 20).to(tl.int32) - 960
            sub = a * 512.; flo = tl.floor(sub); frac = sub - flo
            rounded = flo.to(tl.int32) + ((frac > .5) | ((frac == .5) & ((flo.to(tl.int32) & 1) != 0))).to(tl.int32)
            code = tl.where(a >= 448., 126, tl.where(a < .015625, rounded, normal))
            sign = ((z.to(tl.uint32, bitcast=True) >> 31) & 1).to(tl.int32)
            code = code | (sign << 7)
            tl.store(CODES + off, code.to(tl.uint8), gm[:, None])
            tl.store(EA + row * G + gq, ex.to(tl.int8), gm)


@tr.jit
def _weight_exponents(S, EWG, EWR, BAD, G: tl.constexpr, BG: tl.constexpr):
    # S u8 [rows, G] E8M0 codes (one program per weight row of every expert of the group) -> EWG i8 [rows, G]
    # (code - 127), EWR i8 [rows] (row maximum); codes below 53 or equal to 255 set BAD.
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, BG)
    m = g < G
    code = tl.load(S + row * G + g, m, 0).to(tl.int32)
    bad = tl.sum((m & ((code < 53) | (code == 255))).to(tl.int32), 0)
    if bad != 0:
        tl.atomic_or(BAD, 1)
    tl.store(EWG + row * G + g, (code - 127).to(tl.int8), m)
    tl.store(EWR + row, (tl.max(tl.where(m, code, 0), 0) - 127).to(tl.int8))


@tr.jit
def _row_offsets(EXPERT_IDS, NTP, ROW_OFF, BAD, n, nblocks, cap, max_rows, BLOCKS: tl.constexpr, BM: tl.constexpr):
    # One program. moe_align_block_size(active, BM, n + 1) with the inactive routes mapped to id n sorts the routes
    # into BM-padded blocks per id ascending (the -1 bucket first, then ids 0..n: experts 0..n-1, then the inactive
    # ones) and writes the id of every block below NTP. ROW_OFF[e] = BM x (blocks of ids below e) for e in 0..n, so
    # expert e's padded rows are [ROW_OFF[e], ROW_OFF[e + 1]) and ROW_OFF[n] is the live row count. BAD: a -1 block
    # (cannot happen: no -1 id is passed), live rows beyond the workspace, or an expert beyond the grid bound.
    b = tl.arange(0, BLOCKS)
    ntp = tl.load(NTP)
    inside = (b < nblocks) & ((b * BM) < ntp)
    eid = tl.load(EXPERT_IDS + b, inside, -2)
    neg = tl.sum((inside & (eid == -1)).to(tl.int32), 0)
    acc = neg
    tl.store(ROW_OFF, acc * BM)
    big = neg * 0                                   # loop-carried int32 scalars start as tensors
    for e in range(0, n):
        c = tl.sum((inside & (eid == e)).to(tl.int32), 0)
        big = tl.maximum(big, c)
        acc = acc + c
        tl.store(ROW_OFF + e + 1, acc * BM)
    if (neg != 0) | (acc * BM > cap) | (big * BM > max_rows):
        tl.atomic_or(BAD, 1)


@tr.jit
def _gather_rows(PLANES, AMETA32, EA, SORTED, LIVE, P, AM32, EAG, first, count, topk, T, cap,
                 K: tl.constexpr, G: tl.constexpr, NC: tl.constexpr, AP: tl.constexpr,
                 BK: tl.constexpr, BG: tl.constexpr, BC: tl.constexpr):
    # PLANES i8 [AP, T, K], AMETA32 i32 [NC, T] (the [NC, T, 4] bytes), EA i8 [T, G]: the layer's prepared rows;
    # SORTED i32 [>= live]: moe_align's route table (route index or the padding sentinel `count`);
    # P i8 [AP, cap, K], AM32 i32 [NC, cap], EAG i8 [cap, G]: the kernel operands in sorted row order.
    r = tl.program_id(0).to(tl.int64)
    if r < tl.load(LIVE).to(tl.int64):
        route = tl.load(SORTED + r)
        real = route < count
        tok = (first + tl.where(real, route, 0) // topk).to(tl.int64)
        k = tl.arange(0, BK)
        for p in tl.static_range(AP):
            for k0 in range(0, K, BK):
                src = tl.load(PLANES + (p * T + tok) * K + k0 + k, real & (k >= 0), 0)
                tl.store(P + (p * cap + r) * K + k0 + k, src)
        c = tl.arange(0, BC)
        cm = c < NC
        meta = tl.load(AMETA32 + c.to(tl.int64) * T + tok, cm & real, 256)      # padding row: {0, 1, 0, 0}
        tl.store(AM32 + c.to(tl.int64) * cap + r, meta, cm)
        g = tl.arange(0, BG)
        gm = g < G
        e = tl.load(EA + tok * G + g, gm & real, 0)
        tl.store(EAG + r * G + g, e, gm)


@tr.jit
def _activate_rows(GU, SORTED, LIVE, WEIGHTS, TABLE, H16, count, I: tl.constexpr, BLOCK: tl.constexpr):
    # GU bf16 [cap, 2 I] (the kernel's gate/up rows: [gate | up]); WEIGHTS f32 [count] (routing weights by route);
    # H16 i16 [cap, I] (the BF16 bits of the activation). The arithmetic is _activate_routes': the SiLU table on the
    # gate's BF16 bits, the up clamp, the routing weight, the BF16 rounding; a padding row is the zero row.
    r = tl.program_id(0).to(tl.int64)
    if r < tl.load(LIVE).to(tl.int64):
        route = tl.load(SORTED + r)
        real = route < count
        weight = tl.load(WEIGHTS + tl.where(real, route, 0))
        for c0 in range(0, I, BLOCK):
            col = c0 + tl.arange(0, BLOCK)
            cm = col < I
            gate = tl.load(GU + r * (2 * I) + col, cm & real, 0.).to(tl.float32)
            up = tl.load(GU + r * (2 * I) + I + col, cm & real, 0.).to(tl.float32)
            si = tl.load(TABLE + (gate.to(tl.uint32, bitcast=True) >> 16))
            value = _bf((si * tl.minimum(10., tl.maximum(-10., up))) * weight)
            value = tl.where(real, value, 0.)
            tl.store(H16 + r * I + col, (value.to(tl.uint32, bitcast=True) >> 16).to(tl.int16), cm)


@tr.jit
def _scatter_rows(Y16, SORTED, LIVE, ROUTES16, count, H: tl.constexpr, BLOCK: tl.constexpr):
    # Y16 i16 [cap, H] (the kernel's BF16 down rows as bits) -> ROUTES16 i16 [count, H] at the real routes.
    r = tl.program_id(0).to(tl.int64)
    if r < tl.load(LIVE).to(tl.int64):
        route = tl.load(SORTED + r)
        if route < count:
            for c0 in range(0, H, BLOCK):
                col = c0 + tl.arange(0, BLOCK)
                cm = col < H
                v = tl.load(Y16 + r * H + col, cm, 0)
                tl.store(ROUTES16 + route.to(tl.int64) * H + col, v, cm)


# ----------------------------------------------------------------------------- launches (all on the current stream)

def quant_codes_rows(x, codes, ea, live=None):
    rows, K = x.shape
    G = K // 32
    assert codes.shape == (rows, K) and ea.shape == (rows, G) and x.dtype == torch.bfloat16
    _quant_codes_rows[(rows,)](x, codes, ea, live if live is not None else ea, G, 8, live is not None,
                               enable_fp_fusion=False, num_warps=4)


def weight_exponents(s, ewg, ewr, bad):
    """s u8 [E, N, G] -> ewg i8 [E, N, G], ewr i8 [E, N]; bad i32 [1] accumulates the admission violations."""
    E, N, G = s.shape
    _weight_exponents[(E * N,)](s, ewg, ewr, bad, G, tr.next_power_of_2(G), num_warps=1)


def row_offsets(expert_ids, ntp, row_off, bad, n, cap, max_rows, BM=64):
    nblocks = expert_ids.numel()
    _row_offsets[(1,)](expert_ids, ntp, row_off, bad, n, nblocks, cap, max_rows, tr.next_power_of_2(nblocks), BM, num_warps=4)


def gather_rows(planes, ameta, ea, sorted_ids, live, P, AM, EAg, first, count, topk):
    AP, T, K = planes.shape
    NC = ameta.shape[0]
    G = K // 32
    cap = P.shape[1]
    assert K % 1024 == 0
    _gather_rows[(cap,)](planes, ameta.view(torch.int32), ea, sorted_ids, live, P, AM.view(torch.int32), EAg,
                         first, count, topk, T, cap, K, G, NC, AP, 1024, tr.next_power_of_2(G),
                         tr.next_power_of_2(NC), num_warps=4)


def activate_rows(gu, sorted_ids, live, weights_flat, table, h, count):
    cap, twoI = gu.shape
    I = twoI // 2
    _activate_rows[(cap,)](gu, sorted_ids, live, weights_flat, table, h.view(torch.int16), count, I, 1024,
                           enable_fp_fusion=False, num_warps=4)


def scatter_rows(y, sorted_ids, live, routes_flat, count):
    cap, H = y.shape
    _scatter_rows[(cap,)](y.view(torch.int16), sorted_ids, live, routes_flat.view(torch.int16), count, H, 1024,
                          num_warps=4)
