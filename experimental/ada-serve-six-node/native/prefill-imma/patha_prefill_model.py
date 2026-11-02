"""NumPy model of patha_prefill.cu (r3-P): the same tile, chunk, plane-group, combine, shift and finish structure,
evaluated on the CPU and required to equal contract_a's oracle bit for bit (test_patha_prefill_local.py).

What is mirrored exactly (names as in patha_sequence.h / patha_prefill.cu):
  * prep_activations: per (row, fold chunk of 4F groups) the AP base-128 digit planes of n << (ea_g - ea_min) when the
    row's spread is within the budget of AP digits (3 bits with three planes, 10 with four; the low digits balanced in
    [-64, 63], the top digit in the full int8 range), else of n; ameta = {ea_min, eligible, EA, 0} per fold chunk.
  * expand_weights_dense / expand_weights_expert: the same per weight row (dense m = +-N_OF_CODE, three planes,
    budget 2; expert w' halves shifted by up to 3 bits, one int8 plane, produced on the device by the prmt byte lookup
    modelled in expand_lut_word); bmeta = {ew_min, eligible, EW, 0} per fold chunk.
  * gemm: CTA tiles of BM = 64 token rows (starting at row_off[e] + 64 t) x BN output rows; the fold decision per
    16 token rows (one m16 MMA tile, aligned to the expert's first row) x 16 output rows (one warp's columns): all live
    token rows and all 16 weight rows eligible, min chunk shift >= 0 (the per-element arithmetic does not depend on
    the warp partition, so the model only mirrors the decision tiles); fold: Q_s = sum_{i+j=s} A_i B_j^T over the fold
    chunk accumulated in int32 (the MMA accumulators), combine lo = Q0 + Q1<<7, mid = Q2 + Q3<<7 (int32),
    Dc = lo + mid<<14 (+ Q4<<28) (int64), acc += Dc << shift with the overflow check; per-group path: the same per
    group with the shift u = ea_eff + ew_eff - B where the effective exponent undoes the shift actually applied to
    the row (ea_min if the row was folded, ea_g otherwise), floor for negative u (right shifts clamped at 63, M5).
  * metadata check (L5): a group exponent above the row maximum the kernel was given (ea_g > EA or ew_g > EW, a
    stale exponent table) or a term shift above the margin sets stats['meta_bad'] as the kernel sets ERR_META (the
    kernel checks the staged exponents once per row per stage; the per-term check of the fold path is implied by
    them, as the test gate fold_shift_check_implied_by_staged_exponent_checks shows).
  * finish: the one finish rule (M1): f32_rne(A) by one direct conversion, the scale 2^(B - SA - SW) applied on the
    exponent field; overflow -> flagged signed infinity, underflow -> flagged signed zero; then bf16 RNE.
The row base margin is per kind: MARGIN_DENSE = 13 (WP = 3), MARGIN_EXPERT = 20 (WP = 1) (H1).
The plane products are evaluated with float32 BLAS matmuls, which are exact here (|digit| <= 127, |plane| <= 96,
K <= 256 per chunk: every partial sum is below 2^24); the int32 bounds the MMA accumulators rely on are asserted on
every call.
"""
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "dense-extension"))
import contract_a as ca  # noqa: E402

v2 = ca.v2

# ---- constants (patha_sequence.h)
MARGIN_DENSE = 13
MARGIN_EXPERT = 20
SA = 9
SW_DENSE = 9
SW_EXPERT = 1
GROUP = 32
BASE_BITS = 7
TOP_DIGIT_MIN, TOP_DIGIT_MAX = -128, 127
ACT_BUDGET = 3             # three digit planes (r3-P; was 2)
ACT_BUDGET_4 = 10          # four digit planes
DENSE_W_BUDGET = 2
EXPERT_W_BUDGET = 3
ACT_PLANES = 3
ACT_PLANES_4 = 4
BM = 64
STAGE_GROUPS = 4
STAGE_K = STAGE_GROUPS * GROUP          # 128: one pipeline stage, one 128-byte plane row
CHUNKS16 = STAGE_K // 16
WARP_M = 32
WARP_N = 16
MT = WARP_M // 16
NT = WARP_N // 8
WARPS_M = 2
DECISION_ROWS = 16                      # fold decision tile: one m16 tile x WARP_N columns
NT_PASS = {3: 1, 1: 2}                  # n8 tiles accumulated per pass: dense one (ldmatrix.x2 for B), expert the pair (x4)
META_BYTES = 4
PLANE_BYTES = BM * STAGE_K              # 8192 activation plane per stage
A_REGION = {3: 3 * PLANE_BYTES + 2 * BM * 4, 4: 4 * PLANE_BYTES + 2 * BM * 4}     # 25088 / 33280
Q_BOUND_BITS = 21          # |Q_s| < 2^21 per stage (top digit 112 x expert plane 96 x 128, or dense 1,966,080)
LO_BOUND_BITS = 30         # |lo|, |mid| < 2^30
# prmt lookup constants of the expert expander
LUT_POS0, LUT_POS1 = 0x03020100, 0x0C080604
LUT_SIGN_SEL_LO, LUT_SIGN_SEL_HI = 0x9D8C, 0xBFAE
LUT_MAG_MASK = 0x7777
# g_error bits (patha_sequence.h)
ERR_ACC_OVERFLOW, ERR_NEG_FOLD_SHIFT, ERR_FINISH_RANGE, ERR_NAN_ACT = 1, 2, 4, 8
ERR_NAN_WEIGHT, ERR_DIGIT_RANGE, ERR_EXPERT_PLANE, ERR_META = 16, 32, 64, 128


# ---- configurations (Cfg<WP, AP, BN, F, S> of patha_sequence.h)

def config(wp, ap=3, fold_groups=4):
    """The kernel configuration the binding selects for (kind, planes, fold chunk): dict with the geometry and the
    shared-memory budget (every number the kernel's static_asserts pin)."""
    if wp == 3:
        assert ap == 3 and fold_groups == 4, "the dense kernel takes three planes and a fold chunk of 4 groups"
        bn, stages = 32, 2
    else:
        assert (ap, fold_groups) in ((3, 4), (3, 8), (4, 4)), "expert configurations: (3, 4), (3, 8), (4, 4)"
        bn, stages = 64, 2
    F = fold_groups // STAGE_GROUPS
    warps_n = bn // WARP_N
    warps = WARPS_M * warps_n
    b_region = wp * bn * STAGE_K + 2 * bn * 4
    wcode = bn * STAGE_K if wp == 3 else bn * STAGE_K // 2
    w_region = wcode + bn * 4 * F
    stage_bytes = A_REGION[ap] + b_region + w_region
    return dict(wp=wp, ap=ap, F=F, fold_groups=fold_groups, chunk_k=STAGE_K * F, bn=bn, stages=stages,
                warps_n=warps_n, warps=warps, threads=32 * warps, w_pieces=wcode // bn // 16,
                a_region=A_REGION[ap], b_region=b_region, w_region=w_region, wcode_bytes=wcode,
                stage_bytes=stage_bytes, smem=stage_bytes * stages, budget=act_budget(ap),
                w_budget=DENSE_W_BUDGET if wp == 3 else EXPERT_W_BUDGET, margin=margin_of(wp), ns=ap + wp - 1,
                nt_pass=NT_PASS[wp])


CFG_DENSE = dict(wp=3, ap=3, fold_groups=4)
CFG_GATE_UP = dict(wp=1, ap=3, fold_groups=4)     # r3-P2: the same instantiation as down (Codex's clean build)
CFG_GATE_UP_F2 = dict(wp=1, ap=3, fold_groups=8)  # the 8-group fold-chunk knob: modelled and gated, not instantiated
CFG_DOWN = dict(wp=1, ap=3, fold_groups=4)
CFG_DOWN4 = dict(wp=1, ap=4, fold_groups=4)


def act_budget(ap):
    return ACT_BUDGET if ap == 3 else ACT_BUDGET_4


def margin_of(wp):
    """The row-base margin of the kernel instantiation: WP = 3 dense (13), WP = 1 expert (20)."""
    return MARGIN_DENSE if wp == 3 else MARGIN_EXPERT


def margin_of_sw(SW):
    return MARGIN_DENSE if SW == SW_DENSE else MARGIN_EXPERT


def smem_bytes(wp, ap=3, fold_groups=4):
    return config(wp, ap, fold_groups)["smem"]


def swz(row, piece16):
    return piece16 ^ (row & 7)


def swz_w(row, piece16, npieces):
    return piece16 ^ (row & (npieces - 1))


# ---- digit and code helpers (patha_sequence.h)

def digit0(x):
    return ((x + 64) & 127) - 64


def carry(x, d):
    return (x - d) >> 7


def digits3(x):
    """x int64 array, |x| <= 2,089,024 -> (d0, d1, d2) int8-valued arrays; raises if the top digit leaves [-128, 127]."""
    x = np.asarray(x, dtype=np.int64)
    d0 = digit0(x)
    r1 = carry(x, d0)
    d1 = digit0(r1)
    d2 = carry(r1, d1)
    if np.any(d2 < TOP_DIGIT_MIN) or np.any(d2 > TOP_DIGIT_MAX):
        raise OverflowError("top digit out of range")
    return d0, d1, d2


def digits4(x):
    """x int64 array, |x| <= 267,386,944 -> (d0, d1, d2, d3); raises if the top digit leaves [-128, 127]."""
    x = np.asarray(x, dtype=np.int64)
    d0 = digit0(x)
    r1 = carry(x, d0)
    d1 = digit0(r1)
    r2 = carry(r1, d1)
    d2 = digit0(r2)
    d3 = carry(r2, d2)
    if np.any(d3 < TOP_DIGIT_MIN) or np.any(d3 > TOP_DIGIT_MAX):
        raise OverflowError("top digit out of range")
    return d0, d1, d2, d3


def e4m3_mag(c):
    c = np.asarray(c, dtype=np.int64)
    e = (c >> 3) & 15
    mant = c & 7
    mag = np.where(e == 0, mant, (8 | mant) << np.maximum(e - 1, 0))
    return np.where((c & 127) == 127, 0, mag)


def e4m3_int(c):
    """code byte -> signed integer n (NaN -> 0, as the kernel; the kernel also sets a flag)."""
    c = np.asarray(c, dtype=np.int64)
    return np.where(c & 128, -e4m3_mag(c), e4m3_mag(c))


def e2m1_halves(nib):
    nib = np.asarray(nib, dtype=np.int64)
    mag = nib & 7
    m = np.where(mag < 2, mag, (2 + (mag & 1)) << np.maximum((mag >> 1) - 1, 0))
    return np.where(nib & 8, -m, m)


# ---- the device's byte-lookup expansion of E2M1 nibbles (expand_w<WP = 1> of the .cu)

def prmt(x, y, sel):
    """prmt.b32 default mode (the PTX instruction, as the kernel's inline prmt_b32): result byte i = byte sel_i[2:0]
    of the eight bytes {x[0..3], y[4..7]}; with sel_i bit 3 set the selected byte's msb is replicated over the result
    byte. CUDA's __byte_perm intrinsic is NOT this: it masks bit 3 of every selector nibble (Codex's r3 report), so the
    kernel may use it only with selectors whose nibble bit 3 is clear (the magnitude lookups, masked to 0x7777).
    x, y, sel: uint32 arrays."""
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    sel = np.asarray(sel, dtype=np.uint64)
    src = x | (y << np.uint64(32))
    out = np.zeros(np.broadcast(src, sel).shape, dtype=np.uint64)
    for i in range(4):
        n = (sel >> np.uint64(4 * i)) & np.uint64(15)
        byte = (src >> ((n & np.uint64(7)) * np.uint64(8))) & np.uint64(0xFF)
        rep = np.where(byte & np.uint64(0x80), np.uint64(0xFF), np.uint64(0))
        byte = np.where(n & np.uint64(8), rep, byte)
        out |= byte << np.uint64(8 * i)
    return out.astype(np.uint32)


def expand_lut_word(v, s):
    """One 32-bit code word v (eight E2M1 nibbles, element 2b in the low nibble of byte b) and the row-group shift s
    (0..3) -> two plane words (elements 0-3, 4-7) of int8 bytes e2m1_halves(nib) << s, by the kernel's prmt sequence."""
    v = np.asarray(v, dtype=np.uint32)
    s = np.asarray(s, dtype=np.uint32)
    T0 = (np.uint32(LUT_POS0) << s).astype(np.uint32)
    T1 = (np.uint32(LUT_POS1) << s).astype(np.uint32)
    T2 = ((np.uint32(0x80808080) - T0) ^ np.uint32(0x80808080)).astype(np.uint32)
    T3 = ((np.uint32(0x80808080) - T1) ^ np.uint32(0x80808080)).astype(np.uint32)
    vs = (v << np.uint32(4)).astype(np.uint32)
    sel_lo = v & np.uint32(LUT_MAG_MASK)
    sel_hi = (v >> np.uint32(16)) & np.uint32(LUT_MAG_MASK)
    m_lo = prmt(v, vs, np.uint32(LUT_SIGN_SEL_LO))
    m_hi = prmt(v, vs, np.uint32(LUT_SIGN_SEL_HI))
    pos_lo, neg_lo = prmt(T0, T1, sel_lo), prmt(T2, T3, sel_lo)
    pos_hi, neg_hi = prmt(T0, T1, sel_hi), prmt(T2, T3, sel_hi)
    lo = (pos_lo & ~m_lo) | (neg_lo & m_lo)
    hi = (pos_hi & ~m_hi) | (neg_hi & m_hi)
    return lo.astype(np.uint32), hi.astype(np.uint32)


def word_bytes_int8(w):
    """uint32 array -> int8 array [..., 4] (byte 0 first)."""
    w = np.asarray(w, dtype=np.uint32)
    return np.stack([((w >> np.uint32(8 * i)) & np.uint32(0xFF)).astype(np.uint8).astype(np.int8) for i in range(4)], -1)


# ---- operand preparation

def _planes_of(x_int, e, budget, nplanes, fold_groups, expert_plane=False):
    """x_int [R, K] integers, e [R, G] exponents -> (planes [P, R, K] int8, meta [NC, R, 4] int8, applied [R, G]) with
    the fold chunk of fold_groups groups."""
    R, K = x_int.shape
    G = K // GROUP
    CK = fold_groups * GROUP
    assert K % CK == 0, "K must be a multiple of the fold chunk"
    NC = K // CK
    e = np.asarray(e, dtype=np.int64)
    ec = e.reshape(R, NC, fold_groups)
    emin = ec.min(2)                                             # [R, NC]
    emax = ec.max(2)
    elig = (emax - emin) <= budget
    Erow = e.max(1)                                              # [R]
    shift = np.where(elig[:, :, None], ec - emin[:, :, None], 0)   # [R, NC, fold_groups]
    x = x_int.reshape(R, NC, fold_groups, GROUP) << shift[:, :, :, None]
    x = x.reshape(R, K)
    if expert_plane:
        if np.any(np.abs(x) > 127):
            raise OverflowError("shifted expert weight outside int8")
        planes = x.astype(np.int8)[None]
    elif nplanes == 3:
        d0, d1, d2 = digits3(x)
        planes = np.stack([d0, d1, d2]).astype(np.int8)
    else:
        d0, d1, d2, d3 = digits4(x)
        planes = np.stack([d0, d1, d2, d3]).astype(np.int8)
    meta = np.zeros((NC, R, META_BYTES), dtype=np.int8)
    meta[:, :, 0] = emin.T
    meta[:, :, 1] = elig.T
    meta[:, :, 2] = Erow[None, :]
    return planes, meta, shift.reshape(R, G)


def prep_activations(codes, ea, planes=3, fold_groups=4, rows=None, out=None):
    """E4M3 code bytes [M, K], ea [M, G] -> planes [AP, M, K] int8, ameta [K / (32 fold_groups), M, 4] int8
    (kernel prep_kernel<AP, F>). rows (r3-P3): the live row count; rows at or beyond it are skipped. With out =
    (planes, ameta) buffers the live rows are written into them and the other rows left untouched, as the kernel
    leaves a workspace; without out, the skipped rows of the returned arrays are zero."""
    codes = np.asarray(codes)
    M, K = codes.shape
    live = M if rows is None else max(0, min(M, int(rows)))
    NC = K // (32 * fold_groups)
    if out is None:
        pl = np.zeros((planes, M, K), np.int8)
        meta = np.zeros((NC, M, META_BYTES), np.int8)
    else:
        pl, meta = out
        assert pl.shape == (planes, M, K) and meta.shape == (NC, M, META_BYTES)
    if live > 0:
        n = e4m3_int(codes[:live])
        pl_l, meta_l, _ = _planes_of(n, np.asarray(ea)[:live], act_budget(planes), planes, fold_groups)
        pl[:, :live] = pl_l
        meta[:, :live] = meta_l
    return pl, meta


def expand_weights_dense(wcodes, ew, fold_groups=4):
    """E4M3 code bytes [N, K], ew [N, G] -> planes [3, N, K] int8, bmeta [NC, N, 4] (kernel expand_w, dense)."""
    m = e4m3_int(wcodes)
    planes, meta, _ = _planes_of(m, ew, DENSE_W_BUDGET, 3, fold_groups)
    return planes, meta


def expand_weights_expert(wnib, ew, fold_groups=4):
    """E2M1 nibble bytes [N, K/2], ew [N, G] -> plane [1, N, K] int8, bmeta [NC, N, 4] (kernel expand_w, expert)."""
    wnib = np.asarray(wnib, dtype=np.uint8)
    codes = np.stack((wnib & 15, wnib >> 4), -1).reshape(wnib.shape[0], -1)   # element 2b = low nibble of byte b
    m = e2m1_halves(codes)
    planes, meta, _ = _planes_of(m, ew, EXPERT_W_BUDGET, 1, fold_groups, expert_plane=True)
    return planes, meta


# ---- the GEMM

def _row_tiles(M, row_off, rows):
    """Tiles of `rows` token rows exactly as the kernel's grid (rows = BM) or its decision tiles (rows = 16):
    (expert, row0, nrows) per tile, restarting at every expert's first row."""
    if row_off is None:
        row_off = [0, M]
    tiles = []
    for e in range(len(row_off) - 1):
        lo, hi = int(row_off[e]), int(row_off[e + 1])
        r = lo
        while r < hi:
            tiles.append((e, r, min(rows, hi - r)))
            r += rows
    return tiles


def _combine(Q, ns, stats, F):
    """Q list of int64 arrays Q_0..Q_{ns-1}; mirrors combine<Cfg> and asserts the int32 ranges."""
    for q in Q:
        stats["max_q_bits"] = max(stats["max_q_bits"], int(np.abs(q).max()).bit_length() if q.size else 0)
        assert np.abs(q).max() < (1 << (Q_BOUND_BITS + F - 1)), "plane-group accumulator leaves the int32 bound"
    lo = Q[0] + (Q[1] << 7)
    assert np.abs(lo).max() < (1 << LO_BOUND_BITS)
    if ns == 5:
        mid = Q[2] + (Q[3] << 7)
        assert np.abs(mid).max() < (1 << LO_BOUND_BITS)
        return lo + (mid << 14) + (Q[4] << 28)
    if ns == 4:
        mid = Q[2] + (Q[3] << 7)
        assert np.abs(mid).max() < (1 << LO_BOUND_BITS)
        return lo + (mid << 14)
    return lo + (Q[2] << 14)


def _checked_add(acc, term, stats, mask=None):
    with np.errstate(over="ignore"):
        r = acc + term                                           # int64 wraps silently in NumPy
    ovf = ((acc ^ r) & (term ^ r)) < 0
    if mask is not None:
        ovf = ovf & mask
    if np.any(ovf):
        stats["overflow"] += int(ovf.sum())
    return r


def _shift_term(x, s):
    """The oracle's per-term shift as the kernel's shift_term: left for s >= 0 (s <= margin by the metadata check,
    clamped at 63), floor (arithmetic right) for s < 0 with the right shift clamped at 63 (M5; exact: |x| < 2^47 here)."""
    s = np.asarray(s, np.int64)
    pos = np.where(s >= 0, s, 0).clip(0, 63)
    neg = np.where(s < 0, -s, 0).clip(0, 63)
    with np.errstate(over="ignore"):
        left = (np.asarray(x, np.int64).astype(np.uint64) << pos.astype(np.uint64)).astype(np.int64)
    return np.where(s >= 0, left, np.asarray(x, np.int64) >> neg)


def gemm(aplanes, ameta, ea, bplanes, bmeta, ew, wp, row_off=None, stats=None, rows=None):
    """Returns the int64 row sums A [M, N] (out_i64 of the kernel) and stats. aplanes [AP, M, K], ameta [NC, M, 4],
    ea [M, G], bplanes [wp, N, K] (one expert), bmeta [NC, N, 4], ew [N, G]; the fold chunk (4 or 8 groups) is read
    from ameta's leading dimension; row_off partitions the rows into experts that all use the same weights here (the
    kernel indexes per-expert weights by blockIdx.z; the arithmetic per tile is identical, so the model keeps one
    weight set and only mirrors the row partition). EA and EW are read from the metadata exactly as the kernel reads
    them (a tampered table must raise meta_bad, not change bits silently). The decision tiles are 16 token rows
    (aligned to the expert's first row) x 16 output rows. rows (r3-P3): the live row count of an ungrouped launch;
    rows at or beyond it are neither computed nor stored (zero here; untouched on the device)."""
    if stats is None:
        stats = {}
    stats.setdefault("folded", 0)
    stats.setdefault("fallback_budget", 0)
    stats.setdefault("fallback_negative_shift", 0)
    stats.setdefault("max_q_bits", 0)
    stats.setdefault("overflow", 0)
    stats.setdefault("meta_bad", False)
    if rows is not None:
        assert row_off is None, "rows applies to the ungrouped launch; a grouped launch is bounded by row_off"
        live = max(0, min(aplanes.shape[1], int(rows)))
        stats["margin"] = margin_of(wp)
        stats["config"] = (wp, aplanes.shape[0], (aplanes.shape[2] // ameta.shape[0]) // GROUP)
        if live == 0:
            return np.zeros((aplanes.shape[1], bplanes.shape[1]), np.int64), stats
        A_live, stats = gemm(aplanes[:, :live], ameta[:, :live], np.asarray(ea)[:live], bplanes, bmeta, ew, wp, None, stats, None)
        A = np.zeros((aplanes.shape[1], bplanes.shape[1]), np.int64)
        A[:live] = A_live
        return A, stats
    AP, M, K = aplanes.shape
    N = bplanes.shape[1]
    G = K // GROUP
    NC = ameta.shape[0]
    assert K % NC == 0 and (K // NC) % STAGE_K == 0 and (K // NC) // STAGE_K in (1, 2), "fold chunk of one or two stages"
    CK = K // NC
    F = CK // STAGE_K
    CG = F * STAGE_GROUPS
    assert AP in (3, 4) and bplanes.shape[0] == wp and K % CK == 0
    assert bmeta.shape[0] == NC, "weight metadata must use the same fold chunk"
    cfg = config(wp, AP, CG)
    assert N % cfg["bn"] == 0, "N must be a multiple of the CTA's output rows"
    MARGIN = cfg["margin"]
    stats["margin"] = MARGIN
    stats["config"] = (wp, AP, CG)
    ea = np.asarray(ea, dtype=np.int64)
    ew = np.asarray(ew, dtype=np.int64)
    EA = ameta[0, :, 2].astype(np.int64)
    EW = bmeta[0, :, 2].astype(np.int64)
    # L5: the expander checks every staged weight-row group exponent against the row's EW; each lane checks the
    # stage's activation group exponents of one row against the row's EA
    if bool((ew > EW[:, None]).any()) or bool((ea > EA[:, None]).any()):
        stats["meta_bad"] = True
    Bbase = EA[:, None] + EW[None, :] - MARGIN                   # [M, N]
    A = np.zeros((M, N), dtype=np.int64)
    af = aplanes.astype(np.float32)
    bf = bplanes.astype(np.float32)
    NS = AP + wp - 1
    # decision tiles: 16 token rows per expert, 16 output rows
    tiles = _row_tiles(M, row_off, DECISION_ROWS)
    starts = np.array([r0 for (_, r0, _) in tiles], np.int64)
    tile_of_row = np.zeros(M, np.int64)
    for t, (_, r0, nr) in enumerate(tiles):
        tile_of_row[r0:r0 + nr] = t
    NCT = N // WARP_N
    col_tile = np.repeat(np.arange(NCT), WARP_N)
    stats["decision_tiles_per_chunk"] = len(tiles) * NCT
    for c in range(NC):
        k0, k1 = c * CK, (c + 1) * CK
        ea_min = ameta[c, :, 0].astype(np.int64)
        a_el = ameta[c, :, 1].astype(bool)
        ew_min = bmeta[c, :, 0].astype(np.int64)
        w_el = bmeta[c, :, 1].astype(bool)
        da = ea_min - EA                                         # [M] (<= 0)
        dw = ew_min - EW                                         # [N]
        # chunk-level plane-group products (the fold path's MMA accumulators), exact in float32
        Q = [np.zeros((M, N), dtype=np.float32) for _ in range(NS)]
        for i in range(AP):
            for j in range(wp):
                Q[i + j] += af[i, :, k0:k1] @ bf[j, :, k0:k1].T
        Q = [q.astype(np.int64) for q in Q]
        Dc = _combine(Q, NS, stats, F)
        shift = da[:, None] + dw[None, :] + MARGIN               # [M, N]
        # the decision per (16-row tile, 16-column tile): all rows and columns eligible, min chunk shift >= 0
        a_ok_t = np.logical_and.reduceat(a_el, starts)           # [T]
        da_min_t = np.minimum.reduceat(da, starts)
        w_ok_c = w_el.reshape(NCT, WARP_N).all(1)                # [NCT]
        dw_min_c = dw.reshape(NCT, WARP_N).min(1)
        ok_tc = a_ok_t[:, None] & w_ok_c[None, :]
        fold_tc = ok_tc & (da_min_t[:, None] + dw_min_c[None, :] + MARGIN >= 0)
        stats["folded"] += int(fold_tc.sum())
        stats["fallback_negative_shift"] += int((ok_tc & ~fold_tc).sum())
        stats["fallback_budget"] += int((~ok_tc).sum())
        fold_el = fold_tc[tile_of_row][:, col_tile]              # [M, N]
        if bool((shift[fold_el] > MARGIN).any()):
            stats["meta_bad"] = True
        A_new = _checked_add(A, _shift_term(Dc, shift), stats, fold_el)
        if not bool(fold_tc.all()):
            # per-group path for the other tiles: the same MMAs per group with accumulators from zero, the combine per
            # group, then the oracle's term with effective exponents
            pg = ~fold_el
            ea_c = ea[:, c * CG:(c + 1) * CG]
            ew_c = ew[:, c * CG:(c + 1) * CG]
            ea_eff = np.where(a_el[:, None], ea_min[:, None], ea_c)  # [M, CG]
            ew_eff = np.where(w_el[:, None], ew_min[:, None], ew_c)  # [N, CG]
            acc = A.copy()
            for g in range(CG):
                kg0, kg1 = k0 + g * GROUP, k0 + (g + 1) * GROUP
                Qg = [np.zeros((M, N), dtype=np.float32) for _ in range(NS)]
                for i in range(AP):
                    for j in range(wp):
                        Qg[i + j] += af[i, :, kg0:kg1] @ bf[j, :, kg0:kg1].T
                Dg = _combine([q.astype(np.int64) for q in Qg], NS, stats, 1)
                u = ea_eff[:, g][:, None] + ew_eff[:, g][None, :] - Bbase
                if bool((u[pg] > MARGIN).any()):
                    stats["meta_bad"] = True
                acc = _checked_add(acc, _shift_term(Dg, u), stats, pg)
            A_new = np.where(fold_el, A_new, acc)
        A = A_new
    return A, stats


def finish_with_flag(A, EA, EW, SW):
    """The finish rule (M1), as finish_bf16 of the .cu: f = f32_rne(A) by one direct int64 -> FP32 conversion;
    B = EA + EW - margin; e2 = B - SA - SW; A == 0 -> +0; biased exponent of f plus e2 in 1..254 -> the normal
    f * 2^e2; above -> signed infinity, flagged; below -> signed zero, flagged; then bf16 RNE (contract_v2.bf).
    -> (y float32 BF16-valued [M, N], flagged bool [M, N])."""
    A = np.asarray(A, dtype=np.int64)
    f = A.astype(np.float32)                                     # one RNE (cvtsi2ss / __ll2float_rn), never via float64
    B = np.asarray(EA).astype(np.int64)[:, None] + np.asarray(EW).astype(np.int64)[None, :] - margin_of_sw(SW)
    e2 = np.broadcast_to(B - SA - SW, f.shape)
    u = f.view(np.uint32)
    nz = f != 0
    sign = u & np.uint32(0x80000000)
    ex = ((u >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64) + e2
    over = nz & (ex > 254)
    under = nz & (ex < 1)
    ok = nz & ~over & ~under
    out = u.copy()
    out[ok] = (u[ok] & np.uint32(0x807FFFFF)) | (ex[ok].astype(np.uint32) << np.uint32(23))
    out[over] = sign[over] | np.uint32(0x7F800000)
    out[under] = sign[under]
    return v2.bf(out.view(np.float32)), (over | under)


def finish(A, EA, EW, SW):
    """bf16 of the finish rule; the flag is unreachable on legal inputs (finish_with_flag reports it)."""
    return finish_with_flag(A, EA, EW, SW)[0]


# ---- end-to-end helpers for the tests

def dense_linear(codes, ea, wcodes, ew, row_off=None, stats=None):
    """Full model of the dense path from E4M3 code bytes: returns (y bf16-valued float32, A int64, stats)."""
    ap, am = prep_activations(codes, ea, 3, 4)
    bp, bm = expand_weights_dense(wcodes, ew, 4)
    A, st = gemm(ap, am, ea, bp, bm, ew, 3, row_off=row_off, stats=stats)
    y, bad = finish_with_flag(A, am[0, :, 2], bm[0, :, 2], SW_DENSE)
    st["finish_bad"] = bool(bad.any())
    return y, A, st


def expert_linear(codes, ea, wnib, ew, row_off=None, stats=None, planes=3, fold_groups=4):
    """Full model of an expert projection: planes 3 / fold chunk 4 (gate/up and down), 4 / 4 (the down knob);
    3 / 8 is the modelled 8-group knob."""
    ap, am = prep_activations(codes, ea, planes, fold_groups)
    bp, bm = expand_weights_expert(wnib, ew, fold_groups)
    A, st = gemm(ap, am, ea, bp, bm, ew, 1, row_off=row_off, stats=stats)
    y, bad = finish_with_flag(A, am[0, :, 2], bm[0, :, 2], SW_EXPERT)
    st["finish_bad"] = bool(bad.any())
    return y, A, st
