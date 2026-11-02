"""NumPy instruction-level model of patha_decode.cu (kernel 1 of DESIGN_r1.md). No torch, no GPU.

Mirrors the kernel's exact partitioning and integer arithmetic: which lane holds which 16 elements (lane_role,
lane_elements), the two-lane shuffle combine into the group dot, the per-group epilogue in the kernel's chunk and
group order with int64 two's-complement accumulation and the kernel's overflow test, the split rule, the merge in
split index order, and the finish (one RNE int64 -> f32, exact power-of-two scale on the exponent field, RNE to BF16
as integer bits). Every value the model computes must equal contract_a (the oracle) bit for bit;
test_patha_decode_local.py checks the BF16 outputs, the raw split partials and the operand bounds the device
instructions rely on.

Contract fixes r1 (CONTRACT_FIXES_r1.md): the row base margin is per kind (MARGIN_DENSE = 13, MARGIN_EXPERT = 20;
H1), the finish follows the one finish rule (overflow -> flagged signed infinity, underflow -> flagged signed zero;
M1), the right shift is clamped at 63 (M5), and a per-term metadata check (shift above the margin means ea_g > EA
or ew_g > EW: a stale exponent table) raises the meta flag (L5).
"""
import numpy as np

MARGIN_DENSE = 13
MARGIN_EXPERT = 20
SA = 9
SW_DENSE = 9
SW_EXPERT = 1
BLOCK_K = 32
BPC = 8                   # groups per chunk
WARPS = 2
ROWS_PER_WARP = 16
BN = WARPS * ROWS_PER_WARP
LANES_PER_GROUP = 2
PER_LANE = BLOCK_K // LANES_PER_GROUP     # 16 elements
KIND_DENSE, KIND_EXPERT = 0, 1
# error_flag() bits of the .cu
FLAG_OVERFLOW, FLAG_FINISH, FLAG_NAN_CODE, FLAG_META = 1, 2, 4, 8


def margin_of(kind):
    return MARGIN_DENSE if kind == KIND_DENSE else MARGIN_EXPERT


# ----------------------------------------------------------------------------- geometry (as the .cu)

def cdiv(a, b):
    return -(-a // b)


def tiles_per_split(K, S):
    return cdiv(cdiv(K, BLOCK_K), S)


def split_ranges(K, S):
    """Group ranges [t0, t1) per split: route 3's rule, t0 = s * tps, t1 = min(t0 + tps, G)."""
    G = cdiv(K, BLOCK_K)
    tps = tiles_per_split(K, S)
    return [(s * tps, min((s + 1) * tps, G)) for s in range(S)]


def chunks(t0, t1):
    """Chunks of a split as (t_first, nb) in issue order; nb <= BPC, chunks never cross the split."""
    nblk = max(t1 - t0, 0)
    return [(t0 + c * BPC, min(BPC, nblk - c * BPC)) for c in range(cdiv(nblk, BPC))]


def lane_role(lane):
    """lane -> (row within the warp, half of the group, partner lane) exactly as gemv_kernel computes them."""
    row = (lane & 7) | ((lane >> 4) << 3)
    half = (lane >> 3) & 1
    return row, half, lane ^ 8


def lane_elements(warp, lane, g):
    """(row within the CTA, the 16 k indices) this lane multiplies for group g."""
    row, half, _ = lane_role(lane)
    k0 = g * BLOCK_K + half * PER_LANE
    return warp * ROWS_PER_WARP + row, list(range(k0, k0 + PER_LANE))


def swz(row, chunk16):
    return chunk16 ^ (row & 7)


# ----------------------------------------------------------------------------- arithmetic (as the .cu)

def add_ovf(a, t):
    """Two's-complement int64 add with the kernel's overflow test; returns (sum, overflow mask < 0 where overflow)."""
    with np.errstate(over="ignore"):
        r = (a.astype(np.int64) + t.astype(np.int64)).astype(np.int64)
    return r, ((a ^ r) & (t ^ r))


def shift_term(D, sh):
    """sh >= 0: D << sh (sh <= margin by the metadata check, |D| < 2^41); sh < 0: floor(D / 2^-sh) with the shift
    clamped at 63 (exact: |D| < 2^41 floors to 0 or -1 at any shift of 41 or more)."""
    D = D.astype(np.int64)
    sh = sh.astype(np.int64)
    with np.errstate(over="ignore"):
        left = (D.astype(np.uint64) << np.maximum(sh, 0).clip(0, 63).astype(np.uint64)).astype(np.int64)
    right = D >> np.minimum(-sh, 63).clip(0, 63)
    return np.where(sh >= 0, left, right)


def int_to_f32_rne(a):
    """Exact integer -> float32, one RNE (__ll2float_rn); the explicit form, used to cross-check f32_from_int64."""
    a = int(a)
    s = -1 if a < 0 else 1
    a = abs(a)
    if a < (1 << 24):
        return np.float32(s * a)
    k = a.bit_length() - 24
    mant, rem = a >> k, a & ((1 << k) - 1)
    half = 1 << (k - 1)
    if rem > half or (rem == half and (mant & 1)):
        mant += 1
    return np.float32(s) * np.ldexp(np.float32(mant), k)


def f32_from_int64(A):
    """Pin (a): one direct int64 -> FP32 RNE per element (__ll2float_rn on the device, cvtsi2ss here); never via float64."""
    return np.asarray(A, dtype=np.int64).astype(np.float32)


def finish_bits(A, e2):
    """A int64 [M,N], e2 = B - SA - SW int [M,N] -> (bf16 bits uint16 [M,N], bad mask). finish_bf16 of the .cu:
    f = ll2float_rn(A); A == 0 -> +0; ex = biased exponent of f + e2; 1..254 -> the normal f * 2^e2 (exponent field
    set); ex > 254 -> signed infinity, bad; ex < 1 -> signed zero, bad. Then the integer RNE to BF16 (contract_v2.bf)."""
    A = np.asarray(A, np.int64)
    e2 = np.broadcast_to(np.asarray(e2, np.int64), A.shape)
    f = f32_from_int64(A)
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
    bits = ((out + np.uint32(0x7FFF) + ((out >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
    return bits, (over | under)


# ----------------------------------------------------------------------------- the kernels

def gemv_partials(n, ea, EA, m, ew, EW, kind, S):
    """gemv_kernel (non-fused): -> (P int64 [S, M, N], stats). n int [M,K], ea int [M,G], EA int [M]; m int [N,K],
    ew int [N,G] (dense: every row of a 32-row block carries the block's scale), EW int [N]. EA and EW are taken as
    given (the kernel's inputs); a shift above the margin (ea_g > EA or ew_g > EW) sets stats['meta_bad'] as the
    kernel sets its meta flag (L5)."""
    n = np.asarray(n, np.int64)
    m = np.asarray(m, np.int64)
    M, K = n.shape
    N = m.shape[0]
    G = K // BLOCK_K
    assert N % BN == 0 and K % (BLOCK_K * 4) == 0 and 1 <= M <= 8
    margin = margin_of(kind)
    # lane partials: [M, N, G, half] over the lane's 16 elements, then the shuffle combine over the two halves
    a = n.reshape(M, G, LANES_PER_GROUP, PER_LANE)
    b = m.reshape(N, G, LANES_PER_GROUP, PER_LANE)
    if kind == KIND_DENSE:
        assert int(np.abs(a).max()) < 2 ** 18 and int(np.abs(b).max()) < 2 ** 18
        lane = np.einsum("mghj,nghj->mngh", a, b, dtype=np.int64)              # mad.wide.s32 chain
        assert int(np.abs(lane).max()) < 2 ** 40
    else:
        assert int(np.abs(a).max()) < 2 ** 18 and int(np.abs(b).max()) <= 12
        lane = np.einsum("mghj,nghj->mngh", a.astype(np.int32), b.astype(np.int32), dtype=np.int32)   # IMAD chain
        assert int(np.abs(lane).max()) < 2 ** 26
        lane = lane.astype(np.int64)
    D = lane[..., 0] + lane[..., 1]                                             # shfl.xor combine, exact
    cm = margin - np.asarray(EA, np.int64)[:, None] - np.asarray(EW, np.int64)[None, :]   # [M, N]
    ea = np.asarray(ea, np.int64)
    ew = np.asarray(ew, np.int64)
    P = np.zeros((S, M, N), np.int64)
    ovf = np.zeros((M, N), np.int64)
    # L5, as the kernel checks per staged group: ew_g <= EW[n] (once per group per row) and ea_g <= EA[m]
    meta_bad = bool((ew > np.asarray(EW, np.int64)[:, None]).any()) or bool((ea > np.asarray(EA, np.int64)[:, None]).any())
    for s, (t0, t1) in enumerate(split_ranges(K, S)):
        acc = np.zeros((M, N), np.int64)
        for t_first, nb in chunks(t0, t1):
            for bidx in range(nb):                                              # group order inside the chunk
                g = t_first + bidx
                sh = ea[:, None, g] + ew[None, :, g] + cm                   # <= margin on consistent tables
                acc, o = add_ovf(acc, shift_term(D[:, :, g], sh))
                ovf |= o
        P[s] = acc
    stats = {"max_lane_partial_bits": int(np.abs(lane).max()).bit_length(),
             "max_partial_bits": int(np.abs(P).max()).bit_length(), "overflow": bool((ovf < 0).any()),
             "meta_bad": meta_bad, "margin": margin}
    return P, stats


def merge(P, EA, EW, kind):
    """merge_kernel: int64 sum over the splits in index order with the overflow test; returns (Y bits, flags)."""
    S, M, N = P.shape
    acc = np.zeros((M, N), np.int64)
    ovf = np.zeros((M, N), np.int64)
    for s in range(S):
        acc, o = add_ovf(acc, P[s])
        ovf |= o
    B = np.asarray(EA, np.int64)[:, None] + np.asarray(EW, np.int64)[None, :] - margin_of(kind)
    sasw = SA + (SW_DENSE if kind == KIND_DENSE else SW_EXPERT)
    bits, bad = finish_bits(acc, B - sasw)
    return bits, {"overflow": bool((ovf < 0).any()), "finish_bad": bool(bad.any())}


def gemv(n, ea, m, ew, kind, S):
    """Whole path: EA/EW as the prep kernel and prepare_*_weights derive them; -> (bf16 bits, partials, flags)."""
    EA = np.asarray(ea, np.int64).max(axis=1)
    EW = np.asarray(ew, np.int64).max(axis=1)
    P, st = gemv_partials(n, ea, EA, m, ew, EW, kind, S)
    bits, fl = merge(P, EA, EW, kind)
    fl["overflow"] = fl["overflow"] or st["overflow"]
    return bits, P, {**st, **fl}


# ----------------------------------------------------------------------------- the conversion forms (for the self-test)

def dense_m_arith(c):
    """dense_m_arith of the .cu, vectorised over uint8 codes."""
    c = np.asarray(c, np.uint32)
    e = (c >> 3) & 15
    mant = c & 7
    mag = np.where(e > 0, (8 | mant) << np.maximum(e, 1) - 1, mant).astype(np.int64)
    mag = np.where((c & 127) == 127, 0, mag)
    s = -(c >> 7).astype(np.int64)
    return (mag ^ s) - s


def prmt(a, b, sel):
    """PTX prmt.b32 default mode, vectorised over uint32 arrays: 4 output bytes, selector nibbles from sel[15:0]."""
    a = np.asarray(a, np.uint64)
    b = np.asarray(b, np.uint64)
    sel = np.asarray(sel, np.uint64)
    src = (b << np.uint64(32)) | a
    out = np.zeros(np.broadcast(a, b, sel).shape, np.uint64)
    for i in range(4):
        s = (sel >> np.uint64(4 * i)) & np.uint64(15)
        byte = (src >> ((s & np.uint64(7)) * np.uint64(8))) & np.uint64(0xFF)
        rep = np.where(byte & np.uint64(0x80), np.uint64(0xFF), np.uint64(0))
        val = np.where(s & np.uint64(8), rep, byte)
        out |= val << np.uint64(8 * i)
    return out.astype(np.uint32)


def expert_prmt_form(v, consts):
    """The prmt path of convert_weights<Expert> on a 16-bit nibble window v (elements 0..3): -> int32 [.., 4]."""
    v = np.asarray(v, np.uint32)
    sel = v & np.uint32(0x7777)
    pos = prmt(consts["T0"], consts["T1"], sel)
    neg = prmt(consts["T2"], consts["T3"], sel)
    bm = prmt((v << np.uint32(4)) & np.uint32(0xFFFFFFFF), v, consts["SIGNSEL"])
    packed = (pos & ~bm) | (neg & bm)
    cols = [prmt(packed, np.uint32(0), consts["SEXT%d" % i]).astype(np.int32) for i in range(4)]
    return np.stack(cols, -1)
