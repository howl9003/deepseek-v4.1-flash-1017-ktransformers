"""NumPy instruction-level model of patha_expert_static.cu (r3-E step 1). No torch, no GPU.

Follows the kernel's partitioning: the route table `active`, the CTA-per-route leader rule and slot gather, the
in-prologue EA (per slot) and EW (per row, with the scale admission check), the chunk staging windows, the
cooperative code -> int32 conversion (dense_m_arith, with the NaN flag), r2's inner loop (lane partition, shuffle
combine, per-group floor shift and checked add in ascending group order), the S = 1 fused finish under the finish
rule, the per-route output rows (gate/up as BF16 bits widened to FP32, down as BF16 bits) and the flag word. The
arithmetic helpers are imported from the r2 model (patha_decode_model.py, decode-gemv folder): the same functions
that equal contract_a bit for bit there. Every value computed here must equal contract_a.expert / contract_v2.expert
bit for bit; test_patha_expert_static_local.py checks it.

The quantiser model `quant_codes` transliterates the Triton `_quant_codes` arithmetic line for line (float32 ops)
and must equal contract_v2.quant's codes and ea bit for bit (gate quant_codes_model_equals_contract_quant).
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'decode-gemv'))
import patha_decode_model as r2m  # noqa: E402
from patha_decode_model import (add_ovf, shift_term, finish_bits, dense_m_arith, lane_role, swz,  # noqa: E402,F401
                                f32_from_int64, int_to_f32_rne, prmt, expert_prmt_form)

MARGIN_EXPERT = 20
SA = 9
SW_EXPERT = 1
BLOCK_K = 32
WARPS = 2
ROWS_PER_WARP = 16
BN = WARPS * ROWS_PER_WARP
THREADS = 32 * WARPS
LANES_PER_GROUP = 2
PER_LANE = BLOCK_K // LANES_PER_GROUP
MAX_M = 8
MAX_TOPK = 6
MAX_COUNT = THREADS
MIN_SCALE_CODE = 53
BPC_DEFAULT = 8
EXP_ROW_BYTES = 16
FLAG_OVERFLOW, FLAG_FINISH, FLAG_NAN_CODE, FLAG_META, FLAG_BAD_SCALE, FLAG_SLOTS = 1, 2, 4, 8, 16, 32
W_INT = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int64)   # contract_v2.W_INT


# ----------------------------------------------------------------------------- geometry (as the .cu)

def cdiv(a, b):
    return -(-a // b)


def chunk_bytes(bpc=BPC_DEFAULT):
    return bpc * 16


def exp_words(bpc=BPC_DEFAULT):
    return bpc // 4


def stage_bytes(MT, bpc=BPC_DEFAULT):
    """sizeof(Stage<MT>): W[32][bpc*16] + C[MT][bpc*32] + EA[MT][16] + EW[32][16]."""
    return BN * chunk_bytes(bpc) + MT * bpc * BLOCK_K + MT * EXP_ROW_BYTES + BN * EXP_ROW_BYTES


def x_bytes(MT, bpc=BPC_DEFAULT):
    return 4 * MT * bpc * BLOCK_K


def smem_bytes(MT, bpc=BPC_DEFAULT, stages=3, lut=True):
    return stage_bytes(MT, bpc) * stages + x_bytes(MT, bpc) + (16 * 32 * 4 if lut else 0)


def chunks(G, bpc=BPC_DEFAULT):
    """Chunks of the S = 1 walk as (t_first, nb) in issue order."""
    return [(c * bpc, min(bpc, G - c * bpc)) for c in range(cdiv(G, bpc))]


def grid(N, count):
    return (N // BN, 1, count)


# ----------------------------------------------------------------------------- the route table, leader rule and gather

def active_table(ids, weights, E):
    """active = where((ids >= 0) & (ids < E) & (weights != 0), ids, -1), as the DP4A launcher and the entry compute it."""
    ids = np.asarray(ids, np.int64)
    weights = np.asarray(weights, np.float32)
    return np.where((ids >= 0) & (ids < E) & (weights != 0), ids, -1).astype(np.int64)


def cta_plan(active_flat, z, MT):
    """What CTA z does: (eid, leader, slots, cnt_all). slots = the routes sharing eid in ascending order, clamped to MT;
    leader is False when an earlier route has the same eid (the CTA exits); eid < 0 exits before the leader test."""
    a = np.asarray(active_flat, np.int64).ravel()
    eid = int(a[z])
    if eid < 0:
        return eid, False, [], 0
    if np.any(a[:z] == eid):
        return eid, False, [], 0
    routes = [int(t) for t in np.nonzero(a == eid)[0]]
    return eid, True, routes[:MT], len(routes)


def leaders(active_flat, MT):
    """Every leader CTA's plan: {route z: (eid, slots, cnt_all)}."""
    out = {}
    for z in range(len(active_flat)):
        eid, lead, slots, cnt_all = cta_plan(active_flat, z, MT)
        if lead:
            out[z] = (eid, slots, cnt_all)
    return out


# ----------------------------------------------------------------------------- the quantiser model (_quant_codes)

def quant_codes(x):
    """NumPy transliteration of the Triton _quant_codes (the _quant_v2 arithmetic, codes and ea only; a non-finite
    group gets NaN codes). x float32 or BF16-valued [R, K] -> codes uint8 [R, K], ea int8 [R, K/32]."""
    x = np.asarray(x, np.float32)
    R, K = x.shape
    assert K % BLOCK_K == 0
    v = x.reshape(R, K // BLOCK_K, BLOCK_K)
    with np.errstate(all='ignore'):
        nonfinite = ~np.isfinite(v).all(-1)
        am = np.maximum(np.max(np.abs(v), -1), np.float32(1e-4)).astype(np.float32)
        b = (am * np.float32(1. / 448.)).astype(np.float32).view(np.uint32)
        ex = ((b >> np.uint32(23)) & np.uint32(255)).astype(np.int32) + ((b & np.uint32(0x7fffff)) != 0).astype(np.int32) - 127
        scale = (((ex + 127) << 23).astype(np.uint32)).view(np.float32)
        z = (v / scale[..., None]).astype(np.float32)
        a = np.abs(z).astype(np.float32)
        ab = a.view(np.uint32)
        normal = ((ab + np.uint32(0x7ffff) + ((ab >> np.uint32(20)) & np.uint32(1))) >> np.uint32(20)).astype(np.int32) - 960
        sub = (a * np.float32(512.)).astype(np.float32)
        flo = np.floor(sub).astype(np.float32)
        frac = (sub - flo).astype(np.float32)
        rounded = flo.astype(np.int32) + ((frac > np.float32(.5)) | ((frac == np.float32(.5)) & ((flo.astype(np.int32) & 1) != 0))).astype(np.int32)
        code = np.where(a >= np.float32(448.), 126, np.where(a < np.float32(.015625), rounded, normal))
        sign = ((z.view(np.uint32) >> np.uint32(31)) & np.uint32(1)).astype(np.int32)
        code = code | (sign << 7)
        code = np.where(nonfinite[..., None], 127, code)
    return code.astype(np.uint8).reshape(R, K), ex.astype(np.int8)


# ----------------------------------------------------------------------------- the prologue values (as the .cu)

def codes_to_int(codes):
    """The cooperative conversion: n = dense_m_arith(code) (int32), nan flag if any (code & 127) == 127."""
    codes = np.asarray(codes, np.uint8)
    return dense_m_arith(codes).astype(np.int64), bool(((codes & 127) == 127).any())


def row_EA(ea):
    """EA[m] = max_g ea[m, g], as the warp reduction computes it."""
    return np.asarray(ea, np.int64).max(axis=1)


def row_EW(ws_rows):
    """EW[n] = max_g ws[n, g] - 127 and the admission check (255 or below MIN_SCALE_CODE) over the CTA's rows."""
    ws_rows = np.asarray(ws_rows, np.uint8)
    bad = bool(((ws_rows == 255) | (ws_rows < MIN_SCALE_CODE)).any())
    return ws_rows.astype(np.int64).max(axis=1) - 127, bad


def expert_ints(w_nibbles):
    """E2M1 nibbles [N, K/2] -> w' int64 [N, K] (W_INT, the LUT form; the prmt form equals it on every window)."""
    w = np.asarray(w_nibbles, np.uint8)
    codes = np.stack((w & 15, w >> 4), -1).reshape(w.shape[0], -1)
    return W_INT[codes]


# ----------------------------------------------------------------------------- one CTA (32 rows x all K) for a leader

def cta_rows(n, ea_s, EA, m, ew, EW, bpc=BPC_DEFAULT):
    """The kernel's accumulation for cnt slots x 32 rows: n int [cnt, K] (converted codes), ea_s int [cnt, G], EA
    [cnt]; m int [32, K], ew int [32, G], EW [32]. Returns (A int64 [cnt, 32], overflow, meta_bad). Chunk by chunk,
    group by group in ascending order; lane partials over 16 elements in int32, shuffle-combined into D_g."""
    n = np.asarray(n, np.int64)
    m = np.asarray(m, np.int64)
    cnt, K = n.shape
    G = K // BLOCK_K
    a = n.reshape(cnt, G, LANES_PER_GROUP, PER_LANE)
    b = m.reshape(BN, G, LANES_PER_GROUP, PER_LANE)
    assert int(np.abs(a).max(initial=0)) < 2 ** 18 and int(np.abs(b).max()) <= 12
    lane = np.einsum('mghj,nghj->mngh', a.astype(np.int32), b.astype(np.int32), dtype=np.int32)   # IMAD chain
    assert int(np.abs(lane).max(initial=0)) < 2 ** 26
    D = lane[..., 0].astype(np.int64) + lane[..., 1].astype(np.int64)                            # shfl.xor combine
    EA = np.asarray(EA, np.int64)
    EW = np.asarray(EW, np.int64)
    ea_s = np.asarray(ea_s, np.int64)
    ew = np.asarray(ew, np.int64)
    cm = MARGIN_EXPERT - EA[:, None] - EW[None, :]                                                  # [cnt, 32]
    acc = np.zeros((cnt, BN), np.int64)
    ovf = np.zeros((cnt, BN), np.int64)
    meta_bad = False
    for t_first, nb in chunks(G, bpc):
        # staged windows of this chunk: ea_s[:, t_first:t_first+nb], ew[:, t_first:t_first+nb]
        for bidx in range(nb):
            g = t_first + bidx
            ew_g = ew[:, g]
            meta_bad |= bool((ew_g > EW).any())
            ea_g = ea_s[:, g]
            meta_bad |= bool((ea_g > EA).any())
            sh = ea_g[:, None] + ew_g[None, :] + cm
            acc, o = add_ovf(acc, shift_term(D[:, :, g], sh))
            ovf |= o
    return acc, bool((ovf < 0).any()), meta_bad


# ----------------------------------------------------------------------------- one launch (grid (N/32, 1, count))

def static_projection(codes, ea, w_all, ws_all, active_flat, first, topk, MT, bpc=BPC_DEFAULT, Y=None, A=None,
                      EA_override=None, EW_override=None):
    """One static launch. codes u8 [R, K], ea i8 [R, G] (R = M rows for gate/up, count for down); w_all u8 [E, N, K/2],
    ws_all u8 [E, N, G]; active_flat int [count]; MT = M (the template). Returns (Y, A, flags): Y uint16 BF16 bits
    [count, N] (the gate/up launch writes bits << 16 as float32; the model keeps the bits and widens in the layer
    model), A int64 [count, N] raw row sums; rows of inactive and non-leader-served routes untouched (Y / A may be
    passed in). EA_override / EW_override [count] / [E, N] model a stale table for the L5 defence gate only."""
    codes = np.asarray(codes, np.uint8)
    ea = np.asarray(ea, np.int8)
    w_all = np.asarray(w_all, np.uint8)
    ws_all = np.asarray(ws_all, np.uint8)
    active_flat = np.asarray(active_flat, np.int64).ravel()
    count = active_flat.size
    E, N, half_k = w_all.shape
    K = 2 * half_k
    G = K // BLOCK_K
    assert codes.shape[1] == K and ea.shape == (codes.shape[0], G) and ws_all.shape == (E, N, G)
    assert N % BN == 0 and K % (BLOCK_K * 4) == 0 and 1 <= MT <= MAX_M and count <= MAX_COUNT
    if Y is None:
        Y = np.zeros((count, N), np.uint16)
    if A is None:
        A = np.zeros((count, N), np.int64)
    flags = 0
    n_all, nan = codes_to_int(codes)                           # the conversion is per staged chunk; same values
    for z, (eid, slots, cnt_all) in leaders(active_flat, MT).items():
        if cnt_all > MT:
            flags |= FLAG_SLOTS
        tok = [s // topk if first else s for s in slots]
        n = n_all[tok]
        ea_s = ea[tok].astype(np.int64)
        EA = row_EA(ea_s) if EA_override is None else np.asarray(EA_override, np.int64)[slots]
        if bool(((codes[tok] & 127) == 127).any()):
            flags |= FLAG_NAN_CODE
        m = expert_ints(w_all[eid])
        ew = ws_all[eid].astype(np.int64) - 127
        for row0 in range(0, N, BN):
            EW, bad = row_EW(ws_all[eid][row0:row0 + BN])
            if EW_override is not None:
                EW = np.asarray(EW_override, np.int64)[eid, row0:row0 + BN]
            if bad:
                flags |= FLAG_BAD_SCALE
            acc, ovf, meta = cta_rows(n, ea_s, EA, m[row0:row0 + BN], ew[row0:row0 + BN], EW, bpc)
            if ovf:
                flags |= FLAG_OVERFLOW
            if meta:
                flags |= FLAG_META
            B = EA[:, None] + EW[None, :] - MARGIN_EXPERT
            bits, bad_f = finish_bits(acc, B - SA - SW_EXPERT)
            if bool(bad_f.any()):
                flags |= FLAG_FINISH
            Y[np.asarray(slots), row0:row0 + BN] = bits
            A[np.asarray(slots), row0:row0 + BN] = acc
    return Y, A, flags


def bits_to_f32(bits):
    """The gate/up output as the kernel writes it: BF16 bits << 16 viewed as float32 (exact widening)."""
    return (np.asarray(bits, np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def activate_routes(gu, active_flat, weights_flat, SILU):
    """_activate_routes (dp4a_routes_candidate, unchanged): h[route] = bf(SILU[gate bits >> 16] * clip(up, -10, 10)
    * weight); zeros for eid < 0. gu float32 [count, 2I] -> h float32 [count, I]."""
    gu = np.asarray(gu, np.float32)
    count, two_i = gu.shape
    I = two_i // 2
    h = np.zeros((count, I), np.float32)
    weights_flat = np.asarray(weights_flat, np.float32).ravel()
    for route in range(count):
        if active_flat[route] >= 0:
            g = gu[route, :I]
            u = gu[route, I:]
            key = g.view(np.uint32) >> np.uint32(16)
            si = SILU[key]
            h[route] = r2m_bf((si * np.clip(u, np.float32(-10.), np.float32(10.))).astype(np.float32) * np.float32(weights_flat[route]))
    return h


def r2m_bf(x):
    x = np.asarray(x, np.float32)
    b = x.view(np.uint32)
    return ((b + np.uint32(0x7fff) + ((b >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)


def expert_routes(x, w13, s13, w2, s2, ids, weights, SILU, out=None, MT=None, bpc=BPC_DEFAULT, stages=False):
    """The whole per-layer entry on the model: quant -> gate/up launch -> activate -> quant -> down launch.
    x float32 (BF16-valued) [M, H]; returns out as BF16 bits uint16 [M, topk, H] (inactive slots untouched) and the
    flag word; with stages=True also the intermediates."""
    x = np.asarray(x, np.float32)
    M = x.shape[0]
    H = w2.shape[1]                                            # the down projection's N (= x's width in the model)
    ids = np.asarray(ids, np.int64)
    topk = ids.shape[1]
    E = w13.shape[0]
    MT = M if MT is None else MT
    count = M * topk
    active = active_table(ids, weights, E).ravel()
    codes_x, ea_x = quant_codes(x)
    gu_bits, A13, f1 = static_projection(codes_x, ea_x, w13, s13, active, True, topk, MT, bpc)
    gu = bits_to_f32(gu_bits)
    h = activate_routes(gu, active, weights, SILU)
    codes_h, ea_h = quant_codes(h)
    if out is None:
        out = np.zeros((count, H), np.uint16)
    else:
        out = np.asarray(out, np.uint16).reshape(count, H)
    out, A2, f2 = static_projection(codes_h, ea_h, w2, s2, active, False, topk, MT, bpc, Y=out)
    flags = f1 | f2
    result = out.reshape(M, topk, H)
    if stages:
        return result, flags, dict(active=active, codes_x=codes_x, ea_x=ea_x, gu=gu, A13=A13, h=h, codes_h=codes_h, ea_h=ea_h, A2=A2)
    return result, flags
