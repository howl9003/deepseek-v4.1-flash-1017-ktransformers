import numpy as np
import torch
import oracle as O
import rows as E1
import banks as V

H, I, IH, GH = 5120, 2304, 1152, 36

def gemm_chain(x, codes, sc, qmax):
    q, ea = O.quantise(x, qmax)
    J = O.E2M1_J[codes.astype(np.int64)].astype(np.float64)
    ew = sc.astype(np.int64) - 127
    qd = q.astype(np.float64)
    acc = np.zeros((x.shape[0], codes.shape[0]), np.float32)
    for g in range(x.shape[1] // 32):
        dot = qd[:, 32 * g:32 * g + 32] @ J[:, 32 * g:32 * g + 32].T
        acc = O.fmaf(dot.astype(np.float32), O.exp2_f32(ea[:, g:g + 1].astype(np.int64) + ew[None, :, g] - 1), acc)
    return acc


def down_raw(xd, codes, sc, qmax, bL):
    q, ea = O.quantise(xd, qmax)
    J = O.E2M1_J[codes.astype(np.int64)].astype(np.float64)
    ew = sc.astype(np.int64) - 127
    qd = q.astype(np.float64)
    S = np.zeros((xd.shape[0], codes.shape[0]), np.int64)
    for g in range(xd.shape[1] // 32):
        dot = (qd[:, 32 * g:32 * g + 32] @ J[:, 32 * g:32 * g + 32].T).astype(np.int64)
        sh = ea[:, g:g + 1].astype(np.int64) + ew[None, :, g] - 1 - bL
        assert not np.any(sh + 24 > 62)
        S = S + np.where(sh >= 0, dot << np.clip(sh, 0, 62), dot >> np.clip(-sh, 0, 63))
        assert not np.any(np.abs(S) >= (1 << 62))
    return S


def half_raw(x, ex, w_route, half, qmax, bL, silu):
    """Half raw."""
    cols = slice(0, I) if half is None else slice(IH * half, IH * (half + 1))
    blks = slice(0, I // 32) if half is None else slice(GH * half, GH * (half + 1))
    gate = O.bf16_rne(gemm_chain(x, ex["w1c"][cols], ex["w1s"][cols], qmax))
    up = O.bf16_rne(gemm_chain(x, ex["w3c"][cols], ex["w3s"][cols], qmax))
    xd = O.swiglu(gate, up, w_route, silu)
    return down_raw(xd, ex["w2c"][:, cols], ex["w2s"][:, blks], qmax, bL)


def load_expert(layer, e):
    ex = {}
    for proj, tag in (("w1", "w1"), ("w3", "w3"), ("w2", "w2")):
        packed, sc = E1.gguf_expert(layer, proj, e)
        ex[tag + "p"], ex[tag + "s"] = packed, sc
        ex[tag + "c"] = O.unpack_fp4(packed)
    return ex


def bank(experts, half):
    """Bank."""
    c, b = slice(IH * half, IH * (half + 1)), slice(GH * half, GH * (half + 1))
    w13 = np.stack([np.concatenate((ex["w1p"][c], ex["w3p"][c])) for ex in experts])
    s13 = np.stack([np.concatenate((ex["w1s"][c], ex["w3s"][c])) for ex in experts])
    w2 = np.stack([np.ascontiguousarray(ex["w2p"][:, IH // 2 * half:IH // 2 * (half + 1)]) for ex in experts])
    s2 = np.stack([np.ascontiguousarray(ex["w2s"][:, b]) for ex in experts])
    W13r, S13r = V.repack(E1.to_t(w13), E1.to_t(s13))
    W2r, S2r = V.repack(E1.to_t(w2), E1.to_t(s2))
    return W13r, S13r, W2r, S2r


def oracle_token_raw(x, ids, wts, experts, half, qmax, bL, silu):
    """Oracle token raw."""
    T = x.shape[0]
    S = np.zeros((T, H), np.int64)
    for e in range(len(experts)):
        rows, ks = np.nonzero(ids == e)
        if rows.size == 0:
            continue
        part = half_raw(x[rows], experts[e], wts[rows, ks][:, None], half, qmax, bL, silu)
        np.add.at(S, rows, part)
    return S
