"""Contract v2: integer-exact arithmetic for DeepSeek V4.1-1017 routed experts (NumPy reference).

Quantization grid is the reference's: E4M3 activations in groups of 32 with power-of-two
(E8M0) scales, E2M1 weights in groups of 32 with E8M0 scales, BF16 after each projection,
FP32 SwiGLU with the gate <= 10 and up in [-10, 10] clamps, routing weight before the BF16 cast
and the down quantization. What differs from the FP32 fixed-tree prototype is how the dots are
evaluated:

  * an E4M3 value is an integer n (< 2^18) times 2^-9, an E2M1 value an integer w' (|w'| <= 12)
    times 2^-1, so a 32-wide group dot D = sum(sign*n*w') is an integer below 2^27: exact in
    int32 in any order on any hardware;
  * with power-of-two scales a whole K-row accumulates exactly in int64:
        value = sum_g D_g * 2^(ea_g + ew_g - 10)
        B     = EA + EW - MARGIN,  EA = max_g ea_g (per token), EW = max_g ew_g (per output row)
        A     = sum_g D_g << (ea_g + ew_g - B)      (terms below B: floor shift, deterministic)
        y     = bf16( fp32_rne(A) * 2^(B - 10) )
    |A| < K/32 * 2^27 * 2^MARGIN; MARGIN = 20 keeps K <= 8192 inside int64 and, per 8-element lane
    (|lane| < 2^25), inside the 53-bit exact range of FP64, so the CPU may accumulate in FP64 FMAs. Observed scale
    spreads are 4 (gate/up) and 8 (down), so no term is ever truncated on real data;
  * the cross-group order, SIMD width, tensor-core type, row or K partition and NUMA layout
    therefore never change a bit; the only roundings are fp32_rne(A), the BF16 outputs, the
    SiLU table and the combine's FP32 sum in ascending expert-id order with the shared expert
    added before the single BF16 cast (reference MoE.forward).
"""
import hashlib
from pathlib import Path
import numpy as np

MARGIN = 20
HERE = Path(__file__).resolve().parent

def _frozen(name, count, expected):
    data = (HERE / name).read_bytes()
    if len(data) != count * 4 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('shared arithmetic table has changed: ' + name)
    return np.frombuffer(data, dtype='<f4')

FP8 = _frozen('fp8_table.bin', 256, '422eccfaa21e72a6b26855bb10cdcfead6c1ce3262ecd813c99d8cbf9677f2e2')
SILU = _frozen('silu_table.bin', 65536, '4e08cff10bed7793df85e094f8ec9d9700cce52412bd5e54070eaf221596f9ea')
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
W_INT = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int64)   # FP4 * 2
_c = np.arange(256); _e = (_c & 127) >> 3; _m = _c & 7
N_OF_CODE = np.where(_e == 0, _m, (8 + _m) << np.maximum(_e - 1, 0)).astype(np.int64); N_OF_CODE[[127, 255]] = 0
SIGN_OF_CODE = np.where(_c & 128, -1, 1).astype(np.int64)
assert np.all(np.abs(FP8[np.r_[0:127, 128:255]]) == (N_OF_CODE * 2.0 ** -9)[np.r_[0:127, 128:255]])

def bf(x):
    x = np.asarray(x, dtype=np.float32); b = x.view(np.uint32)
    return ((b + np.uint32(0x7fff) + ((b >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)

def quant(x):
    """BF16-valued FP32 (m, k) -> codes u8 (m,k), ea int8 (m,k/32), hi/lo int16 digits (m,k), EA int8 (m,)."""
    x = np.asarray(x, dtype=np.float32); assert x.ndim == 2 and x.shape[1] % 32 == 0
    if not np.isfinite(x).all(): raise ValueError('finite activations required')
    v = x.reshape(x.shape[0], -1, 32); am = np.maximum(np.max(np.abs(v), -1), np.float32(1e-4))
    t = (am * np.float32(1 / 448)).view(np.uint32)
    ea = (((t >> 23) & 255).astype(np.int64) - 127 + ((t & 0x7fffff) != 0)).astype(np.int8)
    s = np.ldexp(np.float32(1), ea.astype(np.int32)).astype(np.float32)
    z = np.abs(v / s[..., None])
    hi = np.minimum(np.searchsorted(FP8[:127], z), 126); lo = np.maximum(hi - 1, 0)
    dl = z - FP8[lo]; dh = FP8[hi] - z
    c = np.where((dl < dh) | ((dl == dh) & ((lo & 1) == 0)), lo, hi).astype(np.uint8)
    c |= (np.signbit(v).astype(np.uint8) << 7)
    c = c.reshape(x.shape)
    n = N_OF_CODE[c] * SIGN_OF_CODE[c]
    sign = np.sign(n); mag = np.abs(n)
    return c, ea, (sign * (mag >> 9)).astype(np.int16), (sign * (mag & 511)).astype(np.int16), ea.max(axis=1).astype(np.int8)

def nibbles(w):
    """(n, k/2) adjacent-nibble layout -> (n, k/32, 32) codes."""
    w = np.asarray(w, np.uint8); return np.stack((w & 15, w >> 4), -1).reshape(w.shape[0], -1, 32)

def prepare_weights(ws):
    ws = np.asarray(ws, np.uint8)
    if np.any(ws == 255): raise ValueError('E8M0 NaN scale')
    return (ws.max(axis=1).astype(np.int64) - 127).astype(np.int8)   # EW per output row

def group_dots(hi, lo, w, g0=0, g1=None):
    """Exact int64 group dots D (m, n, G) for groups [g0, g1)."""
    m, k = hi.shape; n = w.shape[0]; G = k // 32; g1 = G if g1 is None else g1
    a = (hi.astype(np.int64) << 9) + lo.astype(np.int64)
    a = a.reshape(m, G, 32)[:, g0:g1]
    out = np.empty((m, n, g1 - g0), np.int64)
    wc = nibbles(w)[:, g0:g1]
    for r0 in range(0, n, 256):
        wi = W_INT[wc[r0:r0 + 256]]                          # (nb, G', 32)
        out[:, r0:r0 + 256] = (a[:, None, :, :] * wi[None]).sum(-1)
    return out

def accumulate(D, ea, ws, EA, EW, g0=0, g1=None):
    """Exact int64 accumulation of groups [g0, g1) with the row base B = EA + EW - MARGIN."""
    m, n, Gp = D.shape; g1 = g0 + Gp
    e = ea[:, None, g0:g1].astype(np.int64) + (ws[None, :, g0:g1].astype(np.int64) - 127)   # (m, n, G')
    B = EA[:, None].astype(np.int64) + EW[None, :].astype(np.int64) - MARGIN                  # (m, n)
    sh = e - B[..., None]
    pos = np.where(sh >= 0, sh, 0); neg = np.where(sh < 0, -sh, 0)
    return ((D << pos) >> neg).sum(-1)                                                         # floor for the truncated terms

def finish(A, EA, EW):
    B = EA[:, None].astype(np.int32) + EW[None, :].astype(np.int32) - MARGIN
    f = A.astype(np.float32)                                     # one RNE
    return bf(np.ldexp(f, B - 10).astype(np.float32))

def linear(hi, lo, ea, EA, w, ws, EW):
    D = group_dots(hi, lo, w)
    return finish(accumulate(D, ea, ws, EA, EW), EA, EW)

def activate(gu, weight):
    g, u = np.split(np.asarray(gu, np.float32), 2, -1); key = g.view(np.uint32) >> 16
    return bf((SILU[key] * np.clip(u, -10, 10)) * np.asarray(weight, np.float32)[:, None])

def expert(x, w13, s13, w2, s2, weight, stages=False):
    c, ea, hi, lo, EA = quant(x); gu = linear(hi, lo, ea, EA, w13, s13, prepare_weights(s13))
    h = activate(gu, weight); hc, hea, hhi, hlo, hEA = quant(h)
    out = linear(hhi, hlo, hea, hEA, w2, s2, prepare_weights(s2))
    return (out, dict(input_codes=c, gate_up=gu, activation=h, down_codes=hc)) if stages else out

def combine(experts, expert_ids, shared=None):
    """experts (m, topk, hidden) BF16-valued, expert_ids (m, topk): FP32 sum in ascending expert id,
    shared expert (m, hidden) added before the single BF16 cast (reference MoE.forward)."""
    experts = np.asarray(experts, np.float32); m, topk, hidden = experts.shape
    order = np.argsort(np.asarray(expert_ids), axis=1, kind='stable')
    acc = np.zeros((m, hidden), np.float32)
    for j in range(topk):
        acc = (acc + experts[np.arange(m), order[:, j]]).astype(np.float32)
    if shared is not None: acc = (acc + np.asarray(shared, np.float32)).astype(np.float32)
    return bf(acc)
