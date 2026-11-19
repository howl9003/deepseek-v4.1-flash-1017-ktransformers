import numpy as np

E2M1_J = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int32)  # 2 * E2M1 value


# ---------- FP32 / BF16 primitives ----------

def bf16_rne_bits(x):
    """FP32 -> BF16 bits, RNE (no NaN inputs expected)."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return b.astype(np.uint16)


def bf16_bits_to_f32(b):
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_rne(x):
    return bf16_bits_to_f32(bf16_rne_bits(x))


def exp2_f32(n):
    """Exact 2^n as FP32 for integer n (array); subnormal for -149..-127, 0 below -149. Raises above 127."""
    n = np.asarray(n, np.int64)
    if np.any(n > 127):
        raise OverflowError("scale exponent above 127")
    out = np.zeros(n.shape, np.float32)
    norm = n >= -126
    out[norm] = ((n[norm] + 127).astype(np.uint32) << 23).view(np.float32)
    sub = (n < -126) & (n >= -149)
    out[sub] = (np.uint32(1) << (n[sub] + 149).astype(np.uint32)).view(np.float32)
    return out


def fmaf(a, b, c):
    """Vectorised IEEE FP32 fma(a, b, c) with one RNE rounding, for FP32 arrays where a*b is exact in FP64
    (true here: a is an integer below 2^24, b a power of two). TwoSum in FP64, then a correct final rounding."""
    a = np.asarray(a, np.float32).astype(np.float64)
    b = np.asarray(b, np.float32).astype(np.float64)
    c = np.asarray(c, np.float32).astype(np.float64)
    p = a * b                                    # exact (24-bit integer times a power of two, no FP64 underflow)
    s = p + c
    bp = s - c
    err = (p - bp) + (c - (s - bp))              # TwoSum: p + c == s + err exactly
    f = s.astype(np.float32)                     # RNE of s
    # double rounding is only wrong when s is exactly a midpoint between two FP32 values and err != 0
    fd = f.astype(np.float64)
    up = np.nextafter(f, np.float32(np.inf))
    dn = np.nextafter(f, np.float32(-np.inf))
    other = np.where(s > fd, up, dn)
    mid = (s != fd) & ((s - fd) == (other.astype(np.float64) - s)) & (err != 0)
    toward_other = np.where(s > fd, err > 0, err < 0)
    out = np.where(mid & toward_other, other, f).astype(np.float32)
    # exact cancellation: IEEE gives +0 in RNE for x + (-x); s == 0 already yields +0 unless both terms are -0
    return out


def f32_rne_int64(S):
    """Exact int64 -> FP32 with one RNE rounding (pin (a)), vectorised."""
    S = np.asarray(S, np.int64)
    neg = S < 0
    mag = np.where(neg, -S, S).astype(np.uint64)   # |S| < 2^62 guaranteed by the overflow flag
    out = np.empty(S.shape, np.float32)
    flat_m, flat_o = mag.ravel(), out.ravel()
    for i, v in enumerate(flat_m.tolist()):
        L = v.bit_length()
        if L <= 24:
            flat_o[i] = np.float32(v)
            continue
        sh = L - 24
        m, rem, half = v >> sh, v & ((1 << sh) - 1), 1 << (sh - 1)
        if rem > half or (rem == half and (m & 1)):
            m += 1
        flat_o[i] = np.float32(np.ldexp(np.float64(m), sh))
    return np.where(neg, -out, out).astype(np.float32)


# ---------- contract pieces ----------

def quantise(x, qmax):
    """x: FP32 [M, K] (K % 32 == 0). Returns q int32 [M, K], ea int32 [M, K/32]."""
    x = np.ascontiguousarray(x, np.float32)
    M, K = x.shape
    X = x.reshape(M, K // 32, 32)
    amax = np.maximum(np.abs(X).max(axis=2), np.float32(1e-30)).astype(np.float32)
    r = (amax / np.float32(qmax)).astype(np.float32).view(np.uint32)
    e = ((r >> 23) & 0xFF).astype(np.int64) - 127 + ((r & 0x7FFFFF) != 0)
    inv = exp2_f32(-e)
    q = np.clip(np.rint((X * inv[:, :, None]).astype(np.float32)), -qmax, qmax).astype(np.int32)
    return q.reshape(M, K), e.astype(np.int32)


def unpack_fp4(packed):
    """uint8 [N, K/2] -> nibble codes uint8 [N, K]; low nibble = even k (confirm against the loader)."""
    p = np.asarray(packed, np.uint8)
    out = np.empty((p.shape[0], p.shape[1] * 2), np.uint8)
    out[:, 0::2] = p & 0xF
    out[:, 1::2] = p >> 4
    return out


def gemm(x, wcodes, wscale_e8m0, qmax):
    """x FP32 [M, K]; wcodes uint8 nibbles [N, K]; wscale_e8m0 uint8 [N, K/32]. Returns FP32 acc [M, N] (pre-BF16)."""
    q, ea = quantise(x, qmax)
    J = E2M1_J[np.asarray(wcodes, np.int64)]
    ew = np.asarray(wscale_e8m0, np.int64) - 127
    M, K = q.shape
    N = J.shape[0]
    G = K // 32
    acc = np.zeros((M, N), np.float32)
    for g in range(G):
        dot = q[:, 32 * g:32 * g + 32].astype(np.int64) @ J[:, 32 * g:32 * g + 32].T   # exact
        assert np.abs(dot).max(initial=0) < (1 << 24)
        s = exp2_f32(ea[:, g:g + 1].astype(np.int64) + ew[None, :, g] - 1)
        acc = fmaf(dot.astype(np.float32), s, acc)
    return acc


def swiglu(gate_bf, up_bf, w_route, silu_bits):
    """gate_bf, up_bf: FP32 arrays holding BF16 values [M, I]; w_route FP32 [M, 1]; silu_bits int32/uint32 [65536]."""
    table = np.asarray(silu_bits).astype(np.uint32).view(np.float32)
    u = np.clip(up_bf, np.float32(-10), np.float32(10)).astype(np.float32)
    g = np.minimum(gate_bf, np.float32(10)).astype(np.float32)
    sg = table[bf16_rne_bits(g)]                 # g is already a BF16 value, so this is its exact bit pattern
    h = (sg * u).astype(np.float32)
    h = (h * np.asarray(w_route, np.float32)).astype(np.float32)
    return bf16_rne(h)


def gemm_int64_partial(x, wcodes, wscale_e8m0, qmax, bL, blocks=None):
    """Spec r2 down projection: exact int64 partial S over the given 32-blocks (default: all). The quantiser is per
    block, so a part holding only some blocks of x quantises exactly those blocks the same way."""
    q, ea = quantise(x, qmax)
    J = E2M1_J[np.asarray(wcodes, np.int64)]
    ew = np.asarray(wscale_e8m0, np.int64) - 127
    M, K = q.shape
    N = J.shape[0]
    G = K // 32
    S = np.zeros((M, N), np.int64)
    for g in (range(G) if blocks is None else blocks):
        dot = q[:, 32 * g:32 * g + 32].astype(np.int64) @ J[:, 32 * g:32 * g + 32].T   # exact, |dot| < 2^24
        assert np.abs(dot).max(initial=0) < (1 << 24)
        sh = ea[:, g:g + 1].astype(np.int64) + ew[None, :, g] - 1 - bL
        if np.any(sh + 24 > 62):
            raise OverflowError("down term exceeds the int64 grid")
        left = dot << np.clip(sh, 0, 62)
        right = dot >> np.clip(-sh, 0, 63)          # numpy >> on int64 is arithmetic (floor)
        S = S + np.where(sh >= 0, left, right)
        if np.any(np.abs(S) >= (1 << 62)):
            raise OverflowError("down accumulator overflow flag")
    return S


def int64_finish(S, bL):
    """acc = f32_rne(S) * 2^bL (one IEEE FP32 multiply by a normal power of two), then BF16 RNE."""
    if np.any(np.abs(S) >= (1 << 62)):
        raise OverflowError("down accumulator overflow flag")
    scale = exp2_f32(np.array(bL))
    assert scale > 0 and bL >= -126
    acc = (f32_rne_int64(S) * scale).astype(np.float32)
    return bf16_rne(acc)


def expert_forward(x_bf, w1, w1s, w3, w3s, w2, w2s, w_route, silu_bits, qmax, bL):
    """Spec r2. One routed expert. x_bf FP32 holding BF16 [M, H]. w*: nibble codes [N, K]; w*s: UE8M0 [N, K/32].
    bL: the layer's grid exponent. Returns y_e as FP32 holding BF16 [M, H]."""
    gate = bf16_rne(gemm(x_bf, w1, w1s, qmax))
    up = bf16_rne(gemm(x_bf, w3, w3s, qmax))
    xd = swiglu(gate, up, w_route, silu_bits)
    return int64_finish(gemm_int64_partial(xd, w2, w2s, qmax, bL), bL)


def expert_forward_r1(x_bf, w1, w1s, w3, w3s, w2, w2s, w_route, silu_bits, qmax):
    """Spec r1 (superseded for down): FP32 ascending chain for all three projections."""
    gate = bf16_rne(gemm(x_bf, w1, w1s, qmax))
    up = bf16_rne(gemm(x_bf, w3, w3s, qmax))
    xd = swiglu(gate, up, w_route, silu_bits)
    return bf16_rne(gemm(xd, w2, w2s, qmax))
