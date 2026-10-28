"""Route 3 reference: the per-row sequence of the dense FP8 projections, and the mma.m16n8k32 8-bit fragment map.

Runs anywhere with NumPy; no GPU. Two things are reproduced here so that Codex's component harness has an
independent oracle for everything *around* the tensor-core instruction:

  1. The FP32 epilogue and split-K structure of the shared sequence (ROUTE3_MMA_DESIGN.md section 1): given the
     per-block MMA results D_t (hardware values captured on the GPU), compute the output exactly as the kernels
     must, for either contraction form, with per-split partials in the kernel's [4, M, N] layout.
  2. The fragment layout of mma.sync.m16n8k32 for 8-bit operands (A row-major 16x32, B col-major 32x8, C/D 16x8
     f32), as the PTX ISA figures for the .s8/.u8/.e4m3 k32 shape give it: lane = 4*groupID + tig.
     Verify against the ISA before relying on it in a CUDA kernel; the self-test checks coverage only.

The MMA itself (Ada QMMA: two fused 16-product blocks, 13 fractional bits, truncation) is NOT modelled here;
route 3 never needs its bits in software, only that both kernels issue it identically.
"""
import numpy as np

BLOCK_K = 32
SPLIT_K = 4
F32 = np.float32


# ---------------------------------------------------------------- 1. the shared sequence around the MMA

def split_ranges(K, split_k=SPLIT_K, block_k=BLOCK_K):
    """Block index ranges [lo, hi) per split, exactly as the kernel: tiles_per_split = cdiv(cdiv(K, 32), 4)."""
    T = -(-K // block_k)
    per = -(-T // split_k)
    return [(s * per, min((s + 1) * per, T)) for s in range(split_k)]


def fma_rn(a, b, c):
    """Correctly rounded fused a*b+c in float32 (single rounding), via float64 which holds the exact product
    and the sum exactly enough for one RNE to float32 here (|a|,|b|,|c| are ordinary float32 values)."""
    return F32(np.float64(a) * np.float64(b) + np.float64(c))


def epilogue_partials(D, As, Bs, contraction):
    """D: [M, N, T] float32 per-block MMA results (C = 0); As: [M, T] float32; Bs: [N//32, T] float32.
    contraction: 'mul_mul_add' = mul.rn, mul.rn, add.rn  |  'mul_fma' = t = mul.rn(D, As); acc = fma.rn(t, Bs, acc).
    Returns partials [SPLIT_K, M, N] float32, each split accumulated in ascending block order from 0.0."""
    M, N, T = D.shape
    out = np.zeros((SPLIT_K, M, N), dtype=F32)
    for s, (lo, hi) in enumerate(split_ranges(T * BLOCK_K)):
        acc = np.zeros((M, N), dtype=F32)
        for t in range(lo, hi):
            a_s = As[:, t].astype(F32)[:, None]
            b_s = Bs[:, t].astype(F32)[np.arange(N) // 32][None, :]
            d = D[:, :, t].astype(F32)
            if contraction == 'mul_mul_add':
                acc = (acc + (d * a_s) * b_s).astype(F32)          # NumPy float32 ops round after each op (no FMA)
            elif contraction == 'mul_fma':
                t1 = (d * a_s).astype(F32)
                acc = np.vectorize(fma_rn, otypes=[F32])(t1, b_s, acc)
            else:
                raise ValueError(contraction)
        out[s] = acc
    return out


def reduce_sequential(partials):
    """((p0 + p1) + p2) + p3 in float32. The kernel's tl.sum order is not asserted here: the design reuses the
    kernel's own reduce so its order never has to be known. This function exists only for local experiments."""
    acc = partials[0].astype(F32)
    for s in range(1, partials.shape[0]):
        acc = (acc + partials[s]).astype(F32)
    return acc


# ---------------------------------------------------------------- 2. mma.m16n8k32 fragment map (8-bit operands)

def a_fragment(lane, i):
    """A (16x32 row-major): element index i in 0..15 held by this lane as 4 x b32 (i//4 = register). Returns (row, col)."""
    g, tig = lane // 4, lane % 4
    row = g + (8 if (i // 4) % 2 == 1 else 0)          # a0..3 row g; a4..7 row g+8; a8..11 row g; a12..15 row g+8
    col = (tig * 4) + (i % 4) + (16 if i >= 8 else 0)  # a0..7 cols 0..15 block; a8..15 cols 16..31
    return row, col


def b_fragment(lane, i):
    """B (32x8 col-major): element i in 0..7 held by this lane as 2 x b32. Returns (k, n)."""
    g, tig = lane // 4, lane % 4
    k = (tig * 4) + (i % 4) + (16 if i >= 4 else 0)
    return k, g


def c_fragment(lane, i):
    """C/D (16x8 f32): element i in 0..3. Returns (row, col)."""
    g, tig = lane // 4, lane % 4
    row = g + (8 if i >= 2 else 0)
    col = tig * 2 + (i % 2)
    return row, col


def _coverage(frag, count, shape):
    seen = np.zeros(shape, dtype=np.int32)
    for lane in range(32):
        for i in range(count):
            r, c = frag(lane, i)
            assert 0 <= r < shape[0] and 0 <= c < shape[1], (frag.__name__, lane, i, r, c)
            seen[r, c] += 1
    return seen


def self_test():
    rng = np.random.default_rng(3)
    # fragment coverage: every element exactly once
    for frag, count, shape in ((a_fragment, 16, (16, 32)), (b_fragment, 8, (32, 8)), (c_fragment, 4, (16, 8))):
        seen = _coverage(frag, count, shape)
        assert (seen == 1).all(), (frag.__name__, np.argwhere(seen != 1)[:5])
    # a lane's 4 A elements of one register are 4 consecutive columns of one row (one b32 load)
    for lane in range(32):
        for reg in range(4):
            rows_cols = [a_fragment(lane, reg * 4 + j) for j in range(4)]
            assert len({r for r, _ in rows_cols}) == 1 and [c for _, c in rows_cols] == list(range(rows_cols[0][1], rows_cols[0][1] + 4))
        for reg in range(2):
            ks = [b_fragment(lane, reg * 4 + j) for j in range(4)]
            assert len({n for _, n in ks}) == 1 and [k for k, _ in ks] == list(range(ks[0][0], ks[0][0] + 4))
    print('1. m16n8k32 8-bit fragment map: A/B/C cover every element once; each register is 4 consecutive K bytes of one row/column')
    # split ranges for the checkpoint K values
    for K, expect in ((5120, [(0, 40), (40, 80), (80, 120), (120, 160)]), (2304, [(0, 18), (18, 36), (36, 54), (54, 72)]),
                      (8192, [(0, 64), (64, 128), (128, 192), (192, 256)]), (1280, [(0, 10), (10, 20), (20, 30), (30, 40)])):
        assert split_ranges(K) == expect, (K, split_ranges(K))
    print('2. split-K ranges fixed by K alone: 5120->40 blocks/split, 2304->18, 8192->64, 1280->10')
    # the sequence is per row: a row's partials do not depend on which other rows are in the batch
    M, N, T = 6, 64, 40
    D = rng.standard_normal((M, N, T)).astype(F32) * F32(50.0)
    As = np.exp2(rng.integers(-12, -2, size=(M, T))).astype(F32)
    Bs = np.exp2(rng.integers(-9, -3, size=(N // 32, T))).astype(F32)
    full = epilogue_partials(D, As, Bs, 'mul_fma')
    for m in range(M):
        alone = epilogue_partials(D[m:m + 1], As[m:m + 1], Bs, 'mul_fma')
        assert np.array_equal(alone[:, 0, :], full[:, m, :])
    print('3. epilogue partials of a row are identical alone and inside a 6-row batch (per-row sequence)')
    # with power-of-two scales (UE8M0 activation and weight scales) both multiplies are exact while no product
    # underflows (a subnormal result would be rounded, or flushed under .ftz), so the only rounding per block is the
    # add: mul,mul,add and mul,fma give the same bits here. With arbitrary FP32 scales they would not. The kernels
    # retain the exact emitted sequence regardless; this check only explains why the choice is invisible on real data.
    a = epilogue_partials(D, As, Bs, 'mul_mul_add'); b = epilogue_partials(D, As, Bs, 'mul_fma')
    assert np.array_equal(a, b)
    As_f = (As * rng.uniform(1.0, 1.9, size=As.shape)).astype(F32); Bs_f = (Bs * rng.uniform(1.0, 1.9, size=Bs.shape)).astype(F32)
    a2 = epilogue_partials(D, As_f, Bs_f, 'mul_mul_add'); b2 = epilogue_partials(D, As_f, Bs_f, 'mul_fma')
    frac = float(np.mean(a2 != b2))
    assert frac > 0.0
    print('4. power-of-two scales: mul,mul,add == mul,fma bit for bit (exact multiplies, one rounding per block);'
          ' arbitrary FP32 scales would differ on %.1f%% of partials, so the selector must assert power-of-two scales' % (100 * frac))
    # a split boundary moved by one block changes bits: SPLIT_K and BLOCK_K are part of the arithmetic
    def shifted(D, As, Bs):
        Mx, Nx, Tx = D.shape; out = np.zeros((SPLIT_K, Mx, Nx), dtype=F32)
        ranges = split_ranges(Tx * BLOCK_K); ranges = [(0, ranges[0][1] + 1)] + [(ranges[1][0] + 1, ranges[1][1])] + ranges[2:]
        for s, (lo, hi) in enumerate(ranges):
            acc = np.zeros((Mx, Nx), dtype=F32)
            for t in range(lo, hi):
                acc = np.vectorize(fma_rn, otypes=[F32])((D[:, :, t] * As[:, t][:, None]).astype(F32), Bs[:, t][np.arange(Nx) // 32][None, :], acc)
            out[s] = acc
        return out
    assert not np.array_equal(reduce_sequential(shifted(D, As, Bs)), reduce_sequential(b))
    print('5. moving one block across a split boundary changes the reduced bits: the partition is frozen with the selector')
    print('SELFTEST_PASS')


if __name__ == '__main__':
    self_test()
