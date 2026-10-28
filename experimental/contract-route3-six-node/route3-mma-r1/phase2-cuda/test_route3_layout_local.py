"""CPU-side checks of the phase 2 kernel's data movement (no GPU, no torch): the ldmatrix lane addressing must
deliver exactly the mma.m16n8k32 fragments, the smem swizzle must be bank-conflict free, the cp.async assignment must
cover every byte of a chunk once, the split chunking must match the pinned partition, and the shared-memory budget
must allow two CTAs per SM. Run from this folder: python -I test_route3_layout_local.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from route3_sequence_reference import a_fragment, b_fragment, split_ranges  # noqa: E402

TILE_M, BLOCK_K, SPLIT_K = 16, 32, 4
WARPS, COLS_PER_WARP, BN, BPC, STAGES = 2, 16, 32, 8, 3
THREADS = 32 * WARPS
CHUNK_BYTES = BPC * BLOCK_K
CHUNKS16 = CHUNK_BYTES // 16


def swz(row, ch):
    return ch ^ (row & 7)


def ldmatrix_x4_distribution(row_of_lane, chunk_of_lane):
    """Emulate ldmatrix.x4: lane i supplies the address of row (row_of_lane[i]) at 16-byte chunk (chunk_of_lane[i]);
    tile q = i // 8. Thread t receives, for tile q (register q), bytes 4*(t%4) .. +3 of that tile's row t//4.
    Returns regs[t][q] = list of (row, kbyte) for the 4 bytes."""
    regs = [[None] * 4 for _ in range(32)]
    for q in range(4):
        rows = [row_of_lane[q * 8 + r] for r in range(8)]
        chunks = [chunk_of_lane[q * 8 + r] for r in range(8)]
        for t in range(32):
            r = t // 4
            regs[t][q] = [(rows[r], chunks[r] * 16 + 4 * (t % 4) + j) for j in range(4)]
    return regs


def test_fragments():
    # A operand: lane -> (row, logical chunk within the block pair); block b uses chunks 2b and 2b+1
    a_row = [(lane & 7) + 8 * ((lane >> 3) & 1) for lane in range(32)]
    a_ch = [(lane >> 4) & 1 for lane in range(32)]
    regs = ldmatrix_x4_distribution(a_row, a_ch)
    for t in range(32):
        for q in range(4):
            for j, (row, kbyte) in enumerate(regs[t][q]):
                assert (row, kbyte) == a_fragment(t, q * 4 + j), ("A", t, q, j, (row, kbyte), a_fragment(t, q * 4 + j))
    # B operand: lane -> (column, chunk); registers (q=0,1) -> n-tile 0 b0/b1, (q=2,3) -> n-tile 1 b0/b1
    b_row = [(lane & 7) + 8 * ((lane >> 4) & 1) for lane in range(32)]
    b_ch = [(lane >> 3) & 1 for lane in range(32)]
    regs = ldmatrix_x4_distribution(b_row, b_ch)
    for t in range(32):
        for q in range(4):
            ntile, reg = q // 2, q % 2
            for j, (col, kbyte) in enumerate(regs[t][q]):
                k, n = b_fragment(t, reg * 4 + j)
                assert (col, kbyte) == (n + 8 * ntile, k), ("B", t, q, j, (col, kbyte), (n + 8 * ntile, k))
    print("1. ldmatrix.x4 lane addressing delivers exactly the m16n8k32 A and B fragments (both n-tiles)")


def test_swizzle_conflicts():
    # an ldmatrix phase reads 8 rows (one tile) at the same logical chunk; bank group = (phys chunk) % 8 must differ
    for r0 in (0, 8):
        for L in range(CHUNKS16):
            groups = {(swz(r0 + r, L)) % 8 for r in range(8)}
            assert len(groups) == 8, (r0, L, groups)
    print("2. row swizzle (chunk ^ (row & 7)) is bank-conflict free for every ldmatrix phase")


def test_cp_async_coverage():
    for nb in (BPC, 5, 2, 1):
        seen_a, seen_b = {}, {}
        for tid in range(THREADS):
            for i in range(tid, TILE_M * CHUNKS16, THREADS):
                row, ch = i // CHUNKS16, i % CHUNKS16
                key = (row, swz(row, ch)); assert key not in seen_a; seen_a[key] = ch
            for i in range(tid, WARPS * COLS_PER_WARP * CHUNKS16, THREADS):
                w = i // (COLS_PER_WARP * CHUNKS16); r = (i // CHUNKS16) % COLS_PER_WARP; ch = i % CHUNKS16
                key = (w, r, swz(r, ch)); assert key not in seen_b; seen_b[key] = ch
        assert len(seen_a) == TILE_M * CHUNKS16 and len(seen_b) == WARPS * COLS_PER_WARP * CHUNKS16
        # the swizzle is a bijection per row, so every physical chunk is written exactly once
        for row in range(TILE_M):
            assert sorted(ch for (r, p), ch in seen_a.items() if r == row) == list(range(CHUNKS16))
    print("3. cp.async assignment writes every (row, chunk) of A and B exactly once; swizzle is a per-row bijection")


def test_split_chunking():
    for K, N in ((5120, 1792), (5120, 4608), (2304, 5120), (8192, 5120), (1280, 32768), (1280, 4096)):
        T = K // BLOCK_K
        tps = -(-T // SPLIT_K)
        ranges = [(s * tps, min((s + 1) * tps, T)) for s in range(SPLIT_K)]
        assert ranges == split_ranges(K), (K, ranges, split_ranges(K))
        for lo, hi in ranges:
            nblk = max(hi - lo, 0); nchunk = -(-nblk // BPC)
            # every block of the split is covered by exactly one chunk, chunks never cross the split boundary
            blocks = [t for c in range(nchunk) for t in range(lo + c * BPC, lo + min(BPC, nblk - c * BPC) + c * BPC)]
            assert blocks == list(range(lo, hi)), (K, lo, hi, blocks[:4])
        assert N % BN == 0 and (N // BN) * SPLIT_K >= 80, (K, N)   # at least one CTA per SM
    print("4. split partition equals the pinned tiles_per_split rule on all six shapes; chunks never cross a split")


def test_smem_budget():
    stage = TILE_M * CHUNK_BYTES + WARPS * COLS_PER_WARP * CHUNK_BYTES + TILE_M * BPC * 4 + BPC * 4
    total = stage * STAGES
    assert stage % 16 == 0 and total <= 48 * 1024, total           # no opt-in attribute needed
    assert 2 * (total + 1024) <= 100 * 1024, total                   # two CTAs per SM within the 100 KiB carveout
    print("5. shared memory per CTA %d bytes (%d per stage): under 48 KiB, two CTAs per SM" % (total, stage))


if __name__ == "__main__":
    test_fragments(); test_swizzle_conflicts(); test_cp_async_coverage(); test_split_chunking(); test_smem_budget()
    print("SELFTEST_PASS")
