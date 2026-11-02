"""One 64-value tile of a V41_FP4 cache row on SM89 (SGLANG_SM89_FP4_COMPRESSED_KV=1, Claude, 2 Nov 2026).

The ratio-1/2 latents are E2M1 with E4M3 scales per 16, so this layout holds them without the second rounding the V4 layout needs.
Decoded values match what `_rope_fq4` hands the V4 store: E2M1 x E4M3 has at most 6 significant bits, so it is exact in BF16.
"""
import triton
import triton.language as tl

FP4_DATA_BYTES, FP4_SCALE_BYTES = tl.constexpr(256), tl.constexpr(32)
FP4_BYTES_PER_TOKEN = 288


@triton.jit
def _e2m1_bits(code):
    """The FP32 bits of a 4-bit E2M1 code, in integers: bit 3 the sign, bits 0-2 index [0, .5, 1, 1.5, 2, 3, 4, 6]. A negative zero stays -0.0."""
    m = code & 7
    bits = tl.where(m >= 2, (((m >> 1) + 126) << 23) | ((m & 1) << 22), tl.where(m == 1, 0x3F000000, 0))
    return bits | ((code & 8) << 28)


@triton.jit
def fp4_tile(kv_block_base, data_off, scale_off, valid_2d, tile):
    """Tile 0-7 of each token, [tokens, 64] BF16: 32 bytes of codes, the even element in the low nibble, and the tile's four scales."""
    BYTES: tl.constexpr = 32
    offs_b = tl.arange(0, BYTES)
    base = kv_block_base[:, None] + data_off[:, None]
    packed = tl.load(base + tile * BYTES + offs_b[None, :], mask=valid_2d, other=0).to(tl.int32)
    scale_u8 = tl.load(kv_block_base[:, None] + scale_off[:, None] + tile * 4 + offs_b[None, :] // 8, mask=valid_2d, other=0)
    scale = scale_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    lo = _e2m1_bits(packed & 15).to(tl.float32, bitcast=True) * scale
    hi = _e2m1_bits(packed >> 4).to(tl.float32, bitcast=True) * scale
    kv = tl.interleave(lo, hi).to(tl.bfloat16)
    return tl.where(valid_2d, kv, 0.0)
