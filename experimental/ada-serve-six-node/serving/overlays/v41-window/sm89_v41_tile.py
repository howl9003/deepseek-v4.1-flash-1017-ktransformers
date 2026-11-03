"""One 64-value tile of a V41 window KV row on SM89 (SGLANG_SM89_V41_KV=1, Claude, 2 Nov 2026).

The 1017 reference quantizes all 512 values after RoPE, E4M3 with one UE8M0 scale per 32. The V4 layout SM89 used holds 448 of them with a scale per 64 and leaves the RoPE values in BF16.
Decoded values match the reference's fused quant + dequant: E4M3 x a power of two has at most 4 significant bits, so it is exact in BF16.
"""
import triton
import triton.language as tl

V41_DATA_BYTES, V41_SCALE_BYTES = tl.constexpr(512), tl.constexpr(16)
V41_BYTES_PER_TOKEN = 528


@triton.jit
def v41_tile(kv_block_base, data_off, scale_off, valid_2d, tile):
    """Tile 0-7 of each token, [tokens, 64] BF16: 64 E4M3 values and the tile's two scales. A UE8M0 byte b is 2^(b - 127), which is the FP32 with exponent field b."""
    BYTES: tl.constexpr = 64
    offs_b = tl.arange(0, BYTES)
    base = kv_block_base[:, None] + data_off[:, None]
    fp8 = tl.load(base + tile * BYTES + offs_b[None, :], mask=valid_2d, other=0).to(tl.float8e4nv, bitcast=True)
    scale_u8 = tl.load(kv_block_base[:, None] + scale_off[:, None] + tile * 2 + offs_b[None, :] // 32, mask=valid_2d, other=127)
    scale = (scale_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    kv = (fp8.to(tl.float32) * scale).to(tl.bfloat16)
    return tl.where(valid_2d, kv, 0.0)
