"""Overlay the FP4 compressed KV on SM89, selector SGLANG_SM89_FP4_COMPRESSED_KV, default off. argv: source sglang package, destination package, sm89_fp4_tile.py."""
import re
import shutil
import sys
from pathlib import Path

SRC, DST, TILE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def rd(rel):
    return (SRC / rel).read_text(encoding='utf-8')


def wr(rel, s):
    if rel.endswith('.py'):
        compile(s, rel, 'exec')
    p = DST / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(s.encode('utf-8'))


def sub(s, old, new, n=1):
    assert s.count(old) == n, (s.count(old), old[:90])
    return s.replace(old, new)


# Layout rule
rel = 'kernels/ops/attention/dsv4/kv_layout.py'
s = rd(rel)
s = sub(s, 'import enum\nimport functools\n', 'import enum\nimport functools\nimport os\n')
s = sub(s, 'def is_valid_kv_layout_pair(kv: KVLayout, extra_kv: KVLayout) -> bool:\n',
        'def sm89_fp4_compressed_kv() -> bool:\n'
        '    """SGLANG_SM89_FP4_COMPRESSED_KV=1: on SM89 the ratio-1/2 caches keep their e2m1\n'
        '    codes next to the V4 main cache, so the V4 layout\'s second rounding goes."""\n'
        '    return os.environ.get("SGLANG_SM89_FP4_COMPRESSED_KV", "0") == "1"\n'
        '\n\n'
        'def is_valid_kv_layout_pair(kv: KVLayout, extra_kv: KVLayout) -> bool:\n')
s = sub(s, '    return extra_kv is kv or (\n        kv.encoding is KVLayout.V41\n',
        '    if kv is KVLayout.V4 and extra_kv is KVLayout.V41_FP4:\n'
        '        return sm89_fp4_compressed_kv()\n'
        '    return extra_kv is kv or (\n        kv.encoding is KVLayout.V41\n')
wr(rel, s)

rel = 'srt/mem_cache/deepseek_v4_memory_pool.py'
s = rd(rel)
s = sub(s, 'from sglang.kernels.ops.attention.dsv4.kv_layout import (\n',
        'from sglang.kernels.ops.attention.dsv4.kv_layout import (\n    sm89_fp4_compressed_kv,\n')
s = sub(s, '    if kv_layout is KVLayout.V4:\n        assert option in (None, "fp8"), "the V4 main cache only pairs with V4 caches"\n',
        '    if kv_layout is KVLayout.V4:\n'
        '        if compress_ratio in (1, 2) and sm89_fp4_compressed_kv():\n'
        '            assert option in (None, "fp4"), "the SM89 fp4 selector picks the layout"\n'
        '            assert torch.cuda.get_device_capability() == (8, 9), "SM89 only"\n'
        '            return KVLayout.V41_FP4\n'
        '        assert option in (None, "fp8"), "the V4 main cache only pairs with V4 caches"\n')
wr(rel, s)

# Writer: software E2M1 codes below SM100
rel = 'kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh'
s = rd(rel)
s = sub(s, '// FP4 (e2m1) helpers: per-32 UE8M0 for the indexer, per-16 E4M3 for compressed KV.\n',
        '// E2M1 codes in software on ROCm and below SM100, which has no cvt.rn.satfinite.e2m1x2.\n'
        '#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000)\n'
        '#define SGL_FP4_SOFT_CODES 1\n'
        '#else\n'
        '#define SGL_FP4_SOFT_CODES 0\n'
        '#endif\n'
        '\n'
        '// FP4 (e2m1) helpers: per-32 UE8M0 for the indexer, per-16 E4M3 for compressed KV.\n')
s = sub(s, 'constexpr uint32_t kCompressedKVBlockSize = 16;\n#ifdef USE_ROCM\n/// \\brief `cvt.rn.satfinite.e2m1x2.f32` for one value',
        'constexpr uint32_t kCompressedKVBlockSize = 16;\n#if SGL_FP4_SOFT_CODES\n/// \\brief `cvt.rn.satfinite.e2m1x2.f32` for one value')
wr(rel, s)

rel = 'kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh'
s = rd(rel)
s = sub(s, '#ifdef USE_ROCM\n    out[i] = static_cast<uint8_t>(fp4::e2m1x2_code(', '#if SGL_FP4_SOFT_CODES\n    out[i] = static_cast<uint8_t>(fp4::e2m1x2_code(')
wr(rel, s)

rel = 'kernels/ops/attention/dsv4/dequant_k_cache.py'
s = rd(rel)
s = sub(s, '            SINGLE_ROUTE and layout is KVLayout.V41\n            and torch.cuda.get_device_capability(quant_k_cache.device) == (8, 9)\n',
        '            (SINGLE_ROUTE and layout is KVLayout.V41 or sm89_fp4_compressed_kv() and layout is KVLayout.V41_FP4)\n            and torch.cuda.get_device_capability(quant_k_cache.device) == (8, 9)\n')
s = sub(s, 'from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout\n',
        'from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout, sm89_fp4_compressed_kv\n')
wr(rel, s)

# Readers
shutil.copyfile(TILE, DST / 'kernels/ops/attention/sm89_fp4_tile.py')

rel = 'kernels/ops/attention/sm89_gather_staged.py'
s = rd(rel)
s = sub(s, 'import triton.language as tl\n\n_KV_FP8_TY',
        'import triton.language as tl\n\nfrom sglang.kernels.ops.attention.sm89_fp4_tile import FP4_BYTES_PER_TOKEN, FP4_DATA_BYTES, FP4_SCALE_BYTES, fp4_tile\n\n_KV_FP8_TY')
s = sub(s, 'def _block_valid(kv, idx_base, offs_n, n_limit, topk_len, block_size, stride_kv, HAS_LEN: tl.constexpr):',
        'def _block_valid(kv, idx_base, offs_n, n_limit, topk_len, block_size, stride_kv, HAS_LEN: tl.constexpr, KV_FP4: tl.constexpr):')
s = sub(s, '    return kv_block_base, offset_64 * 576, block_size * 576 + offset_64 * 8, valid\n',
        '    if KV_FP4:\n'
        '        data_off, scale_off = offset_64 * FP4_DATA_BYTES, block_size * FP4_DATA_BYTES + offset_64 * FP4_SCALE_BYTES\n'
        '    else:\n'
        '        data_off, scale_off = offset_64 * 576, block_size * 576 + offset_64 * 8\n'
        '    return kv_block_base, data_off, scale_off, valid\n')
s = sub(s, 'mx_ptrs, qk_base, sm_scale, stride_q_d, HAS_LEN: tl.constexpr,\n                 BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, TILE_SIZE: tl.constexpr, D_NOPE: tl.constexpr):',
        'mx_ptrs, qk_base, sm_scale, stride_q_d, HAS_LEN: tl.constexpr,\n                 BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, TILE_SIZE: tl.constexpr, D_NOPE: tl.constexpr, KV_FP4: tl.constexpr):')
s = sub(s, 'n_limit, topk_len, block_size, stride_kv, HAS_LEN)\n    valid_2d = valid[:, None]\n    qk = tl.zeros',
        'n_limit, topk_len, block_size, stride_kv, HAS_LEN, KV_FP4)\n    valid_2d = valid[:, None]\n    qk = tl.zeros')
dots = [f'    qk += tl.dot(q_{i}, tl.trans(_tile_nope(kv_block_base, data_off, scale_off, valid, valid_2d, offs_tile, {i}, TILE_SIZE))).to(tl.float32)\n' for i in range(7)]
dots.append('    qk += tl.dot(q_7, tl.trans(_tile_rope(kv_block_base, data_off, valid_2d, offs_tile, D_NOPE))).to(tl.float32)\n')
old = ''.join(dots)
new = '    if KV_FP4:\n' + ''.join(f'        qk += tl.dot(q_{i}, tl.trans(fp4_tile(kv_block_base, data_off, scale_off, valid_2d, {i}))).to(tl.float32)\n' for i in range(8))
new += '    else:\n' + ''.join('    ' + d for d in dots)
s = sub(s, old, new)
s = sub(s, 'stride_mx_t, stride_mx_b,\n                   HAS_LEN_MAIN: tl.constexpr, HAS_LEN_EXTRA: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr):',
        'stride_mx_t, stride_mx_b,\n                   HAS_LEN_MAIN: tl.constexpr, HAS_LEN_EXTRA: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, EXTRA_FP4: tl.constexpr):')
s = sub(s, 'sm_scale, stride_q_d, HAS_LEN_MAIN, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE)', 'sm_scale, stride_q_d, HAS_LEN_MAIN, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, False)')
s = sub(s, 'sm_scale, stride_q_d, HAS_LEN_EXTRA, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE)', 'sm_scale, stride_q_d, HAS_LEN_EXTRA, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, EXTRA_FP4)')
s = sub(s, 'stride_o_t, stride_o_h, stride_o_d, HAS_LEN_MAIN: tl.constexpr, HAS_LEN_EXTRA: tl.constexpr, HAS_ATTN_SINK: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr):',
        'stride_o_t, stride_o_h, stride_o_d, HAS_LEN_MAIN: tl.constexpr, HAS_LEN_EXTRA: tl.constexpr, HAS_ATTN_SINK: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, EXTRA_FP4: tl.constexpr):')
s = sub(s, 'block_size_main, stride_kv_main, HAS_LEN_MAIN)\n', 'block_size_main, stride_kv_main, HAS_LEN_MAIN, False)\n')
s = sub(s, 'block_size_extra, stride_kv_extra, HAS_LEN_EXTRA)\n', 'block_size_extra, stride_kv_extra, HAS_LEN_EXTRA, EXTRA_FP4)\n')
kv_lines = ('            kv = tl.where(pid_d < 7, _tile_nope(kv_block_base, data_off, scale_off, valid & (pid_d < 7), valid_2d & (pid_d < 7), offs_tile, tl.minimum(pid_d, 6), TILE_SIZE),\n'
            '                          _tile_rope(kv_block_base, data_off, valid_2d & (pid_d == 7), offs_tile, D_NOPE))\n')
assert s.count(kv_lines) == 2
head, tail = s.rsplit(kv_lines, 1)
s = head + ('            if EXTRA_FP4:\n                kv = fp4_tile(kv_block_base, data_off, scale_off, valid_2d, pid_d)\n            else:\n'
            + ''.join('    ' + l + '\n' for l in kv_lines.splitlines()) + tail)
s = sub(s, "    flags = dict(HAS_LEN_MAIN=topk_length is not None, HAS_LEN_EXTRA=extra_topk_length is not None)\n",
        "    extra_fp4 = has_extra and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN\n"
        "    flags = dict(HAS_LEN_MAIN=topk_length is not None, HAS_LEN_EXTRA=extra_topk_length is not None, EXTRA_FP4=extra_fp4)\n")
wr(rel, s)

rel = 'kernels/ops/attention/dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py'
s = rd(rel)
s = sub(s, 'from sglang.srt.utils.common import is_gfx942_supported\n',
        'from sglang.kernels.ops.attention.sm89_fp4_tile import (\n    FP4_DATA_BYTES,\n    FP4_SCALE_BYTES,\n    fp4_tile,\n)\nfrom sglang.srt.utils.common import is_gfx942_supported\n')
start = s.index('@triton.jit\ndef _process_kv_block_aggressive(')
end = s.index('# ============================================================================\n# DSV4 Fused Gather+Dequant+Attention Kernel (Single Scope)')
fn = s[start:end]
body_at = fn.index('    scale_ptrs = kv_block_base + scale_base_offset\n')
tail_at = fn.index('    qk = qk * sm_scale\n')
loads = '    qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)\n\n' + ''.join(
    f'    kv_{i} = fp4_tile(kv_block_base, nope_rope_offset, scale_base_offset, valid_2d, {i})\n'
    f'    qk += tl.dot(q_{i}, tl.trans(kv_{i})).to(tl.float32)\n\n' for i in range(8))
fp4_fn = fn[:body_at].replace('def _process_kv_block_aggressive(', 'def _process_kv_block_fp4(').replace(
    '    Process one block of KV tokens with batch loading.\n    Key optimization: Load all KV tiles first, then process them.\n',
    '    _process_kv_block_aggressive over a V41_FP4 extra cache on SM89: the same dots and folds on\n    the same shapes, only the tiles load differently.\n') + loads + fn[tail_at:]
s = s[:end] + fp4_fn + s[end:]
ks = s.index('def _fused_gather_attn_dsv4_dual_scope_kernel(')
ke = s.index('def _fused_gather_attn_dsv4_dual_scope_splitk_kernel(')
k = s[ks:ke]
k = sub(k, '    BLOCK_H: tl.constexpr,\n    BLOCK_N: tl.constexpr,\n):\n', '    BLOCK_H: tl.constexpr,\n    BLOCK_N: tl.constexpr,\n    EXTRA_FP4: tl.constexpr = False,\n):\n')
off_old = ('            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA\n'
           '            scale_base_offset = (\n'
           '                block_size_extra * BYTES_PER_TOKEN_DATA\n'
           '                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE\n'
           '            )\n')
off_new = ('            if EXTRA_FP4:\n'
           '                nope_rope_offset = offset_in_block_64 * FP4_DATA_BYTES\n'
           '                scale_base_offset = (\n'
           '                    block_size_extra * FP4_DATA_BYTES\n'
           '                    + offset_in_block_64 * FP4_SCALE_BYTES\n'
           '                )\n'
           '            else:\n'
           + ''.join('    ' + l + '\n' for l in off_old.splitlines()))
k = sub(k, off_old, off_new)
m = re.search(r'            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = \(\n                _process_kv_block_aggressive\((?:.|\n)*?\n                \)\n            \)\n', k[k.index('Indices_Extra + pid_t * stride_idx_extra_t'):])
call = m.group(0)
assert k.count(call) == 2  # main, then extra
fp4_call = call.replace('_process_kv_block_aggressive', '_process_kv_block_fp4')
head, tail = k.rsplit(call, 1)
k = (head + '            if EXTRA_FP4:\n' + ''.join('    ' + l + '\n' for l in fp4_call.splitlines())
     + '            else:\n' + ''.join('    ' + l + '\n' for l in call.splitlines()) + tail)
s = s[:ks] + k + s[ke:]
wr(rel, s)

rel = 'kernels/ops/attention/sm89_dual_scope.py'
s = rd(rel)
s = sub(s, 'from sglang.kernels.ops.attention.dsv4.triton_sparse_decode import (\n',
        'from sglang.kernels.ops.attention.sm89_fp4_tile import FP4_BYTES_PER_TOKEN\nfrom sglang.kernels.ops.attention.dsv4.triton_sparse_decode import (\n')
s = sub(s, '    for c in (k_cache, extra_k_cache):\n        if c is None:\n            continue\n'
           '        if c.ndim != 4 or c.shape[-1] != _V4_BYTES_PER_TOKEN or c.element_size() != 1:\n',
        '    for c, widths in _CACHE_WIDTHS(k_cache, extra_k_cache):\n        if c is None:\n            continue\n'
        '        if c.ndim != 4 or c.shape[-1] not in widths or c.element_size() != 1:\n')
s = sub(s, '        if c.stride(-1) != 1 or c.stride(1) != _V4_BYTES_PER_TOKEN:\n', '        if c.stride(-1) != 1 or c.stride(1) != c.shape[-1]:\n')
s = sub(s, 'def supported(q, k_cache, extra_k_cache, head_dim_v, attn_sink) -> bool:\n',
        'def _CACHE_WIDTHS(k_cache, extra_k_cache):\n'
        '    # The extra cache may be FP4 (SGLANG_SM89_FP4_COMPRESSED_KV); the main cache is always V4.\n'
        '    return (k_cache, (_V4_BYTES_PER_TOKEN,)), (extra_k_cache, (_V4_BYTES_PER_TOKEN, FP4_BYTES_PER_TOKEN))\n'
        '\n\n'
        'def supported(q, k_cache, extra_k_cache, head_dim_v, attn_sink) -> bool:\n')
s = sub(s, '                dict(flags, HAS_ATTN_SINK=attn_sink is not None),\n',
        '                dict(flags, HAS_ATTN_SINK=attn_sink is not None, EXTRA_FP4=extra_fp4),\n')
s = sub(s, '    split_k = _split_k(rows, h_q, topk_main + topk_extra)\n',
        '    extra_fp4 = has_extra and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN\n    split_k = _split_k(rows, h_q, topk_main + topk_extra)\n')
s = sub(s, '    topk_per_split = (topk_main + topk_extra + split_k - 1) // split_k\n',
        '    if extra_fp4:\n        raise ValueError("the FP4 compressed KV needs SGLANG_SM89_ATTN_SPLITK=0")\n    topk_per_split = (topk_main + topk_extra + split_k - 1) // split_k\n')
s = sub(s, '    for cache in (k_cache, extra_k_cache):\n        if cache is not None and (\n            cache.shape[-1] != _V4_BYTES_PER_TOKEN or cache.element_size() != 1\n        ):\n            raise ValueError("SM89 attention requires the V4 584-byte KV layout")\n',
        '    for cache, widths in _CACHE_WIDTHS(k_cache, extra_k_cache):\n        if cache is not None and (\n            cache.shape[-1] not in widths or cache.element_size() != 1\n        ):\n            raise ValueError("SM89 attention requires the V4 584-byte KV layout")\n')
s = sub(s, '    if not use_dual:\n        from sglang.kernels.ops.attention.flash_mla_sm120_triton import (\n',
        '    if not use_dual:\n'
        '        if extra_k_cache is not None and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN:\n'
        '            raise ValueError("the portable SM89 path cannot read the FP4 compressed KV")\n'
        '        from sglang.kernels.ops.attention.flash_mla_sm120_triton import (\n')
wr(rel, s)
print('patched')
