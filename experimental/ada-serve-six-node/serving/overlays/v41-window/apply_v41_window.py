"""Overlay the V41 window KV on the served SM89 attention route, selector SGLANG_SM89_V41_KV, default off. Runs after apply_fp4_kv.py and implies its FP4 compressed caches.
argv: source sglang package, destination package, sm89_v41_tile.py."""
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
s = sub(s, 'def sm89_fp4_compressed_kv() -> bool:\n',
        'def sm89_v41_kv() -> bool:\n'
        '    """SGLANG_SM89_V41_KV=1: the served SM89 route reads a V41 main cache next to V41_FP4\n'
        '    compressed caches, so window and compressed KV both sit on the reference\'s grid."""\n'
        '    return os.environ.get("SGLANG_SM89_V41_KV", "0") == "1"\n'
        '\n\n'
        'def sm89_fp4_compressed_kv() -> bool:\n')
s = sub(s, '    return os.environ.get("SGLANG_SM89_FP4_COMPRESSED_KV", "0") == "1"\n',
        '    return os.environ.get("SGLANG_SM89_FP4_COMPRESSED_KV", "0") == "1" or sm89_v41_kv()\n')
wr(rel, s)

rel = 'srt/mem_cache/deepseek_v4_memory_pool.py'
s = rd(rel)
s = sub(s, '    sm89_fp4_compressed_kv,\n', '    sm89_fp4_compressed_kv,\n    sm89_v41_kv,\n')
s = sub(s, '    if get_exec().kernel.dsv4_attn_backend == "flashmla_mega" and mode != "v41_new":\n',
        '    if sm89_v41_kv():\n'
        '        assert torch.cuda.get_device_capability() == (8, 9), "SM89 only"\n'
        '        return KVLayout.V41, None\n'
        '    if get_exec().kernel.dsv4_attn_backend == "flashmla_mega" and mode != "v41_new":\n')
wr(rel, s)

rel = 'kernels/ops/attention/dsv4/dequant_k_cache.py'
s = rd(rel)
s = sub(s, '            (SINGLE_ROUTE and layout is KVLayout.V41 or sm89_fp4_compressed_kv()',
        '            ((SINGLE_ROUTE or sm89_v41_kv()) and layout is KVLayout.V41 or sm89_fp4_compressed_kv()')
s = sub(s, 'from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout, sm89_fp4_compressed_kv\n',
        'from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout, sm89_fp4_compressed_kv, sm89_v41_kv\n')
wr(rel, s)

# Writer: DeepSeek floors amax at 1e-4 before dividing by 448, so the scale floor is 2^-22. SGLang floored the quotient, 2^-13.
rel = 'kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh'
s = rd(rel)
s = sub(s, '/// `scale_row` its 16 scale bytes. Scale `2^ceil(log2(max(amax / 448, 1e-4)))` stored as the\n',
        '/// `scale_row` its 16 scale bytes. Scale `2^ceil(log2(max(amax, 1e-4) / 448))` stored as the\n')
s = sub(s, '  // ceil(log2(max(amax / 448, 1e-4))) exactly, from the bits of amax: 448 = 1.75 * 2^8, so the\n'
           '  // quotient\'s exponent is amax\'s minus 8, plus one when the mantissa exceeds 1.75, floored at 2^-13.\n',
        '  // ceil(log2(max(amax, 1e-4) / 448)) exactly, from the bits of amax: 448 = 1.75 * 2^8, so the\n'
        '  // quotient\'s exponent is amax\'s minus 8, plus one when the mantissa exceeds 1.75, floored at 2^-22.\n')
s = sub(s, '      max(static_cast<int32_t>(bits >> 23) - 8 + static_cast<int32_t>((bits & 0x7FFFFFu) > 0x600000u), 114);\n',
        '      max(static_cast<int32_t>(bits >> 23) - 8 + static_cast<int32_t>((bits & 0x7FFFFFu) > 0x600000u), 105);\n')
wr(rel, s)

# Readers
shutil.copyfile(TILE, DST / 'kernels/ops/attention/sm89_v41_tile.py')

rel = 'kernels/ops/attention/sm89_gather_staged.py'
s = rd(rel)
s = sub(s, 'from sglang.kernels.ops.attention.sm89_fp4_tile import FP4_BYTES_PER_TOKEN, FP4_DATA_BYTES, FP4_SCALE_BYTES, fp4_tile\n',
        'from sglang.kernels.ops.attention.sm89_fp4_tile import FP4_BYTES_PER_TOKEN, FP4_DATA_BYTES, FP4_SCALE_BYTES, fp4_tile\n'
        'from sglang.kernels.ops.attention.sm89_v41_tile import V41_BYTES_PER_TOKEN, V41_DATA_BYTES, V41_SCALE_BYTES, v41_tile\n')
s = sub(s, 'stride_kv, HAS_LEN: tl.constexpr, KV_FP4: tl.constexpr):\n    """One token block',
        'stride_kv, HAS_LEN: tl.constexpr, KV_FP4: tl.constexpr, KV_V41: tl.constexpr):\n    """One token block')
s = sub(s, '    if KV_FP4:\n        data_off, scale_off = offset_64 * FP4_DATA_BYTES',
        '    if KV_V41:\n'
        '        data_off, scale_off = offset_64 * V41_DATA_BYTES, block_size * V41_DATA_BYTES + offset_64 * V41_SCALE_BYTES\n'
        '    elif KV_FP4:\n        data_off, scale_off = offset_64 * FP4_DATA_BYTES')
s = sub(s, 'D_NOPE: tl.constexpr, KV_FP4: tl.constexpr):\n', 'D_NOPE: tl.constexpr, KV_FP4: tl.constexpr, KV_V41: tl.constexpr):\n')
s = sub(s, 'n_limit, topk_len, block_size, stride_kv, HAS_LEN, KV_FP4)\n    valid_2d = valid[:, None]\n    qk = tl.zeros',
        'n_limit, topk_len, block_size, stride_kv, HAS_LEN, KV_FP4, KV_V41)\n    valid_2d = valid[:, None]\n    qk = tl.zeros')
s = sub(s, '    if KV_FP4:\n        qk += tl.dot(q_0,',
        '    if KV_V41:\n' + ''.join(f'        qk += tl.dot(q_{i}, tl.trans(v41_tile(kv_block_base, data_off, scale_off, valid_2d, {i}))).to(tl.float32)\n' for i in range(8))
        + '    elif KV_FP4:\n        qk += tl.dot(q_0,')
s = sub(s, 'BLOCK_N: tl.constexpr, EXTRA_FP4: tl.constexpr):\n', 'BLOCK_N: tl.constexpr, EXTRA_FP4: tl.constexpr, MAIN_V41: tl.constexpr):\n', 2)
s = sub(s, 'sm_scale, stride_q_d, HAS_LEN_MAIN, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, False)', 'sm_scale, stride_q_d, HAS_LEN_MAIN, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, False, MAIN_V41)')
s = sub(s, 'sm_scale, stride_q_d, HAS_LEN_EXTRA, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, EXTRA_FP4)', 'sm_scale, stride_q_d, HAS_LEN_EXTRA, BLOCK_H, BLOCK_N, TILE_SIZE, D_NOPE, EXTRA_FP4, False)')
s = sub(s, 'block_size_main, stride_kv_main, HAS_LEN_MAIN, False)\n', 'block_size_main, stride_kv_main, HAS_LEN_MAIN, False, MAIN_V41)\n')
s = sub(s, 'block_size_extra, stride_kv_extra, HAS_LEN_EXTRA, EXTRA_FP4)\n', 'block_size_extra, stride_kv_extra, HAS_LEN_EXTRA, EXTRA_FP4, False)\n')
kv_lines = ('            kv = tl.where(pid_d < 7, _tile_nope(kv_block_base, data_off, scale_off, valid & (pid_d < 7), valid_2d & (pid_d < 7), offs_tile, tl.minimum(pid_d, 6), TILE_SIZE),\n'
            '                          _tile_rope(kv_block_base, data_off, valid_2d & (pid_d == 7), offs_tile, D_NOPE))\n')
s = sub(s, '\n' + kv_lines, '\n            if MAIN_V41:\n                kv = v41_tile(kv_block_base, data_off, scale_off, valid_2d, pid_d)\n            else:\n'
        + ''.join('    ' + l + '\n' for l in kv_lines.splitlines()))
s = sub(s, 'HAS_LEN_EXTRA=extra_topk_length is not None, EXTRA_FP4=extra_fp4)\n',
        'HAS_LEN_EXTRA=extra_topk_length is not None, EXTRA_FP4=extra_fp4, MAIN_V41=k_cache.shape[-1] == V41_BYTES_PER_TOKEN)\n')
wr(rel, s)

rel = 'kernels/ops/attention/dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py'
s = rd(rel)
s = sub(s, 'from sglang.srt.utils.common import is_gfx942_supported\n',
        'from sglang.kernels.ops.attention.sm89_v41_tile import (\n    V41_DATA_BYTES,\n    V41_SCALE_BYTES,\n    v41_tile,\n)\nfrom sglang.srt.utils.common import is_gfx942_supported\n')
start = s.index('@triton.jit\ndef _process_kv_block_fp4(')
end = s.index('# ============================================================================\n# DSV4 Fused Gather+Dequant+Attention Kernel (Single Scope)')
fn = s[start:end]
v41_fn = fn.replace('def _process_kv_block_fp4(', 'def _process_kv_block_v41(').replace('fp4_tile', 'v41_tile').replace(
    '    _process_kv_block_aggressive over a V41_FP4 extra cache on SM89: the same dots and folds on\n',
    '    _process_kv_block_aggressive over a V41 main cache on SM89: the same dots and folds on\n')
assert v41_fn != fn and v41_fn.count('V41_FP4') == 0 and 'V41 main cache' in v41_fn
s = s[:end] + v41_fn + s[end:]
ks = s.index('def _fused_gather_attn_dsv4_dual_scope_kernel(')
ke = s.index('def _fused_gather_attn_dsv4_dual_scope_splitk_kernel(')
k = s[ks:ke]
k = sub(k, '    EXTRA_FP4: tl.constexpr = False,\n):\n', '    EXTRA_FP4: tl.constexpr = False,\n    MAIN_V41: tl.constexpr = False,\n):\n')
off_old = ('            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA\n'
           '            scale_base_offset = (\n'
           '                block_size_main * BYTES_PER_TOKEN_DATA\n'
           '                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE\n'
           '            )\n')
off_new = ('            if MAIN_V41:\n'
           '                nope_rope_offset = offset_in_block_64 * V41_DATA_BYTES\n'
           '                scale_base_offset = (\n'
           '                    block_size_main * V41_DATA_BYTES\n'
           '                    + offset_in_block_64 * V41_SCALE_BYTES\n'
           '                )\n'
           '            else:\n'
           + ''.join('    ' + l + '\n' for l in off_old.splitlines()))
k = sub(k, off_old, off_new)
main_at = k.index('kv_block_base = KV_Cache_Main + block_idx_64 * stride_kv_block_main_64')
m = re.compile(r'            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = \(\n                _process_kv_block_aggressive\((?:.|\n)*?\n                \)\n            \)\n').search(k, main_at)
call = m.group(0)
v41_call = call.replace('_process_kv_block_aggressive', '_process_kv_block_v41')
k = (k[:m.start()] + '            if MAIN_V41:\n' + ''.join('    ' + l + '\n' for l in v41_call.splitlines())
     + '            else:\n' + ''.join('    ' + l + '\n' for l in call.splitlines()) + k[m.end():])
s = s[:ks] + k + s[ke:]
wr(rel, s)

rel = 'kernels/ops/attention/sm89_dual_scope.py'
s = rd(rel)
s = sub(s, 'from sglang.kernels.ops.attention.sm89_fp4_tile import FP4_BYTES_PER_TOKEN\n',
        'from sglang.kernels.ops.attention.sm89_fp4_tile import FP4_BYTES_PER_TOKEN\nfrom sglang.kernels.ops.attention.sm89_v41_tile import V41_BYTES_PER_TOKEN\n')
s = sub(s, '    # The extra cache may be FP4 (SGLANG_SM89_FP4_COMPRESSED_KV); the main cache is always V4.\n'
           '    return (k_cache, (_V4_BYTES_PER_TOKEN,)), (extra_k_cache, (_V4_BYTES_PER_TOKEN, FP4_BYTES_PER_TOKEN))\n',
        '    # The extra cache may be FP4 (SGLANG_SM89_FP4_COMPRESSED_KV), the main cache V41 (SGLANG_SM89_V41_KV).\n'
        '    return (k_cache, (_V4_BYTES_PER_TOKEN, V41_BYTES_PER_TOKEN)), (extra_k_cache, (_V4_BYTES_PER_TOKEN, FP4_BYTES_PER_TOKEN))\n')
s = sub(s, '    extra_fp4 = has_extra and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN\n    split_k = _split_k(',
        '    extra_fp4 = has_extra and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN\n    main_v41 = k_cache.shape[-1] == V41_BYTES_PER_TOKEN\n    split_k = _split_k(')
s = sub(s, 'dict(flags, HAS_ATTN_SINK=attn_sink is not None, EXTRA_FP4=extra_fp4),\n', 'dict(flags, HAS_ATTN_SINK=attn_sink is not None, EXTRA_FP4=extra_fp4, MAIN_V41=main_v41),\n')
s = sub(s, '    if extra_fp4:\n        raise ValueError("the FP4 compressed KV needs SGLANG_SM89_ATTN_SPLITK=0")\n',
        '    if extra_fp4 or main_v41:\n        raise ValueError("the FP4 and V41 KV layouts need SGLANG_SM89_ATTN_SPLITK=0")\n')
s = sub(s, '        if extra_k_cache is not None and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN:\n            raise ValueError("the portable SM89 path cannot read the FP4 compressed KV")\n',
        '        if k_cache.shape[-1] == V41_BYTES_PER_TOKEN or (extra_k_cache is not None and extra_k_cache.shape[-1] == FP4_BYTES_PER_TOKEN):\n            raise ValueError("the portable SM89 path cannot read the FP4 or V41 KV layouts")\n')
wr(rel, s)
print('patched')
