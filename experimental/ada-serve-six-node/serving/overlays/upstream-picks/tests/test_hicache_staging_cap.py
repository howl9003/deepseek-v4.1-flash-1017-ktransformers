"""CPU-only tests for apply_hicache_staging_cap.py (upstream-picks-r1).

argv: the served sglang package directory (srt/mem_cache/memory_pool_host.py and kernels/ops/kvcache/hicache.py at
the served pins). The test copies memory_pool_host.py, applies the overlay to the copy, and checks:
  1. _staging_page_capacity: unset gives min(pages, 64) for every input; 32 MiB gives 6 pages for our 'swa' pool and
     leaves the c1/c2 pools at 64 (geometry from the served launcher.log); bad values are refused;
  2. both pools' _init_write_back_staging_buffers, served and patched, run on stand-ins: unset allocates exactly the
     served buffer; 32 MiB allocates the capped one (shapes only, nothing allocated);
  3. the served staged write-back loop (transfer_hicache_all_layer_mla_staged_lf_pf, unchanged) with a numpy stand-in
     for the jit kernel writes byte-identical host pages through a 6-page buffer and through the 64-page one."""
import ast
import hashlib
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import faketorch as torch  # noqa: E402
import apply_hicache_staging_cap as overlay  # noqa: E402

HICACHE_PIN = '322981f1354d6449a0c6e2bf568ce9b487a99f196ffadf6dbfa4e1304e6b0d85'
CHECKS = []
# (pool, layers, pages, item_bytes) as the served launcher.log printed them on 3 Nov
POOLS = [('swa', 40, 256, 135168), ('deepseek_v4_c1', 1, 4096, 73728), ('deepseek_v4_c1_indexer', 1, 4096, 17408),
         ('deepseek_v4_c2', 3, 4096, 36864), ('deepseek_v4_c2_indexer', 3, 4096, 8704)]


def check(cond, what):
    CHECKS.append(what)
    assert cond, what


def extract(src, names, ns):
    tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), 'extract', 'exec'), ns)
    return ns


def methods(src, cls_name):
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == '_init_write_back_staging_buffers']


class Env(dict):
    pass


def capacity_fn(src, env):
    ns = {'_WRITE_BACK_STAGING_PAGE_CHUNK': 64}
    extract(src, {'_staging_page_capacity'}, ns)
    fn = ns['_staging_page_capacity']
    import os as real_os

    def call(pages, page_bytes):
        saved = real_os.environ.pop('SGLANG_HICACHE_STAGING_MB', None)
        real_os.environ.update(env)
        try:
            return fn(pages, page_bytes)
        finally:
            real_os.environ.pop('SGLANG_HICACHE_STAGING_MB', None)
            if saved is not None:
                real_os.environ['SGLANG_HICACHE_STAGING_MB'] = saved
    return call


def test_capacity(patched_src):
    off = capacity_fn(patched_src, {})
    grid = [(p, b) for p in (0, 1, 5, 63, 64, 65, 256, 4096) for b in (1, 8704, 135168 * 40, 1 << 30)]
    check(all(off(p, b) == min(p, 64) for p, b in grid), 'unset: min(pages, 64) for every input')
    check(capacity_fn(patched_src, {'SGLANG_HICACHE_STAGING_MB': ' '})(256, 1) == 64, 'blank: unset')
    cap = capacity_fn(patched_src, {'SGLANG_HICACHE_STAGING_MB': '32'})
    got = {name: cap(pages, layers * item) for name, layers, pages, item in POOLS}
    check(got == {'swa': 6, 'deepseek_v4_c1': 64, 'deepseek_v4_c1_indexer': 64, 'deepseek_v4_c2': 64,
                  'deepseek_v4_c2_indexer': 64}, f'32 MiB: swa 6 pages, the rest 64 ({got})')
    before = sum(min(p, 64) * l * i for _, l, p, i in POOLS)
    after = sum(cap(p, l * i) * l * i for _, l, p, i in POOLS)
    check((before - after) == 313_589_760, f'32 MiB frees {(before - after) / 2**20:.1f} MiB of VRAM')
    check(all(cap(p, b) >= 1 and cap(p, b) * b <= max(32 << 20, b) for p, b in grid if p), '32 MiB: at least one page, never over the cap unless one page is bigger')
    for bad in ('0', '-3', 'x', '1.5'):
        try:
            capacity_fn(patched_src, {'SGLANG_HICACHE_STAGING_MB': bad})(256, 4096)
        except ValueError:
            check(True, f'{bad!r} refused')
        else:
            raise AssertionError(f'{bad!r} accepted')


def init_buffers(src, cls_name, env, layers, pages, item):
    """Run one pool class's _init_write_back_staging_buffers on a stand-in self; return the staging shape."""
    shapes = []
    fake_torch = types.SimpleNamespace(empty=lambda shape, dtype=None, device=None: shapes.append(tuple(shape)) or object())
    ns = {'torch': fake_torch, '_is_cuda': True, '_WRITE_BACK_STAGING_PAGE_CHUNK': 64,
          'can_use_write_back_jit_kernel': lambda element_size: True,
          'current_platform': types.SimpleNamespace(capabilities=types.SimpleNamespace(hicache_device_kernels=True))}
    if '_staging_page_capacity' in src:
        extract(src, {'_staging_page_capacity'}, ns)
    body = methods(src, cls_name)
    exec(compile(ast.Module(body=body, type_ignores=[]), cls_name, 'exec'), ns)
    self = types.SimpleNamespace(layout='page_first', num_host_pages=pages, layer_num=layers, item_bytes=item,
                                 state_page_bytes=item, dtype=types.SimpleNamespace(itemsize=1), gpu_device='cuda')
    import os as real_os
    saved = real_os.environ.pop('SGLANG_HICACHE_STAGING_MB', None)
    real_os.environ.update(env)
    try:
        ns['_init_write_back_staging_buffers'](self)
    finally:
        real_os.environ.pop('SGLANG_HICACHE_STAGING_MB', None)
        if saved is not None:
            real_os.environ['SGLANG_HICACHE_STAGING_MB'] = saved
    return shapes[0], self.can_use_write_back_jit


def test_pools(served_src, patched_src):
    for cls in ('DeepSeekV4PagedHostPool', 'DeepSeekV4StateHostPool'):
        for name, layers, pages, item in POOLS:
            a = init_buffers(served_src, cls, {}, layers, pages, item)
            b = init_buffers(patched_src, cls, {}, layers, pages, item)
            check(a == b, f'{cls} {name}: unset allocates the served buffer {a[0]}')
            c = init_buffers(patched_src, cls, {'SGLANG_HICACHE_STAGING_MB': '32'}, layers, pages, item)
            want = (6 if name == 'swa' else 64, layers, item)
            check(c[0] == want, f'{cls} {name}: 32 MiB allocates {want}')


def test_staged_copy(hicache_src):
    """The served staged write-back loop through a 64-page and a 6-page buffer: same host bytes, more launches."""
    launches = []

    def launch(dst, dst_indices, staging, src_pages, ptr_src, page_size):
        launches.append(staging.shape[0])
        assert staging.shape[0] >= src_pages.numel()
        for i, page in enumerate(src_pages.a):
            for layer, buf in enumerate(ptr_src):
                staging.a[i, layer] = buf[page]          # device gather into the staging rows
        dst.a[dst_indices.a] = staging.a[: src_pages.numel()]   # one copy of the staged pages to host memory

    ns = {'DEFAULT_BLOCK_QUOTA': 1, '_default_unroll': lambda element_size: 1, 'torch': torch,
          'debug_kernel_api': lambda f: f,   # the served logging decorator (kernel_api_logging): passes the call through
          '_jit_hicache_staged_module': lambda **kw: types.SimpleNamespace(launch_all_mla_lf_pf_staged=launch)}
    extract(hicache_src, {'transfer_hicache_all_layer_mla_staged_lf_pf'}, ns)
    copy = ns['transfer_hicache_all_layer_mla_staged_lf_pf']
    rng = np.random.default_rng(3)
    layers, item, dev_pages, host_pages = 5, 48, 300, 256
    device_layers = [rng.integers(0, 256, (dev_pages, item), dtype=np.uint8) for _ in range(layers)]
    for n in (1, 6, 7, 50, 64, 65, 200):
        src = torch.Tensor(rng.choice(dev_pages, n, replace=False).astype(np.int64), torch.int64)
        dst_idx = torch.Tensor(rng.choice(host_pages, n, replace=False).astype(np.int64), torch.int64)
        outs = []
        for cap in (64, 6):
            launches.clear()
            dst = torch.Tensor(np.zeros((host_pages, layers, item), dtype=np.uint8), torch.uint8)
            staging = torch.Tensor(np.zeros((cap, layers, item), dtype=np.uint8), torch.uint8)
            copy(device_layers, src, dst_idx, staging, dst, page_size=1, element_size=item)
            check(len(launches) == math.ceil(n / cap), f'{n} pages through {cap}: {math.ceil(n / cap)} launches')
            outs.append(dst.a.copy())
        want = np.zeros_like(outs[0])
        for s, d in zip(src.a, dst_idx.a):
            want[d] = np.stack([device_layers[l][s] for l in range(layers)])
        check(np.array_equal(outs[0], outs[1]) and np.array_equal(outs[1], want), f'{n} pages: host bytes identical, 64 vs 6')


def main(root):
    root = Path(root)
    mph, hic = root / 'srt/mem_cache/memory_pool_host.py', root / 'kernels/ops/kvcache/hicache.py'
    assert hashlib.sha256(mph.read_bytes()).hexdigest() == overlay.PIN
    assert hashlib.sha256(hic.read_bytes()).hexdigest() == HICACHE_PIN
    tmp = Path(tempfile.mkdtemp(prefix='upstream-picks-r1-'))
    try:
        (tmp / 'srt/mem_cache').mkdir(parents=True)
        shutil.copyfile(mph, tmp / 'srt/mem_cache/memory_pool_host.py')
        overlay.main(tmp)
        served_src = mph.read_text(encoding='utf-8')
        patched_src = (tmp / 'srt/mem_cache/memory_pool_host.py').read_text(encoding='utf-8')
        test_capacity(patched_src)
        test_pools(served_src, patched_src)
        test_staged_copy(hic.read_text(encoding='utf-8'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'test_hicache_staging_cap: {len(CHECKS)} checks passed')


if __name__ == '__main__':
    main(sys.argv[1])
