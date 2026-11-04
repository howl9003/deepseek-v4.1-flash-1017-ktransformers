"""CPU-only tests for apply_swa_hicache_lend.py (upstream-picks-r2).

argv: the served sglang package directory (srt/layers/moe/kt_kv_prefill_lend.py at the served pin). The test copies
the file, applies the overlay to the copy, and runs the real SWA-lend code of both files (the allocator wrapper,
swa_map, swa_flush, bind_tree, swa_trim) on stand-ins for torch, kv_vmm, the allocator and the radix tree:
  1. the load-back race: an SWA allocation that takes lent pages in decode defers their mapping to the next forward;
     a HiCache copy started before it (start_loading) writes into unmapped pages. Served, and patched with the switch
     unset, the copy hits unmapped pages, as today; with KT_PREFILL_SWA_LOAD_MAP=1 the pending pages are mapped
     before the copy. Wrapped once, only with the SWA lend on and a cache controller present;
  2. the trim: unset, the trim works to the pool's budget exactly as served; with KT_PREFILL_SWA_HOST_TRIM=1 and a
     cache controller it works to KT_PREFILL_SWA_LEND_KEEP; without a controller it keeps the budget."""
import ast
import hashlib
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import faketorch as torch  # noqa: E402
import apply_swa_hicache_lend as overlay  # noqa: E402

CHECKS = []
NAMES = {'CHECK', 'SWA_ENABLED', 'SWA_KEEP', 'SWA_SLACK', 'SWA', 'SWA_LOAD_MAP', 'SWA_HOST_TRIM', 'SWA_CLEAR_RELEASE_HOT'}
FUNCS = {'_log', '_free_count', '_swa_free', '_swa_in_use', 'bind_tree', '_wrap_hicache_load', 'swa_trim', 'swa_map',
         'swa_flush', '_defer_ok', '_extend_pages', '_wrap_swa', 'swa_restore_all', 'hot_release'}


def check(cond, what):
    CHECKS.append(what)
    assert cond, what


class VMM:
    """kv_vmm: which bytes of each lent tensor are mapped."""
    ENABLED = True

    def __init__(self):
        self.mapped, self.calls = {}, []

    def track(self, flat, upto):
        self.mapped[flat.data_ptr()] = np.zeros(flat.numel(), dtype=bool)
        self.mapped[flat.data_ptr()][:upto] = True

    def restore(self, flat, off, ln):
        m = self.mapped[flat.data_ptr()]
        new = int((~m[off:off + ln]).sum())
        m[off:off + ln] = True
        self.calls.append(('restore', off, ln))
        return new

    def release(self, flat, off, ln):
        self.mapped[flat.data_ptr()][off:off + ln] = False
        return ln

    def is_mapped(self, flat, lo, hi):
        return bool(self.mapped[flat.data_ptr()][lo:hi].all())


def stubs(phase='decode'):
    for n in ('sglang', 'sglang.srt', 'sglang.srt.layers', 'sglang.srt.mem_cache'):
        sys.modules[n] = types.ModuleType(n)
    moe = types.ModuleType('sglang.srt.layers.moe')
    moe.kt_lend = types.SimpleNamespace(ENABLED=True, STATE={'phase': phase})
    sys.modules['sglang.srt.layers.moe'] = moe
    sys.modules['sglang.srt.layers.moe.kt_lend'] = moe.kt_lend
    sys.modules['sglang.srt.mem_cache.base_prefix_cache'] = types.SimpleNamespace(
        EvictParams=lambda swa_num_tokens: types.SimpleNamespace(swa_num_tokens=swa_num_tokens))
    sys.modules['torch'] = torch


def lend_module(src, env):
    """The file's SWA-lend statements and functions, run under `env` with the stand-ins."""
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in NAMES for t in node.targets):
            keep.append(node)
    vmm, log = VMM(), []
    ns = {'os': types.SimpleNamespace(environ=dict(env)), 'torch': torch, 'time': time, 'kt_kv_vmm': vmm,
          'STATE': {'lent': None}, 'restore': lambda why: None}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'kt_kv_prefill_lend(extract)', 'exec'), ns)
    ns['_log'] = lambda msg: log.append(msg)
    return ns, vmm, log


class Allocator:
    """A paged SWA allocator: one free list of page ids, handed out from the front."""

    def __init__(self, ps, free):
        self.page_size, self.release_pages = ps, None
        self.free_pages = torch.Tensor(np.array(free, dtype=np.int64), torch.int64, 'cuda')

    def alloc(self, need_size):
        pages = -(-need_size // self.page_size)
        if pages > len(self.free_pages):
            return None
        out, self.free_pages = self.free_pages[:pages], self.free_pages[pages:]
        return out

    def available_size(self):
        return len(self.free_pages) * self.page_size


def race(src, env):
    """bind_tree (the first prefill plan), then a load-back's SWA allocation in decode, then its copy."""
    torch.reset()
    stubs('decode')
    ns, vmm, log = lend_module(src, env)
    n, ps, pb, C = 12, 256, 64, 128
    flat = torch.zeros(((n + 1) * pb,), dtype=torch.uint8, device='cuda')
    vmm.track(flat, 4 * pb)                              # pages 0..3 mapped, 4..11 lent
    a = Allocator(ps, [1, 2, 3])
    ns['SWA'].update(alloc=a, tensors=[(flat, pb, C, 0)], ps=ps, n=n, off=False, row=pb / ps,
                     lent=torch.Tensor(np.arange(4, 12, dtype=np.int64), torch.int64, 'cuda'))
    ns['_wrap_swa'](a)
    events = []

    class Controller:
        def __init__(self):
            self.queue = []

        def start_loading(self):
            # the host-to-device copy, on its own stream, of every page queued for load-back
            events.append('copy')
            bad = [int(p) for p in self.queue if not vmm.is_mapped(flat, int(p) * pb, (int(p) + 1) * pb)]
            self.queue = []
            if bad:
                raise RuntimeError(f'illegal memory access: copy into unmapped SWA pages {bad}')
            return 0

    tree = types.SimpleNamespace(cache_controller=Controller())
    scheduler = types.SimpleNamespace(tree_cache=tree)
    ns['bind_tree'](scheduler)
    ns['bind_tree'](scheduler)                           # every chunk_for call binds again
    pages = a.alloc(5 * ps)                              # the load-back takes five pages: two are lent
    tree.cache_controller.queue = list(pages.a)
    orig_restore = vmm.restore
    vmm.restore = lambda *x: (events.append('map'), orig_restore(*x))[1]
    try:
        tree.cache_controller.start_loading()
        fault = None
    except RuntimeError as e:
        fault = str(e)
    return dict(ns=ns, fault=fault, events=events, pages=list(pages.a), log=log, cc=tree.cache_controller)


def test_race(served_src, patched_src):
    s = race(served_src, {'KT_PREFILL_SWA_LEND': '1'})
    check(s['pages'] == [1, 2, 3, 4, 5] and s['fault'] and '4, 5' in s['fault'],
          'served: a load-back copy started before the next forward writes into the unmapped lent pages 4, 5')
    u = race(patched_src, {'KT_PREFILL_SWA_LEND': '1'})
    check(u['fault'] == s['fault'] and u['events'] == s['events'] == ['copy'], 'switch unset: exactly as served')
    check(not getattr(u['cc'], '_kt_swa_load', False), 'switch unset: start_loading not wrapped')
    p = race(patched_src, {'KT_PREFILL_SWA_LEND': '1', 'KT_PREFILL_SWA_LOAD_MAP': '1'})
    check(p['fault'] is None, 'KT_PREFILL_SWA_LOAD_MAP=1: the copy finds every page mapped')
    check(p['events'] == ['map', 'copy'], 'KT_PREFILL_SWA_LOAD_MAP=1: the pending maps run before the copy')
    check(not any(p['ns']['SWA']['pending']), 'KT_PREFILL_SWA_LOAD_MAP=1: nothing left pending after the load')
    check(sum('start_loading maps the SWA pages' in m for m in p['log']) == 1 and p['cc']._kt_swa_load,
          'KT_PREFILL_SWA_LOAD_MAP=1: wrapped once for repeated binds')
    off = race(patched_src, {'KT_PREFILL_SWA_LEND': '0', 'KT_PREFILL_SWA_LOAD_MAP': '1'})
    check('tree' not in off['ns']['SWA'] and not getattr(off['cc'], '_kt_swa_load', False),
          'SWA lend off: nothing bound or wrapped even with the switch')
    # no HiCache: a tree without a controller is left alone
    torch.reset()
    stubs()
    ns, _, _ = lend_module(patched_src, {'KT_PREFILL_SWA_LEND': '1', 'KT_PREFILL_SWA_LOAD_MAP': '1'})
    tree = types.SimpleNamespace()
    ns['bind_tree'](types.SimpleNamespace(tree_cache=tree))
    check(ns['SWA']['tree'] is tree and not hasattr(tree, 'start_loading'), 'no cache controller: bound, nothing wrapped')


def trim(src, env, in_use_pages, controller):
    torch.reset()
    stubs()
    ns, _, log = lend_module(src, env)
    n, ps = 100, 256
    lent = 20
    free = n - lent - in_use_pages
    a = Allocator(ps, list(range(1, free + 1)))
    calls = []
    tree = types.SimpleNamespace(evict=lambda p: calls.append(p.swa_num_tokens) or types.SimpleNamespace(
        swa_num_tokens_evicted=p.swa_num_tokens))
    if controller:
        tree.cache_controller = object()
    ns['SWA'].update(alloc=a, ps=ps, n=n, tree=tree, budget=11520,
                     lent=torch.Tensor(np.arange(200, 200 + lent, dtype=np.int64), torch.int64, 'cuda'))
    ns['swa_trim']('end of window')
    return calls


def test_trim(served_src, patched_src):
    for pages, controller in ((17, True), (30, True), (30, False), (45, True), (60, True), (60, False)):
        a = trim(served_src, {'KT_PREFILL_SWA_LEND': '1'}, pages, controller)
        b = trim(patched_src, {'KT_PREFILL_SWA_LEND': '1'}, pages, controller)
        want = [pages * 256 - 11520] if pages * 256 - 11520 > 4 * 256 else []
        check(a == b == want, f'unset: {pages} pages in use, controller {controller}: trims {want} as served')
        c = trim(patched_src, {'KT_PREFILL_SWA_LEND': '1', 'KT_PREFILL_SWA_HOST_TRIM': '1'}, pages, controller)
        budget = 4096 if controller else 11520
        want = [pages * 256 - budget] if pages * 256 - budget > 4 * 256 else []
        check(c == want, f'KT_PREFILL_SWA_HOST_TRIM=1: {pages} pages, controller {controller}: trims {want}')


def main(root):
    root = Path(root)
    src = root / overlay.REL
    assert hashlib.sha256(src.read_bytes()).hexdigest() == overlay.PIN
    tmp = Path(tempfile.mkdtemp(prefix='upstream-picks-r2-'))
    try:
        (tmp / overlay.REL).parent.mkdir(parents=True)
        shutil.copyfile(src, tmp / overlay.REL)
        overlay.main(tmp)
        served_src = src.read_text(encoding='utf-8')
        patched_src = (tmp / overlay.REL).read_text(encoding='utf-8')
        test_race(served_src, patched_src)
        test_trim(served_src, patched_src)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'test_swa_hicache_lend: {len(CHECKS)} checks passed')


if __name__ == '__main__':
    main(sys.argv[1])
