"""CPU-only tests for apply_stream_cpu_short.py (upstream-picks-r1).

argv: a directory holding the served sglang package files (srt/layers/moe/kt_stream_prefill.py,
srt/layers/moe/contract_v2/adapter.py, srt/layers/moe/contract_v2/stream_graph_candidate.py at the served pins).
The test copies them, applies the overlay to the copy, and checks:
  1. the share parsing and stream_cpu_cutoff: with KT_STREAM_CPU_SHORT unset the patched cutoff equals the served one
     for every row count and KT_STREAM_CPU_GROUPS value; with it set, the split and its groups partition the experts;
  2. the real contract v2 adapter and graph stream, served and patched, on numpy stand-ins for torch, kt-kernel and
     the GPU kernels (one per-route function for every owner, as the contract requires): unset is call-for-call and
     bit-for-bit the served path; set, the combined output is bit-identical to the unset output, every route is owned
     once, the CPU's rows are padded to a bucket, and no buffer is made per prompt length."""
import ast
import hashlib
import importlib.util
import os
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
import apply_stream_cpu_short as overlay  # noqa: E402

FILES = list(overlay.PINS)
E, H, K, NRES = 384, 8, 6, 1          # experts, hidden (small), top-k, resident GPU experts
THRESHOLD = 512                       # KT_GPU_STREAM_PREFILL on the served profile
CHECKS = []


def check(cond, what):
    CHECKS.append(what)
    assert cond, what


# ------------------------------------------------------------------ 1. the share parsing and the cutoff
def cutoff_namespace(src, env):
    """Run kt_stream_prefill's CPU-share statements (CPU_GROUPS, the KT_STREAM_CPU_SHORT parse and guard, the helpers,
    stream_cpu_cutoff) from the file's own source, under `env`."""
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                '_parse_cpu_short', 'cpu_short_entry', 'cpu_groups_for', 'stream_cpu_cutoff'):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in ('CPU_GROUPS', 'CPU_SHORT') for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.If) and 'KT_STREAM_CPU_SHORT' in ast.unparse(node):
            keep.append(node)
    ns = {'os': types.SimpleNamespace(environ=dict(env))}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'kt_stream_prefill(extract)', 'exec'), ns)
    return ns


def test_cutoff(served_src, patched_src):
    method = types.SimpleNamespace(num_gpu_experts=NRES, global_num_experts=E)
    st = types.SimpleNamespace(G=4)
    rows = list(range(1, 2100)) + [4095, 4096, 4097, 36864]
    for groups in ('', '0', '1', '8', '48', '95', '96', '200'):
        env = {'KT_CONTRACT_V2': '1'}
        if groups:
            env['KT_STREAM_CPU_GROUPS'] = groups
        a, b = cutoff_namespace(served_src, env), cutoff_namespace(patched_src, env)
        check(b['CPU_SHORT'] == [], 'unset parses to no entry')
        want = a['stream_cpu_cutoff'](method, st)
        check(b['stream_cpu_cutoff'](method, st) == want, f'tokens=None equals served (groups {groups!r})')
        check(all(b['stream_cpu_cutoff'](method, st, t) == want for t in rows),
              f'every row count equals served with the share unset (groups {groups!r})')

    def ns(spec, groups=None, contract='1'):
        env = {'KT_STREAM_CPU_SHORT': spec, 'KT_CONTRACT_V2': contract}
        if groups is not None:
            env['KT_STREAM_CPU_GROUPS'] = groups
        return cutoff_namespace(patched_src, env)

    n = ns(' 2048:24 , 1024:48,')
    check(n['CPU_SHORT'] == [(1024, 48), (2048, 24)], 'entries sorted, blanks and spaces ignored')
    check(n['cpu_groups_for'](600) == 48 and n['cpu_groups_for'](1024) == 48, 'first entry that fits (<= bound)')
    check(n['cpu_groups_for'](1025) == 24 and n['cpu_groups_for'](2048) == 24, 'second entry')
    check(n['cpu_groups_for'](2049) == 0 and n['cpu_groups_for'](None) == 0, 'past the bounds: KT_STREAM_CPU_GROUPS (0)')
    check(ns('1024:48', groups='8')['cpu_groups_for'](4096) == 8, 'past the bounds: KT_STREAM_CPU_GROUPS (8)')
    check(ns('1024:0', groups='8')['cpu_groups_for'](600) == 0, 'an entry of k=0 turns the share off for its rows')
    for bad in ('1024', '1024:x', '0:4', '1024:-1', '1024:4,1024:8'):
        try:
            ns(bad)
        except ValueError:
            check(True, f'malformed {bad!r} refused')
        else:
            raise AssertionError(f'malformed {bad!r} accepted')
    try:
        ns('1024:48', contract='0')
    except RuntimeError:
        check(True, 'refused without KT_CONTRACT_V2=1')
    else:
        raise AssertionError('KT_STREAM_CPU_SHORT accepted without contract v2')

    # the split: resident [0, first), streamed groups [first, cutoff) in G, CPU [cutoff, E): each expert once
    for k in (1, 24, 48, 64, 72, 95, 96, 500):
        f = ns(f'1024:{k}')['stream_cpu_cutoff']
        cut = f(method, st, 1000)
        groups = [list(range(b, min(b + st.G, cut))) for b in range(NRES, cut, st.G)]
        streamed = [e for g in groups for e in g]
        owner = list(range(NRES)) + streamed + list(range(cut, E))
        check(sorted(owner) == list(range(E)), f'k={k}: every expert owned once')
        check(len(groups) == max(1, 96 - k) and all(len(g) == 4 for g in groups), f'k={k}: {max(1, 96 - k)} full groups streamed')
        check(f(method, st, 1025) == E, f'k={k}: 1,025 rows stream everything')
    check(ns('1024:48')['stream_cpu_cutoff'](method, st, 600) == 193, '1024:48 streams physical 1..192, CPU takes 193..383')


# ------------------------------------------------------------------ 2. the adapter and graph stream on stand-ins
def bf16(a):
    """Round float32 to BF16 (nearest even) and back: the per-route rows are BF16."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def route_value(xrow, e, w):
    """The contract: one value per (token, expert, weight), whoever computes it."""
    return bf16(np.float32(w) * xrow.astype(np.float32) * np.float32(1.0 + e / 997.0))


class Log(list):
    pass


LOG = Log()


def make_fakes():
    pkg = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[name] = m
        pkg[name] = m
        return m

    sys.modules['torch'] = torch

    class KExpertsCPUBuffer:
        buffer_depth = 2
        capture_buffers = {}

        @classmethod
        def get_buffer(cls, x, k):
            T, hidden = x.shape
            if T not in cls.capture_buffers:
                LOG.append(('kt_buffer', T))
                mk = lambda shape, dt: [torch.zeros(shape, dtype=dt, pin_memory=True) for _ in range(cls.buffer_depth)]
                cls.capture_buffers[T] = (mk((T, hidden), torch.bfloat16), mk((T, k), torch.int64), mk((T, k), torch.int64),
                                          mk((T, k), torch.float32), mk((T, hidden), torch.bfloat16),
                                          [torch.full((1,), T, dtype=torch.int32) for _ in range(cls.buffer_depth)],
                                          [torch.zeros((T, hidden), dtype=torch.bfloat16, device='cuda') for _ in range(cls.buffer_depth)])
            return cls.capture_buffers[T]

    class BaseMoEWrapper:
        _layer_has_pending_deferred = {}

    mod('kt_kernel')
    mod('kt_kernel.experts_base', KExpertsCPUBuffer=KExpertsCPUBuffer, BaseMoEWrapper=BaseMoEWrapper)
    for n in ('sglang', 'sglang.srt', 'sglang.srt.layers'):
        mod(n)
    moe_pkg = mod('sglang.srt.layers.moe')
    mod('sglang.srt.layers.moe.token_dispatcher',
        StandardCombineInput=lambda hidden_states: types.SimpleNamespace(hidden_states=hidden_states))
    hot = mod('sglang.srt.layers.moe.kt_kv_hot_lend', route=lambda wrapper, dispatch: None)

    def pack_to_host(x, w, ids, hx, hw, hids):
        LOG.append(('pack', x.shape[0]))
        hx.copy_(x), hw.copy_(w), hids.copy_(ids)
        return True

    ep = mod('sglang.srt.layers.moe.kt_ep_wrapper', _KT_HANDOFF_PACK=False, _KT_SHARED_AFTER_SUBMIT=True,
             pack_to_host=pack_to_host)
    moe_pkg.kt_kv_hot_lend, moe_pkg.kt_ep_wrapper = hot, ep
    mod('sglang.srt.layers.moe.contract_v2')

    def expert_routes(x, w13, s13, w2, s2, ids, weights, out=None):
        """The resident rows: physical ids 0..NRES-1 are the local rows."""
        LOG.append(('resident', x.shape[0]))
        ia, wa = ids.a, weights.a
        for t, r in zip(*np.nonzero(ia >= 0)):
            out.a[t, r] = route_value(x.a[t], ia[t, r], wa[t, r])

    order = np.random.default_rng(5).permutation(E)

    def combine_routes(routes, ids, order=None):
        """FP32 sum of each token's routes in ascending logical expert id."""
        ra, ia = routes.a, ids.a
        out = np.zeros((ra.shape[0], ra.shape[2]), dtype=np.float32)
        for t in range(ra.shape[0]):
            live = [r for r in range(ra.shape[1]) if ia[t, r] >= 0]
            for r in sorted(live, key=lambda r: order[ia[t, r]]):
                out[t] = np.float32(out[t] + ra[t, r])
        return torch.Tensor(out, torch.float32, 'cuda')

    mod('sglang.srt.layers.moe.contract_v2.moe', expert_routes=expert_routes, combine_routes=combine_routes)
    mod('sglang.srt.layers.moe.contract_v2.placement_order', order_for=lambda wrapper, device: order)
    return moe_pkg


def source_module(src, env, streamer):
    """kt_stream_prefill as the adapter sees it: the real share code from `src`, the stream switches of our profile."""
    ns = cutoff_namespace(src, env)
    m = types.ModuleType('sglang.srt.layers.moe.kt_stream_prefill')
    m.__dict__.update({k: v for k, v in ns.items() if not k.startswith('__')})
    m.ZEROCOPY = m.W16 = m.USE_GRAPH = m.CONTRACT_STREAM = True
    m.SHARED_INPUTS = False
    m._block_rows = lambda T: min(T, 4096)
    m.stream_prefill_for = lambda wrapper, layer, rows: streamer if rows >= THRESHOLD else None
    return m


class Slot:
    def __init__(self):
        self.dma_done, self.compute_done, self.compute_used = torch.Event(), torch.Event(), False


class Streamer:
    G, world = 4, 1

    def __init__(self):
        self._gin, self.slots, self.calls = {}, [Slot(), Slot()], 0

    def _stage_zc(self, lid, gi, experts, stats):
        LOG.append(('stage', gi, experts[0], len(experts)))

    def _graph_for(self, layer, T, gi, base, n, g):
        def replay():
            LOG.append(('replay', T, gi, base, n))
            ia, wa, xa, acc = g['ids'].a, g['w'].a, g['xq'].a, g['acc'].a
            for t, r in zip(*np.nonzero((ia >= base) & (ia < base + n))):
                acc[t, r] = route_value(xa[t], ia[t, r], wa[t, r])
        return types.SimpleNamespace(replay=replay)


class CPU:
    """kt-kernel's per-route API: rows of id -1 are skipped and left as garbage, the rest get the contract value."""

    def __init__(self):
        self.pending = []

    def forward_routes_task(self, bsz_ptr, k, ids_ptr, w_ptr, x_ptr, out_ptr):
        def run():
            bsz = int(torch.lookup(bsz_ptr).a[0])
            ids, w, x, out = (torch.lookup(p).a for p in (ids_ptr, w_ptr, x_ptr, out_ptr))
            assert out.shape[0] >= bsz and ids.shape[0] == bsz and out.shape[1:] == (k, x.shape[1])
            LOG.append(('cpu', bsz, int((ids >= 0).sum())))
            for t in range(bsz):
                for r in range(k):
                    out[t, r] = route_value(x[t], ids[t, r], w[t, r]) if ids[t, r] >= 0 else np.nan
        return run

    def submit_with_cuda_stream(self, stream, task):
        self.pending.append(task)

    def sync_with_cuda_stream(self, stream, flag):
        while self.pending:
            self.pending.pop(0)()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def variant(root, env, src):
    """Load one tree's adapter and graph stream under the canonical names, with kt_stream_prefill from `src`."""
    moe_pkg = sys.modules['sglang.srt.layers.moe']
    streamer = Streamer()
    moe_pkg.kt_stream_prefill = sys.modules['sglang.srt.layers.moe.kt_stream_prefill'] = source_module(src, env, streamer)
    sys.modules['kt_kernel.experts_base'].KExpertsCPUBuffer.capture_buffers = {}
    load('sglang.srt.layers.moe.contract_v2.stream_graph_candidate', root / 'srt/layers/moe/contract_v2/stream_graph_candidate.py')
    adapter = load('sglang.srt.layers.moe.contract_v2.adapter', root / 'srt/layers/moe/contract_v2/adapter.py')
    return adapter


def inputs(T, seed):
    rng = np.random.default_rng(seed)
    x = torch.Tensor(rng.standard_normal((T, H)).astype(np.float32), torch.bfloat16, 'cuda')
    x.a[:] = bf16(x.a)
    ids = np.stack([rng.choice(E, K, replace=False) for _ in range(T)]).astype(np.int32)
    w = rng.random((T, K)).astype(np.float32)
    w[rng.random((T, K)) < 0.03] = 0.0       # zero-weight routes: owned by nobody
    ids[0, :] = [0, 1, 200, 383, 5, 192]     # the resident expert, both sides of the cutoff
    ids[1, 5], w[1, 5] = -1, 0.0             # a dropped route
    return x, torch.Tensor(ids, torch.int32, 'cuda'), torch.Tensor(w, torch.float32, 'cuda')


def run(adapter, T, seed, layer_idx=3):
    x, ids, w = inputs(T, seed)
    native = types.SimpleNamespace(layer_idx=layer_idx, max_deferred_experts_per_token=0, num_experts_per_tok=K,
                                   chunked_prefill_size=37376, moe=CPU())
    native.cpu_infer = native.moe
    wrapper = types.SimpleNamespace(tp_rank=0, num_gpu_experts=NRES, global_num_experts=E, wrapper=native,
                                    kt_config=types.SimpleNamespace(layer_idx=layer_idx))
    dummy = torch.zeros((1,), dtype=torch.uint8, device='cuda')
    layer = types.SimpleNamespace(moe_tp_size=1, moe_ep_size=1, w13_weight=dummy, w13_weight_scale_inv=dummy,
                                  w2_weight=dummy, w2_weight_scale_inv=dummy)
    dispatch = types.SimpleNamespace(hidden_states=x, topk_output=types.SimpleNamespace(topk_ids=ids, topk_weights=w))
    LOG.clear()
    out = adapter.apply(wrapper, layer, dispatch).hidden_states.a.copy()
    return out, list(LOG), wrapper


def test_adapter(served, patched):
    make_fakes()
    served_src = (served / FILES[0]).read_text(encoding='utf-8')
    patched_src = (patched / FILES[0]).read_text(encoding='utf-8')
    rows = (300, 512, 513, 600, 777, 1000, 1024, 1025, 1500, 2000, 2500, 4000)

    def sweep(root, src, env, rows=rows, seed=11):
        torch.reset()
        adapter = variant(root, env, src)
        res = {T: run(adapter, T, seed + T) for T in rows}
        return adapter, res

    base_env = {'KT_CONTRACT_V2': '1'}
    _, a = sweep(served, served_src, base_env)
    _, b = sweep(patched, patched_src, base_env)
    for T in rows:
        check(np.array_equal(a[T][0], b[T][0]) and np.isfinite(b[T][0]).all(), f'unset: {T} rows bit-identical to served')
        check(a[T][1] == b[T][1], f'unset: {T} rows make the same calls as served')
    ref = {T: a[T][0] for T in rows}
    check(not any(e[0] == 'cpu' for T in rows if T >= THRESHOLD for e in b[T][1]), 'unset: no CPU share in a streamed forward')
    check(sum(1 for e in b[1024][1] if e[0] == 'replay') == 96, 'unset: 1,024 rows stream all 96 groups')

    # the static share (KT_STREAM_CPU_GROUPS) is served code today: the patched tree must not change it
    env = dict(base_env, KT_STREAM_CPU_GROUPS='8')
    _, a8 = sweep(served, served_src, env)
    _, b8 = sweep(patched, patched_src, env)
    for T in rows:
        check(np.array_equal(a8[T][0], b8[T][0]) and a8[T][1] == b8[T][1], f'KT_STREAM_CPU_GROUPS=8 unchanged at {T} rows')
        check(np.array_equal(b8[T][0], ref[T]), f'KT_STREAM_CPU_GROUPS=8 bit-identical to no share at {T} rows')

    for spec, k in (('1024:48', 48), ('1024:72', 72), ('1024:96', 96), ('1024:500', 500)):
        env = dict(base_env, KT_STREAM_CPU_SHORT=spec)
        adapter, s = sweep(patched, patched_src, env)
        kept = max(1, 96 - k)
        cut = NRES + 4 * kept
        for T in rows:
            out, log, _ = s[T]
            check(np.array_equal(out, ref[T]), f'{spec}: {T} rows bit-identical to no share')
            cpu = [e for e in log if e[0] == 'cpu']
            replays = [e for e in log if e[0] == 'replay']
            if THRESHOLD <= T <= 1024:
                bucket = min(1024, 1 << (T - 1).bit_length())
                x, ids, w = inputs(T, 11 + T)
                check(cpu == [('cpu', bucket, int((ids.a >= cut).sum()))], f'{spec}: {T} rows: CPU padded to {bucket}, real routes only')
                check(len(replays) == kept and replays[-1][3] == cut - 4, f'{spec}: {T} rows stream {kept} groups')
            elif T > 1024:
                check(not cpu and len(replays) == 96, f'{spec}: {T} rows stream everything')
        bufs = sys.modules['kt_kernel.experts_base'].KExpertsCPUBuffer.capture_buffers
        check(set(b for b in bufs if b >= THRESHOLD) == {512, 1024}, f'{spec}: kt-kernel staging only for buckets 512 and 1024')
        check(not [key for key in adapter._buffers if key[1] >= THRESHOLD], f'{spec}: no per-length route buffers')
        check(sorted(key[0] for key in adapter._short_hosts) == [512, 1024], f'{spec}: one pinned set a bucket (layer slot 1)')
        check(all(v.device.type == 'cpu' and v.pinned for v in adapter._short_hosts.values()),
              f'{spec}: the kept share buffers are pinned host memory, no VRAM')

    # two entries over the static share, the packer path, both layer slots
    env = dict(base_env, KT_STREAM_CPU_SHORT='1024:48,2048:24', KT_STREAM_CPU_GROUPS='8')
    sys.modules['sglang.srt.layers.moe.kt_ep_wrapper']._KT_HANDOFF_PACK = True
    torch.reset()
    adapter = variant(patched, env, patched_src)
    for T, lid in ((600, 3), (600, 4), (1500, 3), (2000, 4), (2500, 3), (4000, 4)):
        out, log, wrapper = run(adapter, T, 11 + T, layer_idx=lid)
        check(np.array_equal(out, ref[T]), f'two entries, packer on: {T} rows (slot {lid % 2}) bit-identical to no share')
        cpu = [e for e in log if e[0] == 'cpu'][0]
        want = 1024 if T <= 1024 else (2048 if T <= 2048 else T)
        check(cpu[1] == want and ('pack', want) in log, f'two entries: {T} rows go to the CPU as {want} rows through the packer')
        check(wrapper._kt_post_submit_event.recorded, f'two entries: {T} rows record the shared-expert fork after the submit')
    check(sorted(set(key[:1] + key[3:] for key in adapter._short_hosts)) == [(1024, 0), (1024, 1), (2048, 0), (2048, 1)],
          'two entries: one pinned set per bucket and slot')
    sys.modules['sglang.srt.layers.moe.kt_ep_wrapper']._KT_HANDOFF_PACK = False


def main(served_root):
    served_root = Path(served_root)
    for rel, pin in overlay.PINS.items():
        assert hashlib.sha256((served_root / rel).read_bytes()).hexdigest() == pin, rel
    tmp = Path(tempfile.mkdtemp(prefix='upstream-picks-r1-'))
    try:
        served, patched = tmp / 'served', tmp / 'patched'
        for root in (served, patched):
            for rel in FILES:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(served_root / rel, root / rel)
        overlay.main(patched)
        test_cutoff((served / FILES[0]).read_text(encoding='utf-8'), (patched / FILES[0]).read_text(encoding='utf-8'))
        test_adapter(served, patched)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'test_stream_cpu_short: {len(CHECKS)} checks passed')


if __name__ == '__main__':
    main(sys.argv[1])
