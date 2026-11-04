"""CPU-only tests for apply_prefill_warmup.py (upstream-picks-r1).

argv: the served sglang package directory and the served engram-adapter directory (served pins). The test copies
srt/entrypoints/warmup.py and engram_backend.py, applies the overlay to the copies and checks:
  1. warmup.py: the served file is a byte prefix of the patched one (no served warm-up changes); prefill_lengths is
     registered, sends nothing with the variable unset, one prompt a length otherwise, fresh ids per launch unless a
     seed is set, the disaggregation fields, and refuses non-positive lengths;
  2. engram_backend.py on numpy stand-ins for torch, libcudart and the row store: with DSV41_ENGRAM_RETIRE_EVENT unset
     a growing eager set synchronises and drops its Works exactly as the served file does; with it set, no
     synchronise, the old set and its Works are held behind an event and dropped at the next eager lookup once the
     event has passed, never during a capture."""
import asyncio
import ctypes
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
import apply_prefill_warmup as overlay  # noqa: E402

CHECKS = []


def check(cond, what):
    CHECKS.append(what)
    assert cond, what


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ 1. warmup.py
def warmup_stubs():
    for n in ('sglang', 'sglang.srt', 'sglang.srt.disaggregation', 'sglang.srt.managers'):
        sys.modules.setdefault(n, types.ModuleType(n))

    class GenerateReqInput:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    sys.modules['sglang.srt.disaggregation.utils'] = types.SimpleNamespace(FAKE_BOOTSTRAP_HOST='fake-host')
    sys.modules['sglang.srt.managers.io_struct'] = types.SimpleNamespace(GenerateReqInput=GenerateReqInput)
    sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda it, **kw: it, trange=range)


class TokenizerManager:
    def __init__(self):
        self.sent = []

    def generate_request(self, req, raw):
        self.sent.append(req)

        async def gen():
            yield {'done': True}
        return gen()


def run_warmup(module, env, mode='null'):
    saved = {k: os.environ.pop(k) for k in ('SGLANG_WARMUP_PREFILL_LENGTHS', 'SGLANG_WARMUP_PREFILL_SEED') if k in os.environ}
    os.environ.update(env)
    try:
        tm = TokenizerManager()
        asyncio.run(module.execute_warmups(mode, ['prefill_lengths'], tm))
        return tm.sent
    finally:
        for k in ('SGLANG_WARMUP_PREFILL_LENGTHS', 'SGLANG_WARMUP_PREFILL_SEED'):
            os.environ.pop(k, None)
        os.environ.update(saved)


def test_warmup(served_path, patched_path):
    served_b, patched_b = served_path.read_bytes(), patched_path.read_bytes()
    check(patched_b.startswith(served_b), 'warmup.py: the served text is an unchanged prefix (only an appended function)')
    warmup_stubs()
    served = load('warmup_served', served_path)
    patched = load('warmup_patched', patched_path)
    check(set(patched._warmup_registry) - set(served._warmup_registry) == {'prefill_lengths'}, 'one new warm-up registered')
    check(run_warmup(patched, {}) == [], 'unset: no prompt')
    check(run_warmup(patched, {'SGLANG_WARMUP_PREFILL_LENGTHS': ' , '}) == [], 'empty list: no prompt')
    sent = run_warmup(patched, {'SGLANG_WARMUP_PREFILL_LENGTHS': '1024, 36864'})
    check([len(r.input_ids) for r in sent] == [1024, 36864], 'one prompt a length, in order')
    check(all(1000 <= min(r.input_ids) and max(r.input_ids) < 2 ** 16 for r in sent), 'ids in [1000, 65536)')
    check(all(r.sampling_params == {'max_new_tokens': 1, 'temperature': 0.0} for r in sent), 'one greedy token')
    check(all(not hasattr(r, 'bootstrap_room') for r in sent), 'no bootstrap fields without disaggregation')
    again = run_warmup(patched, {'SGLANG_WARMUP_PREFILL_LENGTHS': '1024'})
    check(again[0].input_ids != sent[0].input_ids, 'no seed: a fresh prompt each launch')
    s1 = run_warmup(patched, {'SGLANG_WARMUP_PREFILL_LENGTHS': '1024', 'SGLANG_WARMUP_PREFILL_SEED': '7'})
    s2 = run_warmup(patched, {'SGLANG_WARMUP_PREFILL_LENGTHS': '1024', 'SGLANG_WARMUP_PREFILL_SEED': '7'})
    check(s1[0].input_ids == s2[0].input_ids, 'SGLANG_WARMUP_PREFILL_SEED fixes the prompt')
    d = run_warmup(patched, {'SGLANG_WARMUP_PREFILL_LENGTHS': '512'}, mode='prefill')
    check(d[0].bootstrap_room == 0 and d[0].bootstrap_host == 'fake-host', 'disaggregation fields as the other warm-ups')
    for bad in ('0', '-5', '1024,0'):
        try:
            run_warmup(patched, {'SGLANG_WARMUP_PREFILL_LENGTHS': bad})
        except ValueError:
            check(True, f'{bad!r} refused')
        else:
            raise AssertionError(f'{bad!r} accepted')


# ------------------------------------------------------------------ 2. engram_backend.py
class FakeLib:
    def __init__(self):
        self.row_store_lookup = ctypes.c_void_p(0x1234)

    def __getattr__(self, name):
        fn = lambda *a: 0
        fn.argtypes = fn.restype = None
        setattr(self, name, fn)
        return fn


def load_engram(path, name, env):
    real_cdll = ctypes.CDLL
    tmp = Path(tempfile.mkdtemp(prefix='fake-cudart-'))
    (tmp / 'lib').mkdir()
    (tmp / 'lib' / 'libcudart.so.12').write_bytes(b'')
    nvidia = types.ModuleType('nvidia')
    nvidia.cuda_runtime = types.ModuleType('nvidia.cuda_runtime')
    nvidia.cuda_runtime.__file__ = str(tmp / '__init__.py')
    sys.modules['nvidia'], sys.modules['nvidia.cuda_runtime'] = nvidia, nvidia.cuda_runtime
    sys.modules['torch'] = torch
    for n in ('sglang', 'sglang.kernels', 'sglang.kernels.ops', 'sglang.kernels.ops.embeddings'):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules['sglang.kernels.ops.embeddings.engram_gather'] = types.SimpleNamespace(engram_gather=lambda *a: None)
    saved = os.environ.pop('DSV41_ENGRAM_RETIRE_EVENT', None)
    os.environ.update(env)
    ctypes.CDLL = lambda *a, **kw: FakeLib()
    try:
        return load(name, path)
    finally:
        ctypes.CDLL = real_cdll
        os.environ.pop('DSV41_ENGRAM_RETIRE_EVENT', None)
        if saved is not None:
            os.environ['DSV41_ENGRAM_RETIRE_EVENT'] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def owned_fn(eb):
    class EngramEmbedding:
        pass

    class EngramHasher:
        def __init__(self, *a, **kw):
            pass
    module = types.SimpleNamespace(EngramEmbedding=EngramEmbedding, EngramHasher=EngramHasher)
    eb.install(module)
    return EngramEmbedding._owned_rows


def fake_self():
    s = types.SimpleNamespace(_staging={}, _works={}, _retired=[], _store=1)
    s.new_rows = lambda idx: torch.zeros((idx.numel(), 256), dtype=torch.uint8, device='cuda')
    return s


def lookups(owned, s, counts):
    out = []
    for c in counts:
        before = torch.cuda.syncs
        idx = torch.Tensor(np.arange(c, dtype=np.int64), torch.int64, 'cuda')
        owned(s, idx)
        out.append(torch.cuda.syncs - before)
    return out


def shape_of(s):
    return (sorted((str(k), v[0].numel()) for k, v in s._staging.items()),
            sorted((k[2], v.count) for k, v in s._works.items()))


def test_engram(served_path, patched_path):
    torch.reset()
    served = owned_fn(load_engram(served_path, 'engram_served', {}))
    off = owned_fn(load_engram(patched_path, 'engram_off', {}))
    on_mod = load_engram(patched_path, 'engram_on', {'DSV41_ENGRAM_RETIRE_EVENT': '1'})
    on = owned_fn(on_mod)
    check(not sys.modules['engram_off'].RETIRE_EVENT and on_mod.RETIRE_EVENT, 'the switch reads DSV41_ENGRAM_RETIRE_EVENT=1 only')
    counts = [3, 20000, 20000, 40000, 3, 70000]      # decode, eager 32K, same, grow to 64K, decode, grow to 128K

    a, b = fake_self(), fake_self()
    torch.reset()
    sa = lookups(served, a, counts)
    ev_a = torch.Event.created
    torch.reset()
    sb = lookups(off, b, counts)
    check(sa == sb == [0, 0, 0, 1, 0, 1], f'unset: the same synchronises as served at each growth {sb}')
    check(shape_of(a) == shape_of(b) and b._retired == [] and torch.Event.created == ev_a == 0,
          'unset: same staging sets and Works as served, nothing retired, no event')

    c = fake_self()
    torch.reset()
    sc = lookups(on, c, counts[:4])
    check(sc == [0, 0, 0, 0] and torch.Event.created == 1, 'on: growth records an event instead of synchronising')
    check(len(c._retired) == 1 and c._retired[0][1][0].numel() == 32768 and c._retired[0][2],
          'on: the old 32K set and its Works are held')
    check(all(v.ids != c._retired[0][1][0].data_ptr() for v in c._works.values()), 'on: the live Works no longer name the old set')
    check(set(c._staging) == {(0, 4), (0, 'eager')} and c._staging[(0, 'eager')][0].numel() == 65536,
          'on: the eager set grew to 64K')
    lookups(on, c, [40000])
    check(len(c._retired) == 1, 'on: kept while the stream has not passed the event')
    torch.cuda.capturing = True
    lookups(on, c, [3])
    check(len(c._retired) == 1, 'on: a captured decode lookup never queries or drops')
    torch.cuda.capturing = False
    torch.cuda.events_complete = True
    lookups(on, c, [40000])
    check(c._retired == [], 'on: dropped at the next eager lookup once the event has passed')
    torch.cuda.events_complete = False
    lookups(on, c, [70000])
    check(len(c._retired) == 1 and torch.cuda.syncs == 0, 'on: a second growth retires again, still no synchronise')


def main(sglang_root, adapter_root):
    sglang_root, adapter_root = Path(sglang_root), Path(adapter_root)
    w_src, e_src = sglang_root / 'srt/entrypoints/warmup.py', adapter_root / 'engram_backend.py'
    assert hashlib.sha256(w_src.read_bytes()).hexdigest() == overlay.WARMUP_PIN
    assert hashlib.sha256(e_src.read_bytes()).hexdigest() == overlay.ENGRAM_PIN
    tmp = Path(tempfile.mkdtemp(prefix='upstream-picks-r1-'))
    try:
        (tmp / 'sglang/srt/entrypoints').mkdir(parents=True)
        (tmp / 'engram-adapter').mkdir()
        shutil.copyfile(w_src, tmp / 'sglang/srt/entrypoints/warmup.py')
        shutil.copyfile(e_src, tmp / 'engram-adapter/engram_backend.py')
        overlay.main(tmp / 'sglang', tmp / 'engram-adapter')
        test_warmup(w_src, tmp / 'sglang/srt/entrypoints/warmup.py')
        test_engram(e_src, tmp / 'engram-adapter/engram_backend.py')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'test_prefill_warmup: {len(CHECKS)} checks passed')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
