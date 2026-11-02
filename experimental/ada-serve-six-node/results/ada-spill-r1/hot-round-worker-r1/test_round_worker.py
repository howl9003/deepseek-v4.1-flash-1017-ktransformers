"""CPU test of KT_KV_HOT_DYN_WORKER (Claude, 31 Oct 2026). argv: the patched kt_kv_hot_lend.py. Stub CUDA objects; the copies are held back until the test finishes them.

The property: at every forward, each table entry names a slot whose copy is done. Also: selector off plans and copies as before, a busy round plans nothing,
a failed enqueue raises on the scheduler's thread, and the worker runs in the scheduler's grad and inference mode."""
import importlib.util
import os
import random
import sys
import threading
import types

import numpy as np
import torch

FAILS = []


def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what, flush=True)
    if not cond:
        FAILS.append(what)


def load(path, worker):
    for name in ('sglang', 'sglang.srt', 'sglang.srt.layers', 'sglang.srt.layers.moe'):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
    vmm = types.ModuleType('sglang.srt.layers.moe.kt_kv_vmm')
    vmm.ENABLED = False
    sys.modules['sglang.srt.layers.moe.kt_kv_vmm'] = vmm
    sys.modules['sglang.srt.layers.moe'].kt_kv_vmm = vmm
    os.environ['KT_KV_HOT_DYN_WORKER'] = '1' if worker else '0'
    os.environ['KT_KV_HOT_DYN_SWAPS'], os.environ['KT_KV_HOT_DYN_EVERY'], os.environ['KT_KV_HOT_DYN_HYST'] = '1', '4', '1.25'
    spec = importlib.util.spec_from_file_location('hot_lend_' + str(worker), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Sim:
    """In-flight copies: nothing reaches `loaded` until finish(), and only then does an event report done."""

    def __init__(self):
        self.loaded, self.flight, self.events = {}, [], []
        self.lock = threading.Lock()

    def finish(self):
        with self.lock:
            for key, e in self.flight:
                self.loaded[key] = e
            self.flight.clear()
            for ev in self.events:
                ev.done = True


def stub(m, sim, layers):
    class Ev:
        def __init__(self):
            self.done, self.recorded = False, False

        def record(self, s=None):
            self.recorded = True
            if isinstance(s, Side):
                sim.events.append(self)
            else:
                self.done = True

        def query(self):
            return self.done or not self.recorded

        def synchronize(self):
            pass

    class Stream:
        def wait_event(self, ev):
            sim.finish()

    class Side:
        def wait_event(self, ev):
            pass

    m.torch.cuda.Event = Ev
    m.torch.cuda.current_stream = lambda dev=None: Stream()
    m.torch.cuda.stream = lambda s: Null()
    m.torch.cuda.device = lambda d: Null()

    def arena_load(L, e, dst, order):
        with sim.lock:
            sim.flight.append((dst['slot'], e))

    def tbl_write(table, rows, dev):
        if table is not None and rows:
            t = torch.tensor(rows, dtype=torch.int64)
            table[t[:, 0], t[:, 1]] = t[:, 2]

    m._arena_load = arena_load
    if hasattr(m, '_tbl_write'):
        m._tbl_write = tbl_write
    m.LAYERS[:] = layers
    n = 40
    m.STATE.update(cnt=torch.zeros(len(layers), n), tbl=torch.full((len(layers), n), -1, dtype=torch.int64))
    m.DYN.update(on=True, order=0, stream=Side(), ev_unmap=Ev(), ev_copy=Ev(), ev_cnt=None, buf=torch.zeros(len(layers), n), scores=np.zeros((len(layers), n)),
                 roles={L.idx: {'slot': 'p'} for L in layers}, seen=None, pending=None, job=None, calls=0, rounds=0, swaps=0, swaps_logged=0)


class Layer:
    def __init__(self, idx, k):
        self.idx, self.k, self.n, self.res_t, self.res_held, self.res_now = idx, k, 1, None, [], []
        self.held = [-1] * k
        self.slots = {'p': [(idx, j) for j in range(k)]}


def trial(path, worker, seed, forwards=400):
    m = load(path, worker)
    sim = Sim()
    layers = [Layer(i, 4) for i in range(3)]
    stub(m, sim, layers)
    rng = random.Random(seed)
    violations = 0
    weights = [1.0 / (e + 1) for e in range(40)]
    for f in range(forwards):
        if f % 100 == 0:
            rng.shuffle(weights)  # a new text
        for L in layers:
            for e in rng.choices(range(40), weights, k=6):
                m.STATE['cnt'][L.idx, e] += 1
        m._dyn_round('cpu')
        # the graph's route kernel reads the table here
        for L in layers:
            for e, j in enumerate(m.STATE['tbl'][L.idx].tolist()):
                if j >= 0 and sim.loaded.get((L.idx, j)) != e:
                    violations += 1
        if worker and m.DYN.get('job') is not None:
            m.DYN['job']['thread'].join()
        if rng.random() < 0.4:
            sim.finish()
    return m, violations, m.DYN['swaps'], m.DYN.get('deferred', 0)


def main():
    path = sys.argv[1]
    for seed in range(5):
        m, bad, swaps, deferred = trial(path, True, seed)
        check(bad == 0 and swaps > 0, f'worker seed {seed}: {swaps} swaps, {deferred} rounds deferred, no table entry ahead of its copy')
    _, bad, swaps, _ = trial(path, False, 0)
    check(bad == 0 and swaps > 0, f'selector off: {swaps} swaps, no table entry ahead of its copy')

    m = load(path, True)
    sim = Sim()
    stub(m, sim, [Layer(0, 2)])
    L = m.LAYERS[0]

    def forward():
        m.STATE['cnt'][0, 7] += 3
        m.STATE['cnt'][0, 9] += 2
        m._dyn_round('cpu')

    for _ in range(4):
        forward()  # the first round only starts the counts
    for _ in range(4):
        forward()
    job = m.DYN['job']
    check(job is not None and m.DYN['swaps'] == 2, f"a round with room plans its swaps ({m.DYN['swaps']})")
    job['thread'].join()
    for _ in range(4):
        forward()  # a whole cadence with the copies still running
    check(m.DYN['job'] is job and m.DYN['swaps'] == 2 and m.DYN['deferred'] == 1, f"a busy round plans nothing, its counts wait ({m.DYN['deferred']} deferred)")
    sim.finish()
    forward()
    check(m.DYN['job'] is None and sorted(m.STATE['tbl'][0].tolist())[-2:] == [0, 1], 'the copies done, both entries go in at that forward')
    m.DYN['job'] = None
    m._job_start([(L, 0, 1, 33)], {0: [(0, 33, 1)], 1: []}, 'cpu')
    m.DYN['job']['thread'].join()
    m._job_publish('cpu')
    check(m.DYN['job'] is not None and int(m.STATE['tbl'][0, 33]) == -1, 'copies not done: nothing enters the table, the job stays')
    sim.finish()
    m._job_publish('cpu')
    check(m.DYN['job'] is None and int(m.STATE['tbl'][0, 33]) == 1, 'copies done: the entry goes in at that forward')

    def boom(*a):
        raise ValueError('x')

    m._arena_load = boom
    m._job_start([(L, 0, 1, 33)], {0: [], 1: []}, 'cpu')
    try:
        m._job_join()
        check(False, 'a failed enqueue raises on the scheduler thread')
    except RuntimeError:
        check(True, 'a failed enqueue raises on the scheduler thread')
    m.DYN['job'] = None

    seen = {}

    def spy(*a):
        seen.update(inference=torch.is_inference_mode_enabled(), grad=torch.is_grad_enabled(), thread=threading.current_thread().name)

    m._arena_load = spy
    with torch.inference_mode():
        m._job_start([(L, 0, 1, 33)], {0: [], 1: []}, 'cpu')
    m._job_join()
    check(seen == dict(inference=True, grad=False, thread='kt-kv-hot-copy'), f'the worker keeps the scheduler thread\'s modes ({seen})')
    print('ALL_PASS' if not FAILS else 'FAILED ' + str(FAILS))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
