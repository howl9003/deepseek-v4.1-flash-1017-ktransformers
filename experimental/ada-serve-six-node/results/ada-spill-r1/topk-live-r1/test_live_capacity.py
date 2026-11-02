"""Live top-k against the torch path at the served capacities, on the card, no model (Claude, 1 Nov 2026).
Run with the trial's python and tree on PYTHONPATH. argv: optional 'fuzz' or 'price' to run one half. LIVE_TOPK_FILE tests that overlay instead of the tree's module."""
import os
os.environ.update(KT_IDX_LIVE_TOPK='1', KT_IDX_LIVE_TOPK_MAX_L='2097152')
import collections
import importlib.util
import statistics as st
import sys
import time
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4 import live_topk as tree_live
from sglang.kernels.ops.attention.dsv4.candidate_blocks import select_candidate_block_ids, topk_among_blocks
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DecodeScores, write_decode

live = tree_live
if os.environ.get('LIVE_TOPK_FILE'):
    spec = importlib.util.spec_from_file_location('live_topk_overlay', os.environ['LIVE_TOPK_FILE'])
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
print('module', live.__file__, flush=True)
live_decode_slots, live_topk_among_blocks, live_topk_enabled, live_topk_full = live.live_decode_slots, live.live_topk_among_blocks, live.live_topk_enabled, live.live_topk_full

DEV = torch.device('cuda', 0)
K, BS, TOPB, H, CAP = 512, 8, 2048, 32, 1048580
rng = torch.Generator().manual_seed(1101)
inf = float('inf')


def rand_int(lo, hi, n):
    return torch.randint(lo, hi, (n,), generator=rng)


def values(n, kind):
    """n live logits as fp32 with bf16 low bits, as fp4_index_logits_decode leaves them."""
    if kind == 'normal':
        v = torch.randn(n, generator=rng) * 40
    elif kind == 'coarse':     # ties around the k-th key
        v = (torch.randn(n, generator=rng) * 6).round() / 2
    elif kind == 'few':
        v = torch.tensor([-1., 0.5, 3.])[rand_int(0, 3, n)]
    elif kind == 'equal':
        v = torch.full((n,), 2.5)
    elif kind == 'zeros':      # +0 and -0 order differently under the radix key
        v = torch.tensor([0., -0., 1.])[rand_int(0, 3, n)]
    elif kind == 'rising':
        v = torch.arange(n, dtype=torch.float32) / 16
    elif kind == 'falling':
        v = -torch.arange(n, dtype=torch.float32) / 16
    elif kind == 'posinf':
        v = torch.randn(n, generator=rng)
        v[rand_int(0, n, max(1, n // 8))] = inf
    elif kind == 'neginf':     # finite picks run out
        v = torch.randn(n, generator=rng)
        v[rand_int(0, n, max(1, n - n // 8))] = -inf
    elif kind == 'nan':
        v = torch.randn(n, generator=rng)
        v[rand_int(0, n, max(1, n // 8))] = float('nan')
    return v.to(torch.bfloat16).to(torch.float32)


KINDS = ('normal', 'coarse', 'few', 'equal', 'zeros', 'rising', 'falling', 'posinf', 'neginf', 'nan')
LENS = (0, 1, 7, 8, 9, 63, 64, 65, 300, 511, 512, 513, 700, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 8191, 8192, 8193, 16383, 16384, 16385, 17500, 35000, 100000)


def scores_of(bs, L, lens, kind):
    s = torch.full((bs, L), -inf, dtype=torch.float32)
    for b in range(bs):
        n = int(lens[b])
        if n:
            s[b, :n] = values(n, kind)
    return s.to(DEV)


def torch_slots(r2t, req, lens, ratio, lmax):
    j = torch.arange(lmax, device=DEV)
    valid = j[None, :] < lens[:, None]
    slots = r2t[req[:, None], (j * ratio)[None, :]].to(torch.int64) // ratio
    return slots.masked_fill(~valid, 0)


def decode_args(bs, L, lens, scores, slots, out):
    return SimpleNamespace(out_page_indices=out), DecodeScores(bs=bs, lmax=L, lens=lens, slots=slots, scores=scores)


def fuzz():
    pool_rows = 8
    r2t = torch.randint(0, 1 << 22, (pool_rows, CAP), generator=rng, dtype=torch.int32)
    r2t[0, 5000:5100] = -1     # an unset token inside a live range
    r2t = r2t.to(DEV)
    bad = collections.Counter()
    runs = collections.Counter()
    first = {}
    for case in range(1200):
        kind = KINDS[case % len(KINDS)]
        ratio = 2 if case % 3 == 0 else 1
        L = CAP // ratio
        bs = (1, 1, 2, 4, 8)[case % 5]
        lens = torch.tensor([LENS[int(rand_int(0, len(LENS), 1))] for _ in range(bs)], dtype=torch.int64)
        if case % 97 == 0:
            lens[0] = L
        lens = lens.clamp_max(L).to(DEV)
        req = rand_int(0, pool_rows, bs).to(DEV)
        ref = torch_slots(r2t, req, lens, ratio, L)
        slots = live_decode_slots(r2t, req, lens, ratio, L)
        for b in range(bs):
            n = int(lens[b])
            if not torch.equal(ref[b, :n], slots[b, :n]) or (n % 64 and not bool((slots[b, n:(n + 63) // 64 * 64] == 0).all())):
                bad['slots', kind] += 1
                first.setdefault(('slots', kind), (case, bs, ratio, n))
        runs['slots', kind] += 1
        scores = scores_of(bs, L, lens, kind)

        a = torch.full((bs, K + 4), -7, dtype=torch.int32, device=DEV)
        b_ = a.clone()
        write_decode(*decode_args(bs, L, lens, scores, ref, a), scores.topk(K, dim=-1, sorted=False).indices)
        live_topk_full(scores, ref, lens, K, b_)
        runs['full', kind] += 1
        if not torch.equal(a, b_):
            bad['full', kind] += 1
            r = int((a != b_).any(1).nonzero()[0, 0])
            first.setdefault(('full', kind), (case, bs, ratio, lens[r].item(), a[r, :6].tolist(), b_[r, :6].tolist()))

        if ratio == 1:
            source = scores_of(bs, L, lens, KINDS[(case // 3) % len(KINDS)] if case % 2 else kind)
            blocks = select_candidate_block_ids(source, lens[:, None], topk_blocks=TOPB, block_size=BS)
            a = torch.full((bs, K + 4), -7, dtype=torch.int32, device=DEV)
            b_ = a.clone()
            idx = topk_among_blocks(scores, lens, blocks, K, block_size=BS)
            write_decode(*decode_args(bs, L, lens, scores, ref, a), idx.masked_fill(idx < 0, L))
            live_topk_among_blocks(scores, ref, lens, blocks, K, BS, b_)
            runs['blocks', kind] += 1
            if not torch.equal(a, b_):
                bad['blocks', kind] += 1
                r = int((a != b_).any(1).nonzero()[0, 0])
                first.setdefault(('blocks', kind), (case, bs, lens[r].item(), a[r, :6].tolist(), b_[r, :6].tolist()))
    torch.cuda.synchronize()
    print('runs', dict(collections.Counter({k[0]: 0 for k in runs}) + collections.Counter()) or sorted(runs.items())[:3], flush=True)
    print('total', {p: sum(v for k, v in runs.items() if k[0] == p) for p in ('slots', 'full', 'blocks')}, flush=True)
    print('mismatch', dict(sorted(bad.items())), flush=True)
    for k, v in sorted(first.items()):
        print(' first', k, v, flush=True)
    print('LIVE_FUZZ_' + ('ALL_PASS' if not bad else 'HAS_MISMATCH'), flush=True)


def capture(build):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build()
    return g


def replay(g, runs=150):
    torch.cuda.synchronize()
    d = []
    for _ in range(runs):
        t = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        d.append(time.perf_counter() - t)
    return st.median(d) * 1e6


def price():
    assert live_topk_enabled(CAP, DEV)
    page = 64
    pages = CAP // page + 2
    table = torch.randint(0, 256, (pages, page * 68), generator=rng, dtype=torch.uint8)
    table[:, page * 64:] = 127 + 1
    table = table.to(DEV)
    bs = 1
    q = (torch.randn((bs, H, 128), generator=rng) / 4).to(DEV).to(torch.bfloat16)
    w = (torch.randn((bs, H), generator=rng) / 60).to(DEV).to(torch.bfloat16)
    r2t = torch.arange(CAP, dtype=torch.int32).view(1, CAP).to(DEV)
    req = torch.zeros(bs, dtype=torch.int64, device=DEV)
    lens = torch.full((bs,), 1000, dtype=torch.int64, device=DEV)
    out = torch.empty((bs, K), dtype=torch.int32, device=DEV)
    results = {}
    for ratio, label in ((2, 'full ratio 2'), (1, 'full ratio 1')):
        L = CAP // ratio

        def base():
            fp4_index_logits_decode(q, w, torch.zeros((bs, L), dtype=torch.int64, device=DEV), lens, table, page)

        def torch_full():
            slots = torch_slots(r2t, req, lens, ratio, L)
            s = fp4_index_logits_decode(q, w, slots, lens, table, page)
            write_decode(*decode_args(bs, L, lens, s, slots, out), s.topk(K, dim=-1, sorted=False).indices)

        def live_full():
            slots = live_decode_slots(r2t, req, lens, ratio, L)
            s = fp4_index_logits_decode(q, w, slots, lens, table, page)
            live_topk_full(s, slots, lens, K, out)

        for name, build in (('logits only', base), ('torch', torch_full), ('live', live_full)):
            g = capture(build)
            row = {}
            for n in (300, 700, 1500, 4000, 17500, 35000, 100000, L):
                lens.fill_(min(n, L))
                row[n] = round(replay(g), 1)
            results[label, name] = row
            print(label, name, row, flush=True)
            del g

    L = CAP
    source = torch.full((bs, L), -inf, dtype=torch.float32, device=DEV)
    blocks = torch.empty((bs, TOPB), dtype=torch.int32, device=DEV)

    def torch_blocks():
        slots = torch_slots(r2t, req, lens, 1, L)
        s = fp4_index_logits_decode(q, w, slots, lens, table, page)
        idx = topk_among_blocks(s, lens, blocks, K, block_size=BS)
        write_decode(*decode_args(bs, L, lens, s, slots, out), idx.masked_fill(idx < 0, L))

    def live_blocks():
        slots = live_decode_slots(r2t, req, lens, 1, L)
        s = fp4_index_logits_decode(q, w, slots, lens, table, page)
        live_topk_among_blocks(s, slots, lens, blocks, K, BS, out)

    def publish():
        slots = live_decode_slots(r2t, req, lens, 1, L)
        s = fp4_index_logits_decode(q, w, slots, lens, table, page)
        select_candidate_block_ids(s, lens[:, None], topk_blocks=TOPB, block_size=BS)

    for name, build in (('consumer torch', torch_blocks), ('consumer live', live_blocks), ('source slots+logits+publish', publish)):
        lens.fill_(4000)
        g = capture(build)
        row = {}
        for n in (300, 700, 1500, 4000, 17500, 35000, 100000, L):
            lens.fill_(min(n, L))
            s = torch.full((bs, L), -inf, dtype=torch.float32, device=DEV)
            s[0, :n] = (torch.randn(n, device=DEV) * 8).to(torch.bfloat16).to(torch.float32)
            got = select_candidate_block_ids(s, lens[:, None], topk_blocks=TOPB, block_size=BS)
            blocks.fill_(-1)
            blocks[:, :got.shape[1]] = got
            row[n] = round(replay(g), 1)
        print(name, row, flush=True)
        del g
    print('LIVE_PRICE_DONE', flush=True)


def replay_check():
    """One capture, many device-side lengths: the graph's bytes equal the torch path's at each."""
    page = 64
    table = torch.randint(0, 256, (CAP // page + 2, page * 68), generator=rng, dtype=torch.uint8)
    table[:, page * 64:] = 128
    table = table.to(DEV)
    r2t = torch.randint(0, CAP, (8, CAP), generator=rng, dtype=torch.int32).to(DEV)
    checked = 0
    for bs in (1, 2, 4, 8):
        q = (torch.randn((bs, H, 128), generator=rng) / 4).to(DEV).to(torch.bfloat16)
        q2 = (torch.randn((bs, H, 128), generator=rng) / 4).to(DEV).to(torch.bfloat16)
        w = (torch.randn((bs, H), generator=rng) / 60).to(DEV).to(torch.bfloat16)
        req = rand_int(0, 8, bs).to(DEV)
        lens = torch.full((bs,), 1000, dtype=torch.int64, device=DEV)
        for ratio, path in ((2, 'full'), (1, 'full'), (1, 'blocks')):
            L = CAP // ratio
            out = torch.empty((bs, K), dtype=torch.int32, device=DEV)
            blocks = torch.full((bs, TOPB), -1, dtype=torch.int32, device=DEV)
            blocks[:, 0] = 0

            def build():
                slots = live_decode_slots(r2t, req, lens, ratio, L)
                s = fp4_index_logits_decode(q, w, slots, lens, table, page)
                if path == 'full':
                    live_topk_full(s, slots, lens, K, out)
                else:
                    live_topk_among_blocks(s, slots, lens, blocks, K, BS, out)

            g = capture(build)
            for case in range(40):
                lens.copy_(torch.tensor([min(LENS[int(rand_int(1, len(LENS), 1))], L) for _ in range(bs)], dtype=torch.int64).to(DEV))
                slots = torch_slots(r2t, req, lens, ratio, L)
                s = fp4_index_logits_decode(q, w, slots, lens, table, page)
                ref = torch.full((bs, K), -7, dtype=torch.int32, device=DEV)
                if path == 'full':
                    write_decode(*decode_args(bs, L, lens, s, slots, ref), s.topk(K, dim=-1, sorted=False).indices)
                else:
                    source = fp4_index_logits_decode(q2, w, slots, lens, table, page)
                    got = select_candidate_block_ids(source, lens[:, None], topk_blocks=TOPB, block_size=BS)
                    blocks.fill_(-1)
                    blocks[:, :got.shape[1]] = got
                    idx = topk_among_blocks(s, lens, blocks[:, :got.shape[1]], K, block_size=BS)
                    write_decode(*decode_args(bs, L, lens, s, slots, ref), idx.masked_fill(idx < 0, L))
                out.fill_(-9)
                g.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, ref), (bs, ratio, path, case, lens.tolist())
                checked += 1
            del g
    print('LIVE_REPLAY_PASS', checked, flush=True)


which = sys.argv[1:] or ['fuzz', 'price', 'replay']
if 'replay' in which:
    replay_check()
if 'fuzz' in which:
    fuzz()
if 'price' in which:
    price()
