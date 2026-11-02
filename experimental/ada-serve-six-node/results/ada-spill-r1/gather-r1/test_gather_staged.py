"""The staged gather against the unsplit kernel: the output's bytes on random and adversarial calls, then the price in a captured graph. No model. argv: optional 'fuzz' or 'price'."""
import os
os.environ.setdefault('SGLANG_SM89_ATTN_SPLITK', '0')
import importlib.util
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch

from sglang.kernels.ops.attention import sm89_dual_scope as S

spec = importlib.util.spec_from_file_location('gather_staged', Path(__file__).with_name('gather_staged.py'))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
dev = 'cuda'
HEADS, DIM, BYTES = 64, 512, 584


def kv(blocks, block_size, seed, spread=1):
    g = torch.Generator(device='cuda').manual_seed(seed)
    nope = (torch.randint(0x20, 0x48 + 0x20 * spread, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8)
            | (torch.randint(0, 2, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8) << 7))
    rope = (torch.randn(blocks, block_size, 64, generator=g, device=dev) * 0.05 * spread).to(torch.bfloat16).view(torch.uint8).reshape(blocks, block_size, 128)
    row = torch.cat((nope, rope), dim=2).reshape(blocks, -1)
    scales = torch.randint(118, 130, (blocks, block_size, 8), generator=g, device=dev, dtype=torch.uint8).reshape(blocks, -1)
    return torch.cat((row, scales), dim=1).reshape(blocks, block_size, BYTES)


def base(q3, k_main, idx, tl_main, sink, k_extra, idy, tl_extra, scale):
    rows = q3.shape[0]
    return S.sm89_dual_scope_decode(q3, k_main, idx, tl_main, sink, DIM, scale, k_extra, idy, tl_extra)[:, 0]


def staged(q3, k_main, idx, tl_main, sink, k_extra, idy, tl_extra, scale):
    return G.decode(q3, k_main, idx.reshape(q3.shape[0], -1), tl_main, sink, scale, k_extra, idy.reshape(q3.shape[0], -1) if idy is not None else None, tl_extra)


def case(rng, cache, main, extra, rows, mode):
    k_main, k_extra = cache
    q = (torch.randn(rows, HEADS, DIM, device=dev) * rng.choice((0.05, 0.3, 1.0, 4.0))).to(torch.bfloat16)
    if mode == 'hot':  # one token dominates: the running maximum jumps, alpha far from 1
        q = q * 8
    tokens = k_main.shape[0] * k_main.shape[1]
    idx = torch.stack([torch.randperm(tokens, device=dev)[:main] for _ in range(rows)]).to(torch.int32)
    lengths = [rng.choice((main, main, main // 2, 1, 0, max(main - 70, 0))) for _ in range(rows)]
    if mode == 'holes':  # invalid slots inside the valid range
        idx[torch.rand(idx.shape, device=dev) < 0.2] = -1
    tl_main = None if mode == 'nolen' else torch.tensor(lengths, dtype=torch.int32, device=dev)
    sink = None if rng.random() < 0.3 else (torch.randn(HEADS, device=dev) * 2).to(torch.float32)
    idy = tl_extra = extra_cache = None
    if extra:
        tokens = k_extra.shape[0] * k_extra.shape[1]
        idy = torch.stack([torch.randperm(tokens, device=dev)[:extra] for _ in range(rows)]).to(torch.int32)
        tl_extra = None if mode == 'nolen' else torch.tensor([rng.choice((extra, extra // 2, 0)) for _ in range(rows)], dtype=torch.int32, device=dev)
        extra_cache = k_extra
    return q, k_main, idx, tl_main, sink, extra_cache, idy, tl_extra, DIM ** -0.5


def fuzz():
    rng = random.Random(int(os.environ.get('FUZZ_SEED', '1017')))
    caches = [(kv(2048, 256, 1), kv(512, 256, 2)), (kv(2048, 64, 3, 2), kv(512, 128, 4, 2))]
    bad = total = nonfinite = 0
    seen = {}
    for it in range(int(os.environ.get('FUZZ_N', '600'))):
        main = rng.choice((64, 256, 512, 512, 576, 768, 832, 1024, 100, 513, 63))
        extra = rng.choice((0, 0, 64, 128, 256, 65))
        rows = rng.choice((1, 1, 1, 2, 4, 8))
        mode = rng.choice(('plain', 'plain', 'hot', 'holes', 'nolen'))
        a = case(rng, caches[it % 2], main, extra, rows, mode)
        if os.environ.get('FUZZ_FINITE'):  # rounding, not overflow: scores small enough that every output stays finite
            a = ((a[0].float() / a[0].float().abs().max() * rng.choice((0.02, 0.1, 0.5))).to(torch.bfloat16),) + a[1:]
        x, y = base(*a), staged(*a)
        torch.cuda.synchronize()
        same = torch.equal(x.view(torch.uint16), y.view(torch.uint16))
        nonfinite += int(not bool(torch.isfinite(x.float()).all()))
        total += 1
        seen[mode] = seen.get(mode, 0) + 1
        if not same:
            bad += 1
            if bad <= 8:
                d = (x.view(torch.uint16) != y.view(torch.uint16))
                print(f'mismatch case {it}: main {main} extra {extra} rows {rows} mode {mode}: {int(d.sum())} of {d.numel()} elements, finite {bool(torch.isfinite(x.float()).all())}, max delta {float((x.float() - y.float()).abs().nan_to_num(0).max()):.3g}', flush=True)
    print(f'GATHER_STAGED calls {total} mismatched {bad} calls with non-finite output {nonfinite} modes {seen}', flush=True)
    return bad == 0


def price():
    cache = (kv(2048, 256, 1), kv(512, 256, 2))
    rng = random.Random(1)

    def timed(fn, main, extra):
        calls = [case(rng, cache, main, extra, 1, 'plain') for _ in range(40)]
        calls = [(a[0], a[1], a[2], torch.tensor([main], dtype=torch.int32, device=dev), (torch.randn(HEADS, device=dev)).to(torch.float32), a[5], a[6],
                  torch.tensor([extra], dtype=torch.int32, device=dev) if extra else None, a[8]) for a in calls]
        fn(*calls[0])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        keep = []
        with torch.cuda.graph(g):
            for a in calls:
                keep.append(fn(*a))
        torch.cuda.synchronize()
        runs = []
        for _ in range(30):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            runs.append((time.perf_counter() - t0) * 1e6 / 40)
        return st.median(runs)

    for main, extra in ((256, 0), (512, 64), (768, 64), (1024, 256)):
        print(f'main {main:5d} extra {extra:4d}: unsplit {timed(base, main, extra):6.1f} us, staged {timed(staged, main, extra):6.1f} us (graph overhead about 9 in each)', flush=True)


which = sys.argv[1:] or ['fuzz', 'price']
ok = fuzz() if 'fuzz' in which else True
if 'price' in which:
    price()
sys.exit(0 if ok else 1)
