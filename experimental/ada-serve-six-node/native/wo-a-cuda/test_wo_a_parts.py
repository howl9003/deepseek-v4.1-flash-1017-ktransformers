"""Bitwise and speed test of the wo_a CUDA kernel against the Triton _parts, on the card, no model (Claude, 1 Nov 2026).
Run with the trial's python and tree on PYTHONPATH. argv: binary .so (several for a price sweep; the first gets the full identity set)."""
import importlib.util
import statistics as st
import sys
import time
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.gemm import fixed_wo_a_fp8 as wo
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

DEV = torch.device('cuda', 0)
NG, N, K, S = 8, 1024, 4096, 4
rng = torch.Generator().manual_seed(1101)


def load(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.name.split('.')[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def codes(shape):
    c = torch.randint(0, 256, shape, generator=rng, dtype=torch.uint8)
    c[(c & 0x7f) == 0x7f] = 0    # no NaN code exists in a quantised input or a stored weight
    return c


def e4m3(c):
    return c.to(DEV).view(torch.float8_e4m3fn)


def triton_parts(q, qs, w, scale, m, bm=None):
    bm = 16 if bm is None else bm
    p = torch.empty((S, m, NG, N), device=DEV, dtype=torch.float32)
    wo._parts[(triton.cdiv(m, bm), triton.cdiv(N, 128), NG * S)](q, w, qs, scale, p, m, NG, N, K, S, bm, 128, 32, num_warps=4, num_stages=3, enable_fp_fusion=False)
    return p


def equal(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def price(fn, runs=120):
    torch.cuda.synchronize()
    d = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        d.append(time.perf_counter() - t)
    return st.median(d) * 1e6


ext = [load(p) for p in sys.argv[1:]]
checked = 0
w = e4m3(codes((NG, N, K)))
for case in range(240):
    if case % 20 == 0:
        w = e4m3(codes((NG, N, K)))
    m = (1, 2, 3, 4, 5, 7, 8, 12, 16)[case % 9]
    q = e4m3(codes((m * NG, K)))
    kind = case % 6
    if kind < 3:     # the served range: E8M0 powers of two
        qs = torch.ldexp(torch.ones((m * NG, K // 32)), torch.randint(-22, 9, (m * NG, K // 32), generator=rng)).to(DEV)
        scale = torch.ldexp(torch.ones((NG, N // 32, K // 32)), torch.randint(-20, 5, (NG, N // 32, K // 32), generator=rng)).to(DEV)
    elif kind == 3:  # products into the subnormals and past them: a .ftz or an fma would show here
        qs = torch.ldexp(torch.ones((m * NG, K // 32)), torch.randint(-110, -70, (m * NG, K // 32), generator=rng)).to(DEV)
        scale = torch.ldexp(torch.ones((NG, N // 32, K // 32)), torch.randint(-60, -20, (NG, N // 32, K // 32), generator=rng)).to(DEV)
    elif kind == 4:  # overflow to infinity and back through the adds
        qs = torch.ldexp(torch.ones((m * NG, K // 32)), torch.randint(60, 120, (m * NG, K // 32), generator=rng)).to(DEV)
        scale = torch.ldexp(torch.ones((NG, N // 32, K // 32)), torch.randint(30, 100, (NG, N // 32, K // 32), generator=rng)).to(DEV)
    else:            # not powers of two: the literal mul, mul, add
        qs = (torch.rand((m * NG, K // 32), generator=rng) * 3).to(DEV)
        scale = (torch.rand((NG, N // 32, K // 32), generator=rng) * 3).to(DEV)
    old = triton_parts(q, qs, w, scale, m)
    new = torch.full((S, m, NG, N), 12345., device=DEV, dtype=torch.float32)
    ext[0].parts(q, qs, w, scale, new)
    torch.cuda.synchronize()
    assert equal(old, new), (case, m, kind)
    checked += new.numel()

# the whole wrapper on BF16 activations, through the quantiser and the unchanged _merge
for m in (1, 2, 4, 8):
    x = (torch.randn((m, NG, K), generator=rng) * torch.ldexp(torch.ones((m, NG, 1)), torch.randint(-6, 6, (m, NG, 1), generator=rng))).to(DEV).to(torch.bfloat16)
    scale = torch.ldexp(torch.ones((NG, N // 32, K // 32)), torch.randint(-12, 2, (NG, N // 32, K // 32), generator=rng)).to(DEV)
    old = wo.grouped(x, w, scale, splits=4)
    q, qs = sglang_per_token_group_quant_fp8(x.view(m * NG, K), 32, scale_ue8m0=True)
    p = torch.empty((S, m, NG, N), device=DEV, dtype=torch.float32)
    ext[0].parts(q, qs.contiguous(), w, scale, p)
    out = torch.empty((m, NG, N), device=DEV, dtype=x.dtype)
    wo._merge[(triton.cdiv(m * NG * N, 256),)](p, out, m * NG * N, S, 256, num_warps=4, enable_fp_fusion=False)
    torch.cuda.synchronize()
    assert torch.equal(old.view(torch.int16), out.view(torch.int16)), m

print('WO_A_IDENTITY_PASS', dict(partials=checked, cases=240), flush=True)

# price as the graph sees it: 40 layers' weights, cold, one call each. Isolated repeats sit in the 64 MB L2 and flatter everything.
m = 1
q = e4m3(codes((m * NG, K)))
qs = torch.ldexp(torch.ones((m * NG, K // 32)), torch.randint(-12, 2, (m * NG, K // 32), generator=rng)).to(DEV)
scale = torch.ldexp(torch.ones((NG, N // 32, K // 32)), torch.randint(-12, 2, (NG, N // 32, K // 32), generator=rng)).to(DEV)
weights = [e4m3(codes((NG, N, K))) for _ in range(40)]
scales = [scale.clone() for _ in range(40)]
p = torch.empty((S, m, NG, N), device=DEV, dtype=torch.float32)


def chain(call):
    def run():
        for a, b in zip(weights, scales):
            call(a, b)
    return run


def triton_call(a, b):
    wo._parts[(1, 8, NG * S)](q, a, qs, b, p, m, NG, N, K, S, 16, 128, 32, num_warps=4, num_stages=3, enable_fp_fusion=False)


for name, call in [('triton', triton_call)] + [(Path(path).name, (lambda e: lambda a, b: e.parts(q, qs, a, b, p))(e)) for path, e in zip(sys.argv[1:], ext)]:
    print(name, 'us per call, 40 cold weights', round(price(chain(call), 60) / 40, 1), flush=True)
