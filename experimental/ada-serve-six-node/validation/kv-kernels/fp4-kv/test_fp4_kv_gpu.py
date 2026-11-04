"""FP4 compressed KV on the 4080: writer, loader, attention, off arm, compile facts, price. No model. argv: stages from writer loader attention off price. env BASE_SGLANG = the served sglang package."""
import importlib.util
import math
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("SGLANG_SM89_ATTN_SPLITK", "0")
os.environ["SGLANG_SM89_ATTN_STAGED"] = "1"
os.environ["SGLANG_SM89_FP4_COMPRESSED_KV"] = "1"

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention import sm89_dual_scope as S
from sglang.kernels.ops.attention import sm89_gather_staged as G
from sglang.kernels.ops.attention.dsv4 import dequant_k_cache as D
from sglang.kernels.ops.attention.dsv4 import triton_sparse_decode as _pkg  # noqa: F401
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention.dsv4.dsv41_sparse import _rope_fq4

dev = "cuda"
HEADS, DIM, V4_BYTES, FP4 = 64, 512, 584, KVLayout.V41_FP4
fails = []


def report(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + " " + str(detail), flush=True)
    if not ok:
        fails.append(name)


def latent(n, seed, edges=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(n, DIM, generator=g, device=dev) * torch.exp(torch.rand(n, 1, generator=g, device=dev) * 11 - 6)
    if edges:
        x[0, :16] = 0.0
        x[1] = 1e-6
        x[2, :16] = 4000.0
        x[3, ::2] = -0.0
        x[4, 16:32] = -1e-9
    return x.to(torch.bfloat16)


def freqs(n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.polar(torch.ones(n, 32, device=dev), torch.rand(n, 32, generator=g, device=dev) * 2 * math.pi)


def fp4_cache(x, f, P):
    """Pages from the real CUDA store, as the pool lays them out; the view the attention call receives."""
    n = x.shape[0]
    pages = -(-n // P)
    raw = torch.zeros(pages, FP4.page_bytes(P), dtype=torch.uint8, device=dev)
    fused_store_cache(x, raw, torch.arange(n, device=dev, dtype=torch.int64), page_size=P, type="flashmla", layout=FP4, freqs_cis=f)
    return raw, raw[:, : P * FP4.bytes_per_token].view(pages, P, 1, FP4.bytes_per_token)


def writer():
    """The store against the Triton fake-quant, and both against the pure-torch form of the same step, which the Triton kernel is meant to equal."""
    from sglang.srt.layers.attention.dsv4 import dsv41_sparse as DS

    bad = signed = total = store_torch = triton_torch = 0
    for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (1, 64), (7, 128), (65536, 128)) * 4):
        x, f = latent(n, seed, edges=n > 4), freqs(n, seed + 100)
        want = _rope_fq4(x, f, 64, compressed_kv=True).to(torch.bfloat16).view(torch.uint16)
        pure = DS.fake_quant_compressed_kv(DS.rope_tail(x, f, 64)).to(torch.bfloat16).to(torch.float32)
        raw, _ = fp4_cache(x, f, P)
        got = D.dequantize_k_cache_paged_dsv4(raw, torch.arange(n, device=dev), page_size=P, layout=FP4)[:, 0].view(torch.uint16)
        store_torch += int((got.view(torch.bfloat16).to(torch.float32) != pure).sum())
        triton_torch += int((want.view(torch.bfloat16).to(torch.float32) != pure).sum())
        got, want = got.to(torch.int32), want.to(torch.int32)
        diff = got != want
        zeros = diff & ((got & 0x7FFF) == 0) & ((want & 0x7FFF) == 0)
        real = diff & ~zeros
        bad += int(real.sum())
        for r, c in real.nonzero().tolist()[:8]:
            tile = x[r, c // 16 * 16:c // 16 * 16 + 16].float()
            gf = float(torch.tensor([got[r, c]], dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)[0])
            wf = float(torch.tensor([want[r, c]], dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)[0])
            print(f"writer diff n {n} P {P} row {r} col {c}: got {gf} want {wf} x {float(x[r, c])} tile amax {float(tile.abs().max())}", flush=True)
        signed += int(zeros.sum())
        total += want.numel()
    report("B store equals the pure-torch step", store_torch == 0, f"{store_torch} of {total}")
    print(f"B Triton fake-quant against pure torch: {triton_torch} of {total}; store against Triton fake-quant: {bad}, signed-zero only {signed}", flush=True)


@triton.jit
def _tile_kernel(KV, Idx, Out, n, block_size, stride_kv, tile, stride_o0, stride_o1):
    offs_n = tl.program_id(0) * 64 + tl.arange(0, 64)
    kv_block_base, data_off, scale_off, valid = _block_valid(KV, Idx, offs_n, n, 0, block_size, stride_kv, False, True)
    kv = fp4_tile(kv_block_base, data_off, scale_off, valid[:, None], tile)
    tl.store(Out + offs_n[:, None] * stride_o0 + tl.arange(0, 64)[None, :] * stride_o1, kv, mask=(offs_n < n)[:, None])


def loader():
    globals()["_block_valid"], globals()["fp4_tile"] = G._block_valid, G.fp4_tile
    bad = total = 0
    for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (640, 128))):
        x, f = latent(n, seed, edges=True), freqs(n, seed + 100)
        raw, _ = fp4_cache(x, f, P)
        idx = torch.randperm(n, device=dev).to(torch.int32)
        idx[torch.rand(n, device=dev) < 0.1] = -1
        want = D.dequantize_k_cache_paged_dsv4(raw, idx.clamp(min=0), page_size=P, layout=FP4)[:, 0]
        want = torch.where((idx >= 0)[:, None], want, torch.zeros_like(want)).view(torch.uint16)
        out = torch.zeros(n, DIM, dtype=torch.bfloat16, device=dev)
        for tile in range(8):
            t = torch.zeros(n, 64, dtype=torch.bfloat16, device=dev)
            _tile_kernel[(-(-n // 64),)](raw, idx, t, n, P, raw.stride(0), tile, t.stride(0), t.stride(1))
            out[:, tile * 64:tile * 64 + 64] = t
        d = out.view(torch.uint16) != want
        bad += int(d.sum())
        total += d.numel()
    report("C loader equals the V41_FP4 dequant", bad == 0, f"{bad} of {total}")


def twin(path, name, **globs):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    for k, v in globs.items():
        setattr(m, k, v)
    return m


@triton.jit
def oracle_tile(kv_block_base, data_off, scale_off, valid_2d, tile):
    """The same tile from a BF16 row: the byte loads stay, so the dot's kWidth does. Rows sit at 4 x the FP4 data stride."""
    offs = tl.arange(0, 64)
    ptrs = kv_block_base[:, None] + (data_off * 4)[:, None] + tile * 128 + offs[None, :] * 2
    lo = tl.load(ptrs, mask=valid_2d, other=0).to(tl.uint16)
    hi = tl.load(ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
    kv = (lo | (hi << 8)).to(tl.bfloat16, bitcast=True)
    return tl.where(valid_2d, kv, 0.0)


def bf16_rows(x, f, P):
    """The BF16 values of an FP4 cache, one row per token, in a cache view with the FP4 shape and a 1024 B row stride."""
    n = x.shape[0]
    pages = -(-n // P)
    v = D.dequantize_k_cache_paged_dsv4(fp4_cache(x, f, P)[0], torch.arange(n, device=dev), page_size=P, layout=FP4)[:, 0]
    raw = torch.zeros(pages, P * 1024, dtype=torch.uint8, device=dev)
    raw.view(-1)[: n * 1024] = v.view(torch.uint8).reshape(-1)
    return raw, raw[:, : P * FP4.bytes_per_token].view(pages, P, 1, FP4.bytes_per_token)


def v4_cache(blocks, block_size, seed, spread=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    nope = (torch.randint(0x20, 0x48 + 0x20 * spread, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8)
            | (torch.randint(0, 2, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8) << 7))
    rope = (torch.randn(blocks, block_size, 64, generator=g, device=dev) * 0.05 * spread).to(torch.bfloat16).view(torch.uint8).reshape(blocks, block_size, 128)
    row = torch.cat((nope, rope), dim=2).reshape(blocks, -1)
    scales = torch.randint(118, 130, (blocks, block_size, 8), generator=g, device=dev, dtype=torch.uint8).reshape(blocks, -1)
    return torch.cat((row, scales), dim=1).reshape(blocks, block_size, 1, V4_BYTES)


def call(rng, k_main, extra_view, main, extra, rows, mode):
    q = (torch.randn(rows, HEADS, DIM, device=dev) * rng.choice((0.05, 0.3, 1.0, 4.0))).to(torch.bfloat16)
    if mode == "hot":
        q = q * 8
    if mode == "nan":
        q[0, 0, 0] = float("nan")
    tokens = k_main.shape[0] * k_main.shape[1]
    idx = torch.stack([torch.randperm(tokens, device=dev)[:main] for _ in range(rows)]).to(torch.int32)
    lengths = [rng.choice((main, main, main // 2, 1, 0, max(main - 70, 0))) for _ in range(rows)]
    if mode == "holes":
        idx[torch.rand(idx.shape, device=dev) < 0.2] = -1
    tl_main = None if mode == "nolen" else torch.tensor(lengths, dtype=torch.int32, device=dev)
    sink = None if rng.random() < 0.3 else (torch.randn(HEADS, device=dev) * 2).to(torch.float32)
    tokens = extra_view.shape[0] * extra_view.shape[1]
    idy = torch.stack([torch.randperm(tokens, device=dev)[:extra] for _ in range(rows)]).to(torch.int32)
    tl_extra = None if mode == "nolen" else torch.tensor([rng.choice((extra, extra // 2, 0)) for _ in range(rows)], dtype=torch.int32, device=dev)
    return q, k_main, idx, tl_main, sink, DIM, DIM ** -0.5, idy, tl_extra


def attention():
    """The FP4 kernels against twins reading the same values as BF16 rows: staged at 1-8 rows, unsplit above."""
    root = Path(G.__file__).parent
    G2 = twin(root / "sm89_gather_staged.py", "gather_oracle", fp4_tile=oracle_tile)
    F2 = twin(root / "dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py", "fused_oracle", fp4_tile=oracle_tile)
    k_main = v4_cache(2048, 128, 1)
    P = 128
    n = 512 * P
    x, f = latent(n, 2, edges=True), freqs(n, 3)
    _, fp4 = fp4_cache(x, f, P)
    _, rows_view = bf16_rows(x, f, P)
    rng = random.Random(int(os.environ.get("FUZZ_SEED", "1017")))
    bad = total = 0
    seen = {}
    for it in range(int(os.environ.get("FUZZ_N", "3000"))):
        main = rng.choice((64, 256, 512, 768, 1024, 100, 63))
        extra = rng.choice((64, 128, 256, 512, 65))
        rows = rng.choice((1, 1, 1, 2, 4, 8, 16, 64, 256))
        mode = rng.choice(("plain", "plain", "hot", "holes", "nolen", "nan"))
        q, km, idx, tl_main, sink, dim, scale, idy, tl_extra = call(rng, k_main, fp4, main, extra, rows, mode)
        a = S.sm89_dual_scope_decode(q, km, idx, tl_main, sink, dim, scale, fp4, idy, tl_extra)
        if rows <= 8:
            b = G2.decode(q, km, idx.reshape(rows, -1).contiguous(), tl_main, sink, scale, rows_view, idy.reshape(rows, -1).contiguous(), tl_extra).unsqueeze(1)
        else:
            saved = S._F
            S._F = F2
            b = S.sm89_dual_scope_decode(q, km, idx, tl_main, sink, dim, scale, rows_view, idy, tl_extra)
            S._F = saved
        torch.cuda.synchronize()
        same = torch.equal(a.view(torch.uint16), b.view(torch.uint16))
        total += 1
        seen[mode] = seen.get(mode, 0) + 1
        if not same:
            bad += 1
            if bad <= 8:
                d = a.view(torch.uint16) != b.view(torch.uint16)
                print(f"mismatch {it}: main {main} extra {extra} rows {rows} {mode}: {int(d.sum())} of {d.numel()}", flush=True)
    report("D attention equals the BF16-row twin", bad == 0, f"{bad} of {total} calls {seen}")
    import triton.compiler.compiler as CC

    seen_f = set()
    for mod, names in ((G, ("_scores_kernel", "_output_kernel", "_probabilities_kernel")), (S._F, ("_fused_gather_attn_dsv4_dual_scope_kernel",))):
        for nm in names:
            fn = getattr(mod, nm)
            fn = fn if hasattr(fn, "device_caches") else fn.fn
            for cache in getattr(fn, "device_caches", {}).values():
                for k in cache[0].values():
                    m = getattr(k, "metadata", None)
                    seen_f.add((nm, getattr(m, "n_regs", None), getattr(m, "n_spills", None), getattr(m, "shared", None), getattr(k, "n_regs", None), getattr(k, "n_spills", None)))
    for row in sorted(seen_f, key=str):
        print("F", row, flush=True)


def off():
    """V4 caches through the patched kernels against the served files."""
    base = Path(os.environ["BASE_SGLANG"])
    G0 = twin(base / "kernels/ops/attention/sm89_gather_staged.py", "gather_base")
    F0 = twin(base / "kernels/ops/attention/dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py", "fused_base")
    S0 = twin(base / "kernels/ops/attention/sm89_dual_scope.py", "scope_base", _F=F0)
    k_main, k_extra = v4_cache(2048, 128, 1), v4_cache(512, 128, 2)
    rng = random.Random(1)
    bad = 0
    N = int(os.environ.get("OFF_N", "600"))
    for it in range(N):
        main, extra = rng.choice((64, 256, 512, 1024)), rng.choice((64, 128, 256))
        rows = rng.choice((1, 2, 4, 8, 16, 64))
        mode = rng.choice(("plain", "hot", "holes", "nolen"))
        q, km, idx, tl_main, sink, dim, scale, idy, tl_extra = call(rng, k_main, k_extra, main, extra, rows, mode)
        a = S.sm89_dual_scope_decode(q, km, idx, tl_main, sink, dim, scale, k_extra, idy, tl_extra)
        if rows <= 8:
            b = G0.decode(q, km, idx.reshape(rows, -1).contiguous(), tl_main, sink, scale, k_extra, idy.reshape(rows, -1).contiguous(), tl_extra).unsqueeze(1)
        else:
            b = S0.sm89_dual_scope_decode(q, km, idx, tl_main, sink, dim, scale, k_extra, idy, tl_extra)
        torch.cuda.synchronize()
        bad += int(not torch.equal(a.view(torch.uint16), b.view(torch.uint16)))
    report("E selector-off kernels equal the served files", bad == 0, f"{bad} of {N}")


def price():
    P = 128
    k_main = v4_cache(2048, 256, 1)
    n = 4096 * P
    x, f = latent(n, 5), freqs(n, 6)
    _, fp4 = fp4_cache(x, f, P)
    v4 = v4_cache(4096, P, 7)
    rng = random.Random(1)

    def timed(extra_view, main, extra, rows):
        calls = []
        for _ in range(40):
            q, km, idx, _, _, dim, scale, idy, _ = call(rng, k_main, extra_view, main, extra, rows, "plain")
            calls.append((q, km, idx, torch.full((rows,), main, dtype=torch.int32, device=dev), (torch.randn(HEADS, device=dev)).to(torch.float32), dim, scale, extra_view, idy, torch.full((rows,), extra, dtype=torch.int32, device=dev)))
        S.sm89_dual_scope_decode(*calls[0])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        keep = []
        with torch.cuda.graph(g):
            for a in calls:
                keep.append(S.sm89_dual_scope_decode(*a))
        torch.cuda.synchronize()
        runs = []
        for _ in range(30):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            runs.append((time.perf_counter() - t0) * 1e6 / 40)
        return st.median(runs)

    def direct(extra_view, main, extra, rows):
        a = [call(rng, k_main, extra_view, main, extra, rows, "plain") for _ in range(6)]
        runs = []
        for q, km, idx, _, _, dim, scale, idy, _ in a:
            t0 = time.perf_counter()
            S.sm89_dual_scope_decode(q, km, idx, None, None, dim, scale, extra_view, idy, None)
            torch.cuda.synchronize()
            runs.append((time.perf_counter() - t0) * 1e3)
        return st.median(runs[1:])

    worst = 0.0
    for main, extra in ((512, 64), (768, 64), (768, 512), (1024, 512)):
        for rows in (1, 4, 8):
            a, b = timed(v4, main, extra, rows), timed(fp4, main, extra, rows)
            d = (b - a) * 8 / 1000
            if rows == 1:
                worst = max(worst, d)
            print(f"decode main {main} extra {extra} rows {rows}: V4 {a:7.1f} us, FP4 {b:7.1f} us, eight layers {d:+.3f} ms", flush=True)
    for main, extra in ((768, 512),):
        for rows in (256, 1024):
            a, b = direct(v4, main, extra, rows), direct(fp4, main, extra, rows)
            print(f"prefill main {main} extra {extra} rows {rows}: V4 {a:7.2f} ms, FP4 {b:7.2f} ms, {100 * (b / a - 1):+.1f}%", flush=True)
    for m in (1, 4096, 35840):
        xs, fs = latent(m, 8), freqs(m, 9)
        pages = -(-m // P)
        raw = torch.zeros(pages, FP4.page_bytes(P), dtype=torch.uint8, device=dev)
        rawv4 = torch.zeros(pages, KVLayout.V4.page_bytes(P), dtype=torch.uint8, device=dev)
        loc = torch.arange(m, device=dev, dtype=torch.int64)
        t = []
        for fn in (lambda: fused_store_cache(xs, raw, loc, page_size=P, type="flashmla", layout=FP4, freqs_cis=fs),
                   lambda: fused_store_cache(_rope_fq4(xs, fs, 64, compressed_kv=True), rawv4, loc, page_size=P, type="flashmla", layout=KVLayout.V4)):
            fn()
            torch.cuda.synchronize()
            rr = []
            for _ in range(20):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                rr.append((time.perf_counter() - t0) * 1e6)
            t.append(st.median(rr))
        print(f"write rows {m}: FP4 {t[0]:9.1f} us, V4 path {t[1]:9.1f} us", flush=True)
    report("price decode within 0.3 ms a token at one row", worst <= 0.3, f"worst {worst:+.3f} ms")


which = sys.argv[1:] or ["writer", "loader", "attention", "off", "price"]
for name in which:
    globals()[name]()
print("FAILS", fails, flush=True)
sys.exit(1 if fails else 0)
