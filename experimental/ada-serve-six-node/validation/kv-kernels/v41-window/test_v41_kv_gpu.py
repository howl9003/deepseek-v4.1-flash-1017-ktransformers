"""V41 window KV on the 4080: writer against DeepSeek's own act_quant, loader, attention, decode against prefill kernels, off arm, compile facts, price. No model.
argv: stages from writer loader attention pair off price. env BASE_SGLANG = a private copy of the served files, REF_KERNEL = the checkpoint's inference/kernel.py."""
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
os.environ["SGLANG_SM89_V41_KV"] = "1"

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention import sm89_dual_scope as S
from sglang.kernels.ops.attention import sm89_gather_staged as G
from sglang.kernels.ops.attention.dsv4 import dequant_k_cache as D
from sglang.kernels.ops.attention.dsv4 import triton_sparse_decode as _pkg  # noqa: F401
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.test.kernels.deepseek_v4 import dsv41_kv_quant_reference as PURE

dev = "cuda"
HEADS, DIM, V4_BYTES, V41, FP4 = 64, 512, 584, KVLayout.V41, KVLayout.V41_FP4
fails = []


def report(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + " " + str(detail), flush=True)
    if not ok:
        fails.append(name)


def latent(n, seed, edges=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(n, DIM, generator=g, device=dev) * torch.exp(torch.rand(n, 1, generator=g, device=dev) * 11 - 6)
    if edges:
        x[0, :32] = 0.0
        x[1] = 1e-6
        x[2, :32] = 40000.0
        x[3, ::2] = -0.0
        x[4, 32:64] = -1e-9
    return x.to(torch.bfloat16)


def freqs(n, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.polar(torch.ones(n, 32, device=dev), torch.rand(n, 32, generator=g, device=dev) * 2 * math.pi)


def cache(layout, x, f, P):
    """Pages from the real CUDA store, as the pool lays them out; the view the attention call receives."""
    n = x.shape[0]
    pages = -(-n // P)
    raw = torch.zeros(pages, layout.page_bytes(P), dtype=torch.uint8, device=dev)
    args = dict(page_size=P, type="flashmla", layout=layout)
    if layout is FP4:
        args["freqs_cis"] = f
    fused_store_cache(x, raw, torch.arange(n, device=dev, dtype=torch.int64), **args)
    return raw, raw[:, : P * layout.bytes_per_token].view(pages, P, 1, layout.bytes_per_token)


_REF = []


def reference_kernel():
    if not _REF:
        torch.set_default_dtype(torch.bfloat16)
        spec = importlib.util.spec_from_file_location("ref_kernel", os.environ["REF_KERNEL"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.set_default_dtype(torch.float32)
        _REF.append(module)
    return _REF[0]


def writer():
    """The store against DeepSeek's fused quant + dequant, the call `_window_kv` makes, and against the pure-torch quantizer."""
    act_quant = reference_kernel().act_quant
    bad_ref = bad_pure = signed = total = 0
    for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (1, 64), (7, 128), (65536, 128)) * 4):
        x = latent(n, seed, edges=n > 4)
        raw, _ = cache(V41, x, None, P)
        got = D.dequantize_k_cache_paged_dsv4(raw, torch.arange(n, device=dev), page_size=P, layout=V41)[:, 0]
        ref = x.unsqueeze(0).clone()
        act_quant(ref, 32, "ue8m0", torch.float8_e8m0fnu, True)
        ref = ref[0]
        pages = -(-n // P)
        padded = torch.zeros(pages * P, DIM, dtype=torch.bfloat16, device=dev)
        padded[:n] = x
        pure = PURE.dequantize_k_cache_v41(PURE.quantize_k_cache_v41(padded.view(pages, P, DIM)), P).reshape(-1, DIM)[:n].to(torch.bfloat16)
        a, b, c = got.view(torch.uint16).to(torch.int32), ref.view(torch.uint16).to(torch.int32), pure.view(torch.uint16).to(torch.int32)
        d = a != b
        zeros = d & ((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0)
        bad_ref += int((d & ~zeros).sum())
        signed += int(zeros.sum())
        bad_pure += int((got.to(torch.float32) != pure.to(torch.float32)).sum())
        total += a.numel()
        for r, col in (d & ~zeros).nonzero().tolist()[:6]:
            print(f"writer diff n {n} P {P} row {r} col {col}: got {float(got[r, col])} reference {float(ref[r, col])} x {float(x[r, col])}", flush=True)
        _ = c
    report("B store equals DeepSeek's act_quant", bad_ref == 0, f"{bad_ref} of {total}, signed-zero only {signed}")
    print(f"B against SGLang's pure-torch quantizer, which floors the quotient: {bad_pure} of {total}", flush=True)


def writer_fp4():
    """The FP4 store against DeepSeek's fused quant + dequant, the call the compressor's latents take after RoPE."""
    from sglang.srt.layers.attention.dsv4 import dsv41_sparse as DS

    fp4_act_quant = reference_kernel().fp4_act_quant
    bad = signed = total = 0
    for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (7, 128), (65536, 128)) * 4):
        x, f = latent(n, seed, edges=n > 4), freqs(n, seed + 100)
        raw, _ = cache(FP4, x, f, P)
        got = D.dequantize_k_cache_paged_dsv4(raw, torch.arange(n, device=dev), page_size=P, layout=FP4)[:, 0]
        ref = DS.rope_tail(x, f, 64).to(torch.bfloat16).unsqueeze(0).clone()
        fp4_act_quant(ref, 16, True, scale_dtype=torch.float8_e4m3fn)
        ref = ref[0]
        a, b = got.view(torch.uint16).to(torch.int32), ref.view(torch.uint16).to(torch.int32)
        d = a != b
        zeros = d & ((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0)
        bad += int((d & ~zeros).sum())
        signed += int(zeros.sum())
        total += a.numel()
        for r, col in (d & ~zeros).nonzero().tolist()[:4]:
            print(f"fp4 diff n {n} P {P} row {r} col {col}: got {float(got[r, col])} reference {float(ref[r, col])} x {float(x[r, col])}", flush=True)
    report("B2 FP4 store equals DeepSeek's fp4_act_quant", bad == 0, f"{bad} of {total}, signed-zero only {signed}")


@triton.jit
def _tile_kernel(KV, Idx, Out, n, block_size, stride_kv, tile, stride_o0, stride_o1):
    offs_n = tl.program_id(0) * 64 + tl.arange(0, 64)
    kv_block_base, data_off, scale_off, valid = _block_valid(KV, Idx, offs_n, n, 0, block_size, stride_kv, False, False, True)
    kv = v41_tile(kv_block_base, data_off, scale_off, valid[:, None], tile)
    tl.store(Out + offs_n[:, None] * stride_o0 + tl.arange(0, 64)[None, :] * stride_o1, kv, mask=(offs_n < n)[:, None])


def loader():
    globals()["_block_valid"], globals()["v41_tile"] = G._block_valid, G.v41_tile
    bad = total = 0
    for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (640, 128))):
        x = latent(n, seed, edges=True)
        raw, _ = cache(V41, x, None, P)
        idx = torch.randperm(n, device=dev).to(torch.int32)
        idx[torch.rand(n, device=dev) < 0.1] = -1
        want = D.dequantize_k_cache_paged_dsv4(raw, idx.clamp(min=0), page_size=P, layout=V41)[:, 0]
        want = torch.where((idx >= 0)[:, None], want, torch.zeros_like(want)).view(torch.uint16)
        out = torch.zeros(n, DIM, dtype=torch.bfloat16, device=dev)
        for tile in range(8):
            t = torch.zeros(n, 64, dtype=torch.bfloat16, device=dev)
            _tile_kernel[(-(-n // 64),)](raw, idx, t, n, P, raw.stride(0), tile, t.stride(0), t.stride(1))
            out[:, tile * 64:tile * 64 + 64] = t
        d = out.view(torch.uint16) != want
        bad += int(d.sum())
        total += d.numel()
    report("C loader equals the V41 dequant", bad == 0, f"{bad} of {total}")


def twin(path, name, **globs):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    for k, v in globs.items():
        setattr(m, k, v)
    return m


@triton.jit
def oracle_v41(kv_block_base, data_off, scale_off, valid_2d, tile):
    """The same tile from a BF16 row: the byte loads stay, so the dot's kWidth does. Rows sit at 2 x the V41 data stride."""
    offs = tl.arange(0, 64)
    ptrs = kv_block_base[:, None] + (data_off * 2)[:, None] + tile * 128 + offs[None, :] * 2
    lo = tl.load(ptrs, mask=valid_2d, other=0).to(tl.uint16)
    hi = tl.load(ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
    kv = (lo | (hi << 8)).to(tl.bfloat16, bitcast=True)
    return tl.where(valid_2d, kv, 0.0)


@triton.jit
def oracle_fp4(kv_block_base, data_off, scale_off, valid_2d, tile):
    """As oracle_v41, for FP4 rows at 4 x the FP4 data stride."""
    offs = tl.arange(0, 64)
    ptrs = kv_block_base[:, None] + (data_off * 4)[:, None] + tile * 128 + offs[None, :] * 2
    lo = tl.load(ptrs, mask=valid_2d, other=0).to(tl.uint16)
    hi = tl.load(ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
    kv = (lo | (hi << 8)).to(tl.bfloat16, bitcast=True)
    return tl.where(valid_2d, kv, 0.0)


def bf16_rows(layout, x, f, P):
    """The BF16 values of a cache, one row per token, in a cache view with the layout's shape and a 1024 B row stride."""
    n = x.shape[0]
    pages = -(-n // P)
    v = D.dequantize_k_cache_paged_dsv4(cache(layout, x, f, P)[0], torch.arange(n, device=dev), page_size=P, layout=layout)[:, 0]
    raw = torch.zeros(pages, P * 1024, dtype=torch.uint8, device=dev)
    raw.view(-1)[: n * 1024] = v.view(torch.uint8).reshape(-1)
    return raw, raw[:, : P * layout.bytes_per_token].view(pages, P, 1, layout.bytes_per_token)


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


def real_caches():
    P = 128
    n = 512 * P
    x, xe, f = latent(n, 2, edges=True), latent(n, 4, edges=True), freqs(n, 3)
    return (x, xe, f, P), cache(V41, x, None, P)[1], cache(FP4, xe, f, P)[1]


def attention():
    """The V41 kernels against twins reading the same values as BF16 rows: staged at 1-8 rows, unsplit above."""
    root = Path(G.__file__).parent
    G2 = twin(root / "sm89_gather_staged.py", "gather_oracle", v41_tile=oracle_v41, fp4_tile=oracle_fp4)
    F2 = twin(root / "dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py", "fused_oracle", v41_tile=oracle_v41, fp4_tile=oracle_fp4)
    (x, xe, f, P), v41, fp4 = real_caches()
    _, main_rows = bf16_rows(V41, x, None, P)
    _, extra_rows = bf16_rows(FP4, xe, f, P)
    rng = random.Random(int(os.environ.get("FUZZ_SEED", "1017")))
    bad = total = 0
    seen = {}
    for it in range(int(os.environ.get("FUZZ_N", "3000"))):
        main = rng.choice((64, 128, 256, 100, 63))
        extra = rng.choice((64, 128, 256, 512, 65))
        rows = rng.choice((1, 1, 1, 2, 4, 8, 16, 64, 256))
        mode = rng.choice(("plain", "plain", "hot", "holes", "nolen", "nan"))
        q, km, idx, tl_main, sink, dim, scale, idy, tl_extra = call(rng, v41, fp4, main, extra, rows, mode)
        a = S.sm89_dual_scope_decode(q, km, idx, tl_main, sink, dim, scale, fp4, idy, tl_extra)
        if rows <= 8:
            b = G2.decode(q, main_rows, idx.reshape(rows, -1).contiguous(), tl_main, sink, scale, extra_rows, idy.reshape(rows, -1).contiguous(), tl_extra).unsqueeze(1)
        else:
            saved = S._F
            S._F = F2
            b = S.sm89_dual_scope_decode(q, main_rows, idx, tl_main, sink, dim, scale, extra_rows, idy, tl_extra)
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
    import triton.compiler.compiler as CC  # noqa: F401

    seen_f = set()
    for mod, names in ((G, ("_scores_kernel", "_output_kernel", "_probabilities_kernel")), (S._F, ("_fused_gather_attn_dsv4_dual_scope_kernel",))):
        for nm in names:
            fn = getattr(mod, nm)
            fn = fn if hasattr(fn, "device_caches") else fn.fn
            for c in getattr(fn, "device_caches", {}).values():
                for k in c[0].values():
                    m = getattr(k, "metadata", None)
                    seen_f.add((nm, getattr(m, "n_regs", None), getattr(m, "n_spills", None), getattr(m, "shared", None), getattr(k, "n_regs", None), getattr(k, "n_spills", None)))
    for row in sorted(seen_f, key=str):
        print("F", row, flush=True)


def pair():
    """Decode runs staged and prefill unsplit: the same call through both must give the same bits."""
    (_, _, _, _), v41, fp4 = real_caches()
    rng = random.Random(7)
    bad = 0
    N = int(os.environ.get("PAIR_N", "1000"))
    staged = S._STAGED_ROWS
    for it in range(N):
        main, extra = rng.choice((128, 64, 100)), rng.choice((64, 256, 512))
        rows = rng.choice((1, 2, 4, 8))
        mode = rng.choice(("plain", "hot", "holes", "nolen"))
        q, km, idx, tl_main, sink, dim, scale, idy, tl_extra = call(rng, v41, fp4, main, extra, rows, mode)
        S._STAGED_ROWS = staged
        a = S.sm89_dual_scope_decode(q, km, idx, tl_main, sink, dim, scale, fp4, idy, tl_extra)
        S._STAGED_ROWS = 0
        b = S.sm89_dual_scope_decode(q, km, idx, tl_main, sink, dim, scale, fp4, idy, tl_extra)
        S._STAGED_ROWS = staged
        torch.cuda.synchronize()
        bad += int(not torch.equal(a.view(torch.uint16), b.view(torch.uint16)))
    report("D2 staged equals unsplit on real V41 and FP4 caches", bad == 0, f"{bad} of {N}")


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


def head():
    """The head at one and four rows, BF16 logits against FP32, as the logits processor calls it. Real shape, random values."""
    import json

    from sglang.kernels.ops.gemm.fixed_lm_head import fixed_lm_head

    vocab = json.load(open(Path(os.environ["REF_KERNEL"]).parent / "config.json"))["vocab_size"]
    w = (torch.randn(vocab, 5120, device=dev) * 0.02).to(torch.bfloat16)
    for rows in (1, 4):
        h = torch.randn(rows, 5120, device=dev).to(torch.bfloat16)
        t = {}
        for fp32 in (False, True):
            fixed_lm_head(h, w, fp32_output=fp32)
            torch.cuda.synchronize()
            rr = []
            for _ in range(40):
                t0 = time.perf_counter()
                fixed_lm_head(h, w, fp32_output=fp32)
                torch.cuda.synchronize()
                rr.append((time.perf_counter() - t0) * 1e3)
            t[fp32] = st.median(rr)
        print(f"head rows {rows}: BF16 {t[False]:.3f} ms, FP32 {t[True]:.3f} ms, {t[True] - t[False]:+.3f} ms", flush=True)


def price():
    """One call as a layer makes it: 128 window rows plus the compressed top-k, V4 against V41 main, both next to FP4 compressed KV. 40 layers a token."""
    P = 128
    n = 4096 * P
    x, xe, f = latent(n, 5), latent(n, 8), freqs(n, 6)
    _, v41 = cache(V41, x, None, P)
    _, fp4 = cache(FP4, xe, f, P)
    v4 = v4_cache(4096, P, 7)
    rng = random.Random(1)

    def timed(main_view, main, extra, rows):
        calls = []
        for _ in range(40):
            q, km, idx, _, _, dim, scale, idy, _ = call(rng, main_view, fp4, main, extra, rows, "plain")
            calls.append((q, km, idx, torch.full((rows,), main, dtype=torch.int32, device=dev), (torch.randn(HEADS, device=dev)).to(torch.float32), dim, scale, fp4, idy, torch.full((rows,), extra, dtype=torch.int32, device=dev)))
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

    def direct(main_view, main, extra, rows):
        a = [call(rng, main_view, fp4, main, extra, rows, "plain") for _ in range(6)]
        runs = []
        for q, km, idx, _, _, dim, scale, idy, _ in a:
            t0 = time.perf_counter()
            S.sm89_dual_scope_decode(q, km, idx, None, None, dim, scale, fp4, idy, None)
            torch.cuda.synchronize()
            runs.append((time.perf_counter() - t0) * 1e3)
        return st.median(runs[1:])

    worst = 0.0
    for main, extra in ((128, 64), (128, 256), (128, 512)):
        for rows in (1, 4, 8):
            a, b = timed(v4, main, extra, rows), timed(v41, main, extra, rows)
            d = (b - a) * 40 / 1000
            if rows == 1:
                worst = max(worst, d)
            print(f"decode main {main} extra {extra} rows {rows}: V4 {a:7.1f} us, V41 {b:7.1f} us, 40 layers {d:+.3f} ms", flush=True)
    for main, extra in ((128, 512),):
        for rows in (256, 1024):
            a, b = direct(v4, main, extra, rows), direct(v41, main, extra, rows)
            print(f"prefill main {main} extra {extra} rows {rows}: V4 {a:7.2f} ms, V41 {b:7.2f} ms, {100 * (b / a - 1):+.1f}%", flush=True)
    for m in (1, 4096, 35840):
        xs = latent(m, 8)
        pages = -(-m // P)
        loc = torch.arange(m, device=dev, dtype=torch.int64)
        t = []
        for layout in (V41, KVLayout.V4):
            raw = torch.zeros(pages, layout.page_bytes(P), dtype=torch.uint8, device=dev)
            fused_store_cache(xs, raw, loc, page_size=P, type="flashmla", layout=layout)
            torch.cuda.synchronize()
            rr = []
            for _ in range(20):
                t0 = time.perf_counter()
                fused_store_cache(xs, raw, loc, page_size=P, type="flashmla", layout=layout)
                torch.cuda.synchronize()
                rr.append((time.perf_counter() - t0) * 1e6)
            t.append(st.median(rr))
        print(f"write rows {m}: V41 {t[0]:9.1f} us, V4 {t[1]:9.1f} us", flush=True)
    report("price decode within 0.3 ms a token at one row", worst <= 0.3, f"worst {worst:+.3f} ms")


which = sys.argv[1:] or ["writer", "writer_fp4", "loader", "attention", "pair", "off", "price"]
for name in which:
    globals()[name]()
print("FAILS", fails, flush=True)
sys.exit(1 if fails else 0)
