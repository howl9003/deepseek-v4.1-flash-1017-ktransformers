"""GPU checks for tier 0's SM89 KV layouts (14-sm89-fp4-compressed-kv, 15-sm89-v41-window-kv); see docs/sm89.md.

No model is loaded. Run on an idle SM89 card after setup.sh, from the repository:

  PYTHONPATH=third_party/sglang/python .venv/bin/python -B tools/sm89-kv-layout-test.py \
      --model <checkpoint dir> [--base <unpatched sglang package>] [stage ...]

stages (default all but price and head):
  writer     B: the V41 store against the checkpoint's act_quant (block 32, UE8M0), the call
             `_window_kv` makes; B2: the V41_FP4 store against fp4_act_quant after RoPE
  loader     C: the readers' V41 / V41_FP4 tiles against SGLang's dequantization
  attention  D: the dual-scope kernels on V41 / V41_FP4 caches against twins that read the same
             values as BF16 rows, in every split: auto, 0 (unsplit), 2, 4, 8, and the staged
             gather when the optional patch is applied; F: registers and spills of what compiled
  pair       D2: staged gather against the unsplit kernel (optional patch only)
  off        E: V4 caches through the patched kernels against --base, the files before the patches
  price      decode under CUDA graph replay, V4 against V41 + V41_FP4, per split; prefill; writes
  head       the lm_head at one and four rows, BF16 logits against --enable-fp32-lm-head's FP32

--model gives inference/kernel.py (act_quant, fp4_act_quant) and config.json.
"""

import argparse
import importlib.util
import math
import os
import random
import statistics as st
import sys
import time
import types
from pathlib import Path

os.environ["SGLANG_SM89_V41_KV"] = "1"  # implies SGLANG_SM89_FP4_COMPRESSED_KV; read at import

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention import sm89_dual_scope as S
from sglang.kernels.ops.attention.dsv4 import dequant_k_cache as D
from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.sm89_fp4_tile import FP4_DATA_BYTES, FP4_SCALE_BYTES, fp4_tile
from sglang.kernels.ops.attention.sm89_v41_tile import V41_DATA_BYTES, V41_SCALE_BYTES, v41_tile

try:
    from sglang.kernels.ops.attention import sm89_gather_staged as G
except ImportError:  # the optional patch is not applied
    G = None

dev = "cuda"
HEADS, DIM = 64, 512
V4, V41, FP4 = KVLayout.V4, KVLayout.V41, KVLayout.V41_FP4
ARGS = None
fails = []
# SGLang's dequantization refuses SM89 (nothing on SM89 serves through it); its kernels are
# portable Triton, so the checks call them as the reference.
D.current_arch = lambda: types.SimpleNamespace(is_sm_at_least=lambda n: True)


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
    """Pages from the real CUDA store, as the pool lays them out, and the view attention receives."""
    n = x.shape[0]
    pages = -(-n // P)
    raw = torch.zeros(pages, layout.page_bytes(P), dtype=torch.uint8, device=dev)
    kw = dict(page_size=P, type="flashmla", layout=layout)
    if layout is FP4:
        kw["freqs_cis"] = f
    fused_store_cache(x, raw, torch.arange(n, device=dev, dtype=torch.int64), **kw)
    return raw, raw[:, : P * layout.bytes_per_token].view(pages, P, 1, layout.bytes_per_token)


def dequant(raw, idx, P, layout):
    return D.dequantize_k_cache_paged_dsv4(raw, idx, page_size=P, layout=layout)[:, 0]


_REF = []


def reference_kernel():
    if not _REF:
        torch.set_default_dtype(torch.bfloat16)
        spec = importlib.util.spec_from_file_location("ref_kernel", Path(ARGS.model) / "inference/kernel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.set_default_dtype(torch.float32)
        _REF.append(module)
    return _REF[0]


def bits_differ(got, ref):
    a, b = got.view(torch.uint16).to(torch.int32), ref.view(torch.uint16).to(torch.int32)
    d = a != b
    zeros = d & ((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0)
    return d & ~zeros, zeros


def writer():
    act_quant = reference_kernel().act_quant
    bad = floor = signed = total = 0
    for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (1, 64), (7, 128), (65536, 128)) * 4):
        x = latent(n, seed, edges=n > 4)
        raw, _ = cache(V41, x, None, P)
        got = dequant(raw, torch.arange(n, device=dev), P, V41)
        ref = x.unsqueeze(0).clone()
        act_quant(ref, 32, "ue8m0", torch.float8_e8m0fnu, True)
        d, z = bits_differ(got, ref[0])
        # SGLang floors the scale after dividing by 448 (2^-13), the checkpoint floors amax at
        # 1e-4 first (2^-22): only blocks whose largest value is under 0.0448 can differ.
        tiny = (x.float().abs().view(n, 16, 32).amax(-1) < 1e-4 * 448).repeat_interleave(32, 1)
        bad += int((d & ~tiny).sum())
        floor += int((d & tiny).sum())
        signed += int(z.sum())
        total += d.numel()
        for r, c in (d & ~tiny).nonzero().tolist()[:4]:
            print(f"writer diff n {n} P {P} row {r} col {c}: got {float(got[r, c])} reference {float(ref[0, r, c])}", flush=True)
    report("B V41 store equals act_quant", bad == 0,
           f"{bad} of {total} outside the scale floor; {floor} in blocks under it (sgl-project/sglang#48500); signed zero only {signed}")

    from sglang.srt.layers.attention.dsv4 import dsv41_sparse as DS

    fp4_act_quant = reference_kernel().fp4_act_quant
    bad = signed = total = 0
    for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (7, 128), (65536, 128)) * 4):
        x, f = latent(n, seed, edges=n > 4), freqs(n, seed + 100)
        raw, _ = cache(FP4, x, f, P)
        got = dequant(raw, torch.arange(n, device=dev), P, FP4)
        ref = DS.rope_tail(x, f, 64).to(torch.bfloat16).unsqueeze(0).clone()
        fp4_act_quant(ref, 16, True, scale_dtype=torch.float8_e4m3fn)
        d, z = bits_differ(got, ref[0])
        bad += int(d.sum())
        signed += int(z.sum())
        total += d.numel()
    report("B2 V41_FP4 store equals fp4_act_quant after RoPE", bad == 0, f"{bad} of {total}, signed zero only {signed}")


@triton.jit
def _tile_kernel(KV, Idx, Out, n, block_size, stride_kv, tile, stride_o0, stride_o1, IS_FP4: tl.constexpr):
    offs_n = tl.program_id(0) * 64 + tl.arange(0, 64)
    mask = offs_n < n
    idx = tl.load(Idx + offs_n, mask=mask, other=-1)
    valid_2d = (mask & (idx >= 0))[:, None]
    i = tl.maximum(idx, 0)
    base = KV + (i // block_size).to(tl.int64) * tl.cast(stride_kv, tl.int64)
    off = (i % block_size).to(tl.int64)
    if IS_FP4:
        kv = fp4_tile(base, off * FP4_DATA_BYTES, block_size * FP4_DATA_BYTES + off * FP4_SCALE_BYTES, valid_2d, tile)
    else:
        kv = v41_tile(base, off * V41_DATA_BYTES, block_size * V41_DATA_BYTES + off * V41_SCALE_BYTES, valid_2d, tile)
    tl.store(Out + offs_n[:, None] * stride_o0 + tl.arange(0, 64)[None, :] * stride_o1, kv, mask=mask[:, None])


def loader():
    for layout in (V41, FP4):
        bad = total = 0
        for seed, (n, P) in enumerate(((4096, 64), (4096, 128), (640, 128))):
            x = latent(n, seed, edges=True)
            raw, _ = cache(layout, x, freqs(n, seed + 7), P)
            idx = torch.randperm(n, device=dev).to(torch.int32)
            idx[torch.rand(n, device=dev) < 0.1] = -1
            want = dequant(raw, idx.clamp(min=0).to(torch.int64), P, layout)
            want = torch.where((idx >= 0)[:, None], want, torch.zeros_like(want)).view(torch.uint16)
            out = torch.zeros(n, DIM, dtype=torch.bfloat16, device=dev)
            for tile in range(8):
                t = torch.zeros(n, 64, dtype=torch.bfloat16, device=dev)
                _tile_kernel[(-(-n // 64),)](raw, idx, t, n, P, raw.stride(0), tile, t.stride(0), t.stride(1), IS_FP4=layout is FP4)
                out[:, tile * 64 : tile * 64 + 64] = t
            d = out.view(torch.uint16) != want
            bad += int(d.sum())
            total += d.numel()
        report(f"C {layout.value} tiles equal SGLang's dequantization", bad == 0, f"{bad} of {total}")


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
    """The same tile from a BF16 row (the byte loads stay, so the dot's kWidth does); rows at 2 x the V41 data stride."""
    offs = tl.arange(0, 64)
    ptrs = kv_block_base[:, None] + (data_off * 2)[:, None] + tile * 128 + offs[None, :] * 2
    lo = tl.load(ptrs, mask=valid_2d, other=0).to(tl.uint16)
    hi = tl.load(ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
    return tl.where(valid_2d, (lo | (hi << 8)).to(tl.bfloat16, bitcast=True), 0.0)


@triton.jit
def oracle_fp4(kv_block_base, data_off, scale_off, valid_2d, tile):
    """As oracle_v41, for rows at 4 x the V41_FP4 data stride."""
    offs = tl.arange(0, 64)
    ptrs = kv_block_base[:, None] + (data_off * 4)[:, None] + tile * 128 + offs[None, :] * 2
    lo = tl.load(ptrs, mask=valid_2d, other=0).to(tl.uint16)
    hi = tl.load(ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)
    return tl.where(valid_2d, (lo | (hi << 8)).to(tl.bfloat16, bitcast=True), 0.0)


def bf16_rows(layout, x, f, P):
    """A cache's BF16 values, one 1,024-byte row a token, in a view with the layout's shape."""
    n = x.shape[0]
    pages = -(-n // P)
    v = dequant(cache(layout, x, f, P)[0], torch.arange(n, device=dev), P, layout)
    raw = torch.zeros(pages, P * 1024, dtype=torch.uint8, device=dev)
    raw.view(-1)[: n * 1024] = v.view(torch.uint8).reshape(-1)
    return raw[:, : P * layout.bytes_per_token].view(pages, P, 1, layout.bytes_per_token)


def v4_cache(blocks, block_size, seed, spread=1):
    g = torch.Generator(device=dev).manual_seed(seed)
    nope = (torch.randint(0x20, 0x48 + 0x20 * spread, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8)
            | (torch.randint(0, 2, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8) << 7))
    rope = (torch.randn(blocks, block_size, 64, generator=g, device=dev) * 0.05 * spread).to(torch.bfloat16).view(torch.uint8).reshape(blocks, block_size, 128)
    row = torch.cat((nope, rope), dim=2).reshape(blocks, -1)
    scales = torch.randint(118, 130, (blocks, block_size, 8), generator=g, device=dev, dtype=torch.uint8).reshape(blocks, -1)
    return torch.cat((row, scales), dim=1).reshape(blocks, block_size, 1, 584)


def call(rng, k_main, k_extra, main, extra, rows, mode):
    q = (torch.randn(rows, HEADS, DIM, device=dev) * rng.choice((0.05, 0.3, 1.0, 4.0))).to(torch.bfloat16)
    if mode == "hot":
        q = q * 8
    if mode == "nan":
        q[0, 0, 0] = float("nan")
    tokens = k_main.shape[0] * k_main.shape[1]
    idx = torch.stack([torch.randperm(tokens, device=dev)[:main] for _ in range(rows)]).to(torch.int32)
    if mode == "holes":
        idx[torch.rand(idx.shape, device=dev) < 0.2] = -1
    lengths = [rng.choice((main, main, main // 2, 1, 0, max(main - 70, 0))) for _ in range(rows)]
    tl_main = None if mode == "nolen" else torch.tensor(lengths, dtype=torch.int32, device=dev)
    sink = None if rng.random() < 0.3 else (torch.randn(HEADS, device=dev) * 2).to(torch.float32)
    if k_extra is None:
        return q, idx, tl_main, sink, None, None
    tokens = k_extra.shape[0] * k_extra.shape[1]
    idy = torch.stack([torch.randperm(tokens, device=dev)[:extra] for _ in range(rows)]).to(torch.int32)
    tl_extra = None if mode == "nolen" else torch.tensor([rng.choice((extra, extra // 2, 0)) for _ in range(rows)], dtype=torch.int32, device=dev)
    return q, idx, tl_main, sink, idy, tl_extra


def run(scope, arm, q, km, idx, tl_main, sink, ke, idy, tl_extra, staged=None):
    """One call through `scope` (sm89_dual_scope or a twin) in split arm `arm`; "staged" with `staged`."""
    rows = q.shape[0]
    if arm == "staged" and rows <= 8 and staged is not None:
        return staged.decode(q, km, idx.reshape(rows, -1).contiguous(), tl_main, sink, DIM ** -0.5, ke,
                             None if idy is None else idy.reshape(rows, -1).contiguous(), tl_extra).unsqueeze(1)
    scope._SPLITK_MODE = "auto" if arm == "staged" else arm
    return scope.sm89_dual_scope_decode(q, km, idx, tl_main, sink, DIM, DIM ** -0.5, ke, idy, tl_extra)


ARMS = ["auto", "0", "2", "4", "8"] + (["staged"] if G is not None else [])


def set_staged(on):
    if hasattr(S, "_STAGED_ROWS"):
        S._STAGED_ROWS = 8 if on else 0


def real_caches():
    P = 128
    n = 512 * P
    x, xe, f = latent(n, 2, edges=True), latent(n, 4, edges=True), freqs(n, 3)
    return (x, xe, f, P), cache(V41, x, None, P)[1], cache(FP4, xe, f, P)[1]


def attention():
    root = Path(S.__file__).parent
    F, F2 = S._F, twin(root / "dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py", "fused_oracle", v41_tile=oracle_v41, fp4_tile=oracle_fp4)
    G2 = twin(root / "sm89_gather_staged.py", "gather_oracle", v41_tile=oracle_v41, fp4_tile=oracle_fp4) if G is not None else None
    (x, xe, f, P), v41, fp4 = real_caches()
    main_rows, extra_rows = bf16_rows(V41, x, None, P), bf16_rows(FP4, xe, f, P)
    v4 = v4_cache(512, P, 9)
    n_calls = int(os.environ.get("FUZZ_N", "1000"))
    for arm in ARMS:
        rng = random.Random(1017)
        bad = 0
        seen = {}
        for it in range(n_calls):
            main = rng.choice((64, 128, 256, 100, 63))
            extra = rng.choice((64, 128, 256, 512, 65))
            rows = rng.choice((1, 1, 1, 2, 4, 6, 8, 16, 64, 256))
            mode = rng.choice(("plain", "plain", "hot", "holes", "nolen", "nan"))
            # V41 main + V41_FP4 extra (SGLANG_SM89_V41_KV), V4 main + V41_FP4 extra
            # (SGLANG_SM89_FP4_COMPRESSED_KV alone), or a window-only layer on V41
            kind = rng.choice(("v41", "v41", "v41", "fp4", "swa"))
            km, km_rows = (v4, v4) if kind == "fp4" else (v41, main_rows)
            ke, ke_rows = (None, None) if kind == "swa" else (fp4, extra_rows)
            q, idx, tl_main, sink, idy, tl_extra = call(rng, km, ke, main, extra, rows, mode)
            set_staged(arm == "staged")
            S._F = F
            a = run(S, arm, q, km, idx, tl_main, sink, ke, idy, tl_extra)
            set_staged(False)
            S._F = F2
            b = run(S, arm, q, km_rows, idx, tl_main, sink, ke_rows, idy, tl_extra, staged=G2)
            S._F = F
            torch.cuda.synchronize()
            seen[kind] = seen.get(kind, 0) + 1
            if not torch.equal(a.view(torch.uint16), b.view(torch.uint16)):
                bad += 1
                if bad <= 6:
                    d = a.view(torch.uint16) != b.view(torch.uint16)
                    print(f"mismatch arm {arm} call {it}: {kind} main {main} extra {extra} rows {rows} {mode}: {int(d.sum())} of {d.numel()}", flush=True)
        report(f"D split {arm}: attention equals the BF16-row twin", bad == 0, f"{bad} of {n_calls} calls {seen}")
    S._SPLITK_MODE = "auto"
    compiled()


def compiled():
    rows = set()
    mods = [(S._F, ("_fused_gather_attn_dsv4_dual_scope_kernel", "_fused_gather_attn_dsv4_dual_scope_splitk_kernel"))]
    if G is not None:
        mods.append((G, ("_scores_kernel", "_probabilities_kernel", "_output_kernel")))
    for mod, names in mods:
        for nm in names:
            fn = getattr(mod, nm)
            fn = getattr(fn, "fn", fn)
            for c in getattr(fn, "device_caches", {}).values():
                for k in c[0].values():
                    m = getattr(k, "metadata", None)
                    rows.add((nm, getattr(k, "n_regs", None), getattr(k, "n_spills", None), getattr(m, "shared", None)))
    for r in sorted(rows, key=str):
        print("F", r, flush=True)
    if not rows:
        print("F: this Triton does not expose its compiled variants; registers and spills not read", flush=True)
        return
    spills = [r for r in rows if r[2]]
    report("F no spills", not spills, f"{len(rows)} compiled variants (kernel, registers, spills, shared), spills {spills}")


def pair():
    if G is None:
        print("pair: the optional staged-gather patch is not applied; skipped", flush=True)
        return
    (_, _, _, P), v41, fp4 = real_caches()
    v4 = (v4_cache(512, P, 11), v4_cache(512, P, 12))
    rng = random.Random(7)
    bad = 0
    n_calls = int(os.environ.get("PAIR_N", "1000"))
    for it in range(n_calls):
        main, extra = rng.choice((128, 64, 100)), rng.choice((64, 256, 512))
        rows = rng.choice((1, 2, 4, 8))
        mode = rng.choice(("plain", "hot", "holes", "nolen"))
        km, ke = (v41, fp4) if rng.random() < 0.7 else v4
        q, idx, tl_main, sink, idy, tl_extra = call(rng, km, ke, main, extra, rows, mode)
        set_staged(True)
        a = run(S, "staged", q, km, idx, tl_main, sink, ke, idy, tl_extra)
        set_staged(False)
        b = run(S, "0", q, km, idx, tl_main, sink, ke, idy, tl_extra)
        torch.cuda.synchronize()
        bad += int(not torch.equal(a.view(torch.uint16), b.view(torch.uint16)))
    S._SPLITK_MODE = "auto"
    report("D2 staged gather equals the unsplit kernel", bad == 0, f"{bad} of {n_calls}")


def off():
    if ARGS.base is None:
        print("off: no --base; skipped", flush=True)
        return
    base = Path(ARGS.base)
    F0 = twin(base / "kernels/ops/attention/dsv4/triton_sparse_decode/triton_mla_kernels_decode_fused.py", "fused_base")
    S0 = twin(base / "kernels/ops/attention/sm89_dual_scope.py", "scope_base", _F=F0)
    k_main, k_extra = v4_cache(2048, 128, 1), v4_cache(512, 128, 2)
    n_calls = int(os.environ.get("OFF_N", "300"))
    set_staged(False)
    for arm in ("auto", "0", "2", "4", "8"):
        rng = random.Random(1)
        bad = 0
        for it in range(n_calls):
            main, extra = rng.choice((64, 256, 512, 1024)), rng.choice((64, 128, 256))
            rows = rng.choice((1, 2, 4, 8, 16, 64))
            mode = rng.choice(("plain", "hot", "holes", "nolen"))
            q, idx, tl_main, sink, idy, tl_extra = call(rng, k_main, k_extra, main, extra, rows, mode)
            a = run(S, arm, q, k_main, idx, tl_main, sink, k_extra, idy, tl_extra)
            b = run(S0, arm, q, k_main, idx, tl_main, sink, k_extra, idy, tl_extra)
            torch.cuda.synchronize()
            bad += int(not torch.equal(a.view(torch.uint16), b.view(torch.uint16)))
        report(f"E split {arm}: V4 caches equal the files before the patches", bad == 0, f"{bad} of {n_calls}")
    S._SPLITK_MODE = "auto"


def price():
    """One call as a layer makes it: 128 window rows plus the compressed top-k. 40 layers a token."""
    P = 128
    n = 4096 * P
    x, xe, f = latent(n, 5), latent(n, 8), freqs(n, 6)
    _, v41 = cache(V41, x, None, P)
    _, fp4 = cache(FP4, xe, f, P)
    v4m, v4e = v4_cache(4096, P, 7), v4_cache(4096, P, 8)
    rng = random.Random(1)

    def timed(arm, km, ke, main, extra, rows):
        calls = []
        for _ in range(40):
            q, idx, _, _, idy, _ = call(rng, km, ke, main, extra, rows, "plain")
            calls.append((q, km, idx, torch.full((rows,), main, dtype=torch.int32, device=dev), torch.randn(HEADS, device=dev),
                          ke, idy, torch.full((rows,), extra, dtype=torch.int32, device=dev)))
        set_staged(arm == "staged")
        S._SPLITK_MODE = "auto" if arm == "staged" else arm
        go = lambda c: S.sm89_dual_scope_decode(c[0], c[1], c[2], c[3], c[4], DIM, DIM ** -0.5, c[5], c[6], c[7])  # noqa: E731
        go(calls[0])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        keep = []
        with torch.cuda.graph(g):
            for c in calls:
                keep.append(go(c))
        torch.cuda.synchronize()
        runs = []
        for _ in range(30):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            runs.append((time.perf_counter() - t0) * 1e6 / 40)
        set_staged(False)
        return st.median(runs)

    arms = ["auto", "0"] + (["staged"] if G is not None else [])
    print("decode, us a call (40 layers a token): V4 + V4 | V4 + V41_FP4 | V41 + V41_FP4, per split", flush=True)
    for main, extra in ((128, 64), (128, 256), (128, 512)):
        for rows in (1, 4, 8):
            for arm in arms:
                a, b, c = timed(arm, v4m, v4e, main, extra, rows), timed(arm, v4m, fp4, main, extra, rows), timed(arm, v41, fp4, main, extra, rows)
                print(f"  main {main} extra {extra} rows {rows} split {arm:6s}: {a:6.1f} | {b:6.1f} | {c:6.1f}"
                      f"   V41 + FP4 against V4: {(c - a) * 40 / 1000:+.3f} ms a token", flush=True)
    S._SPLITK_MODE = "auto"
    for rows in (256, 1024):
        t = []
        for km, ke in ((v4m, v4e), (v41, fp4)):
            q, idx, _, _, idy, _ = call(rng, km, ke, 128, 512, rows, "plain")
            rr = []
            for _ in range(6):
                t0 = time.perf_counter()
                S.flash_mla_sparse_sm89(q, km, idx, None, None, DIM, DIM ** -0.5, ke, idy, None)
                torch.cuda.synchronize()
                rr.append((time.perf_counter() - t0) * 1e3)
            t.append(st.median(rr[1:]))
        print(f"prefill main 128 extra 512 rows {rows}: V4 {t[0]:.2f} ms, V41 + FP4 {t[1]:.2f} ms, {100 * (t[1] / t[0] - 1):+.1f}%", flush=True)
    for m in (1, 4096, 35840):
        xs, fs = latent(m, 8), freqs(m, 9)
        loc = torch.arange(m, device=dev, dtype=torch.int64)
        t = []
        for layout in (V4, V41, FP4):
            raw = torch.zeros(-(-m // P), layout.page_bytes(P), dtype=torch.uint8, device=dev)
            kw = dict(page_size=P, type="flashmla", layout=layout, **({"freqs_cis": fs} if layout is FP4 else {}))
            fused_store_cache(xs, raw, loc, **kw)
            torch.cuda.synchronize()
            rr = []
            for _ in range(20):
                t0 = time.perf_counter()
                fused_store_cache(xs, raw, loc, **kw)
                torch.cuda.synchronize()
                rr.append((time.perf_counter() - t0) * 1e6)
            t.append(st.median(rr))
        print(f"write rows {m}: V4 {t[0]:.1f} us, V41 {t[1]:.1f} us, V41_FP4 {t[2]:.1f} us", flush=True)


def head():
    """The logits processor's two products at the checkpoint's shape, random values."""
    import json

    cfg = json.load(open(Path(ARGS.model) / "config.json"))
    vocab, dim = cfg["vocab_size"], cfg.get("dim") or cfg["hidden_size"]
    w = (torch.randn(vocab, dim, device=dev) * 0.02).to(torch.bfloat16)
    for rows in (1, 4):
        h = torch.randn(rows, dim, device=dev).to(torch.bfloat16)
        t = {}
        for name, fn in (("BF16", lambda: torch.matmul(h, w.T)), ("FP32", lambda: torch.mm(h, w.T, out_dtype=torch.float32))):
            fn()
            torch.cuda.synchronize()
            rr = []
            for _ in range(40):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                rr.append((time.perf_counter() - t0) * 1e3)
            t[name] = st.median(rr)
        print(f"head rows {rows}: BF16 logits {t['BF16']:.3f} ms, FP32 logits {t['FP32']:.3f} ms", flush=True)
    del w


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory (inference/kernel.py, config.json)")
    ap.add_argument("--base", help="the sglang package before the patches, for the off stage")
    ap.add_argument("stages", nargs="*", default=["writer", "loader", "attention", "pair", "off"])
    ARGS = ap.parse_args()
    assert torch.cuda.get_device_capability() == (8, 9), "an SM89 GPU"
    for name in ARGS.stages:
        globals()[name]()
    print("FAILS", fails, flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
