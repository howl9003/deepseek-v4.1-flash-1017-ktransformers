"""No-model probe of DeepSeek's reference kernels on the 4080 SUPER (Claude, 1 Nov 2026). Does inference/kernel.py compile and run on sm_89, and how far is each kernel from exact math?

Loads the checkpoint's own kernel.py unmodified, from the path in REF. Each case runs in its own process so a compile failure or crash costs that case only. For each: compiled, ran, bytes equal over
repeats (act_quant's num_stages = 0 path is the race other cards patch), distance from an FP64 oracle, and for the GEMMs whether a row's bits depend on the row count. No model, no weights.

argv: output JSON, or --one <case>.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

REF = '/scratch/aster/models/DeepSeek-V4.1-Flash-1017/inference/kernel.py'
KERNEL_SHA = '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
CASES = ('act_quant_fp8', 'act_quant_inplace', 'fp4_quant_e8m0', 'fp4_quant_e4m3', 'fp4_quant_packed', 'fp8_gemm', 'fp4_gemm', 'sparse_attn_h64', 'sparse_attn_h16', 'hc_sinkhorn')
REPEATS = 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def load():
    import torch
    torch.set_default_dtype(torch.bfloat16)
    spec = importlib.util.spec_from_file_location('ref_kernel', REF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, module


def repeats(torch, fn):
    first = fn()
    same = 0
    for _ in range(REPEATS - 1):
        same += all(torch.equal(a.view(torch.uint8) if a.dtype != torch.uint8 else a, b.view(torch.uint8) if b.dtype != torch.uint8 else b) for a, b in zip(first, fn()))
    return first, same == REPEATS - 1


def activations(torch, m, n, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    # Spread across about eight octaves, like a hidden state, with some all-zero groups.
    x = torch.randn(m, n, generator=g, device='cuda', dtype=torch.float64) * torch.exp2(torch.randint(-5, 3, (m, 1), generator=g, device='cuda').double())
    x[:, n // 2:n // 2 + 32] = 0
    return x.to(torch.bfloat16)


def round_scale(torch, amax, limit):
    """2^ceil(log2(amax / limit)) by frexp. r1 and r2 went through torch.log2, which is not exact on powers of two, so amax = 7 x 2^k and 6 x 2^k landed one octave high; the kernel is right."""
    m, e = torch.frexp(amax.double() / limit)
    return torch.exp2((e - (m == 0.5).long()).double()).float()


def act_quant_fp8(torch, k):
    x = activations(torch, 67, 5120, 1).contiguous()
    (y, s), stable = repeats(torch, lambda: k.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu))
    amax = x.reshape(67, -1, 32).float().abs().amax(-1).clamp_min(1e-4)
    want_s = round_scale(torch, amax, 448.0)
    got_s = s.float()
    want_y = (x.reshape(67, -1, 32).float() / want_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(67, 5120)
    return dict(stable=stable, scale_mismatch=int((want_s != got_s).sum()), value_mismatch=int((want_y.view(torch.uint8) != y.view(torch.uint8)).sum()), values=y.numel())


def act_quant_inplace(torch, k):
    x = activations(torch, 67, 5120, 2).contiguous()

    def run():
        z = x.clone()
        k.act_quant(z, 32, 'ue8m0', torch.float8_e8m0fnu, True)
        return (z,)
    (z,), stable = repeats(torch, run)
    amax = x.reshape(67, -1, 32).float().abs().amax(-1).clamp_min(1e-4)
    s = round_scale(torch, amax, 448.0)[..., None]
    want = ((x.reshape(67, -1, 32).float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).to(torch.bfloat16).reshape(67, 5120)
    return dict(stable=stable, mismatch=int((want.view(torch.uint16) != z.view(torch.uint16)).sum()), values=z.numel())


def to_e2m1(torch, v):
    """Round to E2M1, ties to the even significand, saturating at 6."""
    table = torch.tensor(E2M1, dtype=torch.float32, device=v.device)
    even = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32, device=v.device)  # by code: 0.5 is code 1, so 0.75 ties up to 1.0
    a = v.abs()[..., None]
    gap = (a - table).abs()
    best = (gap + (1 - even) * 1e-5).argmin(-1)
    return table[best] * v.sign()


def fp4_quant(torch, k, dtype, block):
    x = activations(torch, 67, 1024, 3).contiguous()

    def run():
        z = x.clone()
        k.fp4_act_quant(z, block, True, scale_dtype=dtype)
        return (z,)
    (z,), stable = repeats(torch, run)
    groups = x.reshape(67, -1, block).float()
    amax = groups.abs().amax(-1)
    if dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax.double() / 6).float().to(torch.float8_e4m3fn).float()
    else:
        s = round_scale(torch, amax.clamp_min(6 * 2.0 ** -126), 6.0)
    want = (to_e2m1(torch, (groups / s[..., None]).clamp(-6, 6)) * s[..., None]).to(torch.bfloat16).reshape(67, 1024)
    return dict(stable=stable, mismatch=int((want.view(torch.uint16) != z.view(torch.uint16)).sum()), values=z.numel())


def fp4_quant_e8m0(torch, k):
    return fp4_quant(torch, k, torch.float8_e8m0fnu, 32)


def fp4_quant_e4m3(torch, k):
    return fp4_quant(torch, k, torch.float8_e4m3fn, 16)


def fp4_quant_packed(torch, k):
    x = activations(torch, 67, 1024, 4).contiguous()
    (y, s), stable = repeats(torch, lambda: k.fp4_act_quant(x, 32))
    return dict(stable=stable, y_dtype=str(y.dtype), s_dtype=str(s.dtype), shape=list(y.shape))


def gemm_report(torch, call, exact, m):
    """call(M) -> rows from the first M activations. Distance from exact math, repeat stability, and row bits across row counts."""
    (c,), stable = repeats(torch, lambda: (call(m),))
    got = c.double()
    relative = float((got - exact).norm() / exact.norm())
    bf16 = exact.float().to(torch.bfloat16)
    out = dict(stable=stable, rel_l2_vs_exact=relative, bf16_equal_exact_rne=float((bf16.view(torch.uint16) == c.view(torch.uint16)).double().mean()), rows_by_count={})
    for count in (1, 2, 16, 32, 33, 64):
        part = call(count)
        out['rows_by_count'][count] = int((part.view(torch.uint16) != c[:count].view(torch.uint16)).any(-1).sum())
    return out


def e8m0(torch, codes):
    return torch.exp2((codes.double() - 127))


def fp8_gemm(torch, k):
    m, n, K = 67, 256, 5120
    g = torch.Generator(device='cuda').manual_seed(5)
    x = activations(torch, m, K, 5).contiguous()
    a, a_s = k.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
    w = torch.randint(0, 256, (n, K), generator=g, device='cuda', dtype=torch.uint8)
    w = torch.where((w & 0x7f) >= 0x78, w & 0x77, w).view(torch.float8_e4m3fn)
    w_c = torch.randint(118, 124, (n // 32, K // 32), generator=g, device='cuda', dtype=torch.uint8)
    w_s = w_c.view(torch.float8_e8m0fnu)
    av = a.double() * e8m0(torch, a_s.view(torch.uint8)).repeat_interleave(32, 1)
    wv = w.double() * e8m0(torch, w_c).repeat_interleave(32, 0).repeat_interleave(32, 1)
    exact = av @ wv.T
    return gemm_report(torch, lambda M: k.fp8_gemm(a[:M].contiguous(), a_s[:M].contiguous(), w, w_s, torch.float8_e8m0fnu, 32), exact, m)


def fp4_gemm(torch, k):
    m, n, K = 67, 512, 5120
    g = torch.Generator(device='cuda').manual_seed(6)
    x = activations(torch, m, K, 6).contiguous()
    a, a_s = k.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
    codes = torch.randint(0, 256, (n, K // 2), generator=g, device='cuda', dtype=torch.uint8)
    w_c = torch.randint(121, 127, (n, K // 32), generator=g, device='cuda', dtype=torch.uint8)
    table = torch.tensor([v for v in E2M1] + [-v for v in E2M1], dtype=torch.float64, device='cuda')
    low, high = table[(codes & 15).long()], table[(codes >> 4).long()]
    values = torch.stack((low, high), -1).reshape(n, K)
    av = a.double() * e8m0(torch, a_s.view(torch.uint8)).repeat_interleave(32, 1)
    exact = av @ (values * e8m0(torch, w_c).repeat_interleave(32, 1)).T
    packed = codes.view(torch.float4_e2m1fn_x2)
    scales = w_c.view(torch.float8_e8m0fnu)
    return gemm_report(torch, lambda M: k.fp4_gemm(a[:M].contiguous(), a_s[:M].contiguous(), packed, scales, torch.float8_e8m0fnu, 32), exact, m)


def sparse_case(torch, k, h):
    b, s, d, n, top = 1, 8, 512, 2048, 128
    g = torch.Generator(device='cuda').manual_seed(7)
    q = (torch.randn(b, s, h, d, generator=g, device='cuda', dtype=torch.float32) * 0.3).to(torch.bfloat16)
    kv = (torch.randn(b, n, d, generator=g, device='cuda', dtype=torch.float32) * 0.3).to(torch.bfloat16)
    sink = torch.randn(h, generator=g, device='cuda', dtype=torch.float32)
    idx = torch.stack([torch.randperm(n, generator=g, device='cuda')[:top] for _ in range(s)]).to(torch.int32)[None]
    idx[:, :, top - 7:] = -1
    scale = (1.0 / d) ** 0.5
    (o,), stable = repeats(torch, lambda: (k.sparse_attn(q, kv, sink, idx, scale),))
    keep = kv[0][idx[0].clamp_min(0).long()].double()
    mask = idx[0] >= 0
    scores = torch.einsum('shd,skd->shk', q[0].double(), keep) * scale
    scores = scores.masked_fill(~mask[:, None, :], float('-inf'))
    top_score = scores.amax(-1, keepdim=True)
    p = torch.exp(scores - top_score)
    total = p.sum(-1) + torch.exp(sink.double()[None] - top_score[..., 0])
    want = torch.einsum('shk,skd->shd', p, keep.masked_fill(~mask[..., None], 0)) / total[..., None]
    return dict(stable=stable, rel_l2_vs_exact=float((o[0].double() - want).norm() / want.norm()))


def sparse_attn_h64(torch, k):
    return sparse_case(torch, k, 64)


def sparse_attn_h16(torch, k):
    return sparse_case(torch, k, 16)


def hc_sinkhorn(torch, k):
    hc, iters, eps = 4, 20, 1e-6
    g = torch.Generator(device='cuda').manual_seed(8)
    size = (2 + hc) * hc
    mixes = torch.randn(3, 41, size, generator=g, device='cuda', dtype=torch.float32)
    scale = torch.tensor([0.7, 1.1, 0.9], device='cuda', dtype=torch.float32)
    base = torch.randn(size, generator=g, device='cuda', dtype=torch.float32)
    out, stable = repeats(torch, lambda: k.hc_split_sinkhorn(mixes, scale, base, hc, iters, eps))
    m, s, b = mixes.double(), scale.double(), base.double()
    comb = (m[..., 2 * hc:] * s[2] + b[2 * hc:]).reshape(3, 41, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return dict(stable=stable, comb_rel_l2_vs_exact=float((out[2].double() - comb).norm() / comb.norm()))


def one(case):
    started = time.time()
    torch, k = load()
    print(json.dumps(dict(case=case, seconds=None, **globals()[case](torch, k)) | dict(seconds=round(time.time() - started, 1))), flush=True)


def main(path):
    assert hashlib.sha256(open(REF, 'rb').read()).hexdigest() == KERNEL_SHA
    import torch
    report = dict(kernel_sha256=KERNEL_SHA, device=torch.cuda.get_device_name(0), torch=torch.__version__, cases={})
    for case in CASES:
        started = time.time()
        try:
            r = subprocess.run([sys.executable, '-I', '-B', __file__, '--one', case], capture_output=True, encoding='utf-8', errors='replace', timeout=1500)
            line = [x for x in r.stdout.splitlines() if x.startswith('{')]
            report['cases'][case] = json.loads(line[-1]) if r.returncode == 0 and line else dict(failed=True, code=r.returncode, error=(r.stderr or r.stdout)[-2500:])
        except subprocess.TimeoutExpired:
            report['cases'][case] = dict(failed=True, error='timeout')
        print(case, round(time.time() - started, 1), json.dumps(report['cases'][case])[:700], flush=True)
        json.dump(report, open(path, 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        try:
            one(sys.argv[2])
        except Exception:
            traceback.print_exc()
            sys.exit(3)
    else:
        main(sys.argv[1])
