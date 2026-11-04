"""Draft experts on the CPU through the real kt-kernel (overlay), CPU only (Claude, 3 Nov 2026).

1. Loader isolation (B1c), real kt_kernel.utils.amx: a target-like NativeMoEWrapper builds the class-level loader on the
   target GGUF (its manifest from KT_MXFP4_GGUF_MANIFEST, as served); a draft wrapper built under _OWN_LOADER (the
   factory kt_dspark_draft gives it: draft GGUF, draft manifest) neither reads, replaces nor releases that loader,
   before or after its own load.
2. Loaded bytes: every expert projection the draft's own reader hands to the native load (load_experts, wrapped) is
   compared for a seeded sample of experts with the draft checkpoint's safetensors bytes (packed E2M1 and E8M0 codes);
   stages 1 and 2 through the same reader's read_projection.
3. Forward: the draft stage's 128 experts loaded into AVX2ContractV2_MOE over six K-parts (subpools on NUMA 4/5, the
   test's own cores), forward_routes and forward at 1, 3, 4 and 8 rows with top-3 routes (one skipped slot), against the
   contract v2 NumPy oracle (contract-v2-r1/contract_v2.py) on the checkpoint's weights, bit for bit.

Stand-ins: pinned allocations become plain ones (no CUDA in this process); LD_PRELOAD cpu_pin_shim.so keeps kt-kernel's
threads on the two cores taskset gave the test. argv: out.json, oracle folder, stage."""
import functools
import glob
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import torch

OUT, ORACLE, STAGE = sys.argv[1], sys.argv[2], int(sys.argv[3])
ALLOWED = {39, 47}
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()
T0 = time.time()
report = dict(stage=STAGE, checks=[], timings={})


def check(name, ok, **detail):
    report['checks'].append(dict(check=name, ok=bool(ok), **detail))
    print('PASS' if ok else 'FAIL', name, detail if not ok else '', flush=True)


def cpus(spec):
    out = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out |= set(range(int(a), int(b or a) + 1))
    return out


def guard(where):
    bad = {}
    for status in glob.glob('/proc/self/task/*/status'):
        try:
            line = next(l for l in open(status) if l.startswith('Cpus_allowed_list'))
        except (OSError, StopIteration):
            continue
        allowed = cpus(line.split(':')[1].strip())
        if not allowed <= ALLOWED:
            bad[status.split('/')[4]] = sorted(allowed)[:8]
    if bad:
        print('ABORT: threads outside the test cores at', where, bad, flush=True)
        os._exit(3)
    return len(glob.glob('/proc/self/task/*'))


for name in ('zeros', 'empty', 'full'):   # pinned host memory needs CUDA; plain memory here
    f = getattr(torch, name)
    setattr(torch, name, functools.wraps(f)(lambda *a, _f=f, **k: _f(*a, **{x: y for x, y in k.items() if x != 'pin_memory'})))

sys.path.insert(0, ORACLE)
import contract_v2 as oracle  # noqa: E402
from kt_kernel import KTMoEWrapper  # noqa: E402
from kt_kernel.utils import amx  # noqa: E402
from sglang.srt.layers.moe import kt_dspark_draft  # noqa: E402

guard('import')
CKPT = '/scratch/aster/bench/05yuki-baseline-3991/draft-checkpoint'
DRAFT = os.environ['KT_DSPARK_DRAFT_GGUF']
TARGET = '/scratch/aster/models/gguf/DeepSeek-V4.1-Flash-1017-MXFP4.gguf'
NODES = [4, 5, 4, 5, 4, 5]
COMMON = dict(hidden_size=5120, moe_intermediate_size=2304, cpuinfer_threads=6, threadpool_count=6, chunked_prefill_size=64,
              method='MXFP4', max_deferred_experts_per_token=0, swiglu_limit=10.0, numa_nodes=NODES)


class Checkpoint:
    def __init__(self, stage):
        self.path = f'{CKPT}/mtp-original-stage-{stage}.safetensors'
        with open(self.path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.hdr, self.base = json.loads(f.read(n)), 8 + n
        self.fd = os.open(self.path, os.O_RDONLY)
        self.stage = stage

    def raw(self, name):
        a, b = self.hdr[name]['data_offsets']
        data = os.pread(self.fd, b - a, self.base + a)
        os.posix_fadvise(self.fd, self.base + a, b - a, os.POSIX_FADV_DONTNEED)
        return np.frombuffer(data, np.uint8).reshape(self.hdr[name]['shape'])

    def expert(self, e):
        p = f'mtp.{self.stage}.ffn.experts.{e}.'
        g, u, d = (self.raw(p + w + '.weight') for w in ('w1', 'w3', 'w2'))
        gs, us, ds = (self.raw(p + w + '.scale') for w in ('w1', 'w3', 'w2'))
        return dict(gate=(g, gs), up=(u, us), down=(d, ds))


def codes_of(bf16_scales):
    bits = bf16_scales.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits == 0x0040, torch.zeros_like(bits), bits >> 7).to(torch.uint8).numpy()


rng = np.random.default_rng(20261103)
SAMPLE = sorted({0, 1, 63, 64, 126, 127} | set(rng.choice(np.arange(2, 126), 6, replace=False).tolist()))
report['sample'] = SAMPLE

# 1. loader isolation: a target-like wrapper first (class-level loader on the target GGUF, as served)
t = time.time()
target_like = KTMoEWrapper(layer_idx=0, num_experts=384, num_experts_per_tok=6, gpu_experts_mask=torch.zeros(384, dtype=torch.bool),
                           weight_path=TARGET, **COMMON)
threads = guard('CPUInfer start')
report['threads'] = threads
cls_loader = amx.NativeMoEWrapper._native_loader_instance
check('target-like wrapper: class-level loader on the target GGUF', cls_loader is not None and target_like.loader is cls_loader
      and str(cls_loader.identity[2]) == str(os.path.getsize(TARGET)))

cfg = kt_dspark_draft.DraftKTConfig(layer_idx=40 + STAGE, num_gpu_experts=0, cpuinfer_threads=6, threadpool_count=6, weight_path=DRAFT,
                                    chunked_prefill_size=64, max_deferred_experts_per_token=0, method='MXFP4', num_layers=None,
                                    draft_stage=STAGE, manifest=os.environ['KT_DSPARK_DRAFT_MANIFEST'],
                                    manifest_sha256=os.environ['KT_DSPARK_DRAFT_MANIFEST_SHA256'])
made, compared = [], {}
ckpt = Checkpoint(STAGE)


def factory():
    reader = kt_dspark_draft.loader_factory(cfg)()
    made.append(reader)
    inner = reader.load_experts

    def load_experts(base_key, device='cpu'):
        out = inner(base_key, device)
        for e in SAMPLE:   # the bytes the wrapper hands to the native load, against the checkpoint
            ref = ckpt.expert(e)
            for proj in ('gate', 'up', 'down'):
                w = out[proj][e].numpy()
                s = codes_of(out[proj + '_scale'][e])
                compared[(STAGE, e, proj)] = (np.array_equal(w, ref[proj][0]), np.array_equal(s, ref[proj][1]))
        return out

    reader.load_experts = load_experts
    return reader


token = amx._OWN_LOADER.set(factory)
try:
    draft = KTMoEWrapper(layer_idx=cfg.layer_idx, num_experts=128, num_experts_per_tok=3, gpu_experts_mask=torch.zeros(128, dtype=torch.bool),
                         weight_path=DRAFT, **COMMON)
finally:
    amx._OWN_LOADER.reset(token)
draft.weight_base_key = f'model.layers.{STAGE}'
check('draft wrapper construction: own factory kept, no loader opened, class-level loader untouched',
      draft.__dict__.get('_own_loader_factory') is factory and draft.loader is None and not made
      and amx.NativeMoEWrapper._native_loader_instance is cls_loader and cls_loader.fd is not None)
report['timings']['construct'] = round(time.time() - t, 1)

t = time.time()
draft.load_weights(torch.arange(128, dtype=torch.int64))
report['timings']['draft_load'] = round(time.time() - t, 1)
guard('after load')
check('draft load: own reader on the draft GGUF and manifest, closed after; class-level loader untouched and open',
      len(made) == 1 and made[0].fd is None and draft.loader is None and made[0].identity[2] == os.path.getsize(DRAFT)
      and hashlib.sha256(json.dumps(made[0].manifest, sort_keys=True).encode()).hexdigest() and made[0].manifest['file_size'] == os.path.getsize(DRAFT)
      and amx.NativeMoEWrapper._native_loader_instance is cls_loader and cls_loader.fd is not None)
check(f'draft load: native backend is AVX2ContractV2_MOE', type(draft.moe).__name__ == 'AVX2ContractV2_MOE', backend=type(draft.moe).__name__)
ok = [k for k, v in compared.items() if all(v)]
check(f'loaded bytes, stage {STAGE}: {len(ok)}/{len(compared)} sampled projections equal the checkpoint (packed E2M1 and E8M0)',
      len(compared) == 3 * len(SAMPLE) and len(ok) == len(compared))
report['loaded_bytes'] = {f'{s}/{e}/{p}': v for (s, e, p), v in compared.items()}

# stages other than STAGE: the same reader code, per sampled expert projection
for other in sorted({0, 1, 2} - {STAGE}):
    c2 = kt_dspark_draft.DraftKTConfig(**{**cfg.__dict__, 'layer_idx': 40 + other, 'draft_stage': other})
    reader = kt_dspark_draft.loader_factory(c2)()
    ck = Checkpoint(other)
    res = []
    try:
        for e in SAMPLE:
            ref = ck.expert(e)
            for proj in ('gate', 'up', 'down'):
                w, s = reader.read_projection(f'blk.{other}.ffn_{proj}_exps.weight', e)
                res.append(np.array_equal(w.numpy(), ref[proj][0]) and np.array_equal(codes_of(s), ref[proj][1]))
    finally:
        reader.close_all_handles()
    check(f'loaded bytes, stage {other} (reader.read_projection): {sum(res)}/{len(res)} sampled projections equal the checkpoint',
          len(res) == 3 * len(SAMPLE) and all(res))
check('class-level loader still the target one after the other stages', amx.NativeMoEWrapper._native_loader_instance is cls_loader)

# 3. forward against the oracle
weights_cache = {}


def oracle_weights(e):
    if e not in weights_cache:
        r = ckpt.expert(e)
        weights_cache[e] = (np.concatenate([r['gate'][0], r['up'][0]]), np.concatenate([r['gate'][1], r['up'][1]]), r['down'][0], r['down'][1])
    return weights_cache[e]


def bf16_bits(a):
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


real_accumulate = oracle.accumulate
lost_terms = [0]


def accumulate(D, ea, ws, EA, EW, g0=0, g1=None):
    """oracle.accumulate, counting the terms whose low bits its floor shift drops (the contract's truncation regime)."""
    m, n, Gp = D.shape
    e = ea[:, None, g0:g0 + Gp].astype(np.int64) + (ws[None, :, g0:g0 + Gp].astype(np.int64) - 127)
    depth = (EA[:, None].astype(np.int64) + EW[None, :].astype(np.int64) - oracle.MARGIN)[..., None] - e
    lost_terms[0] += int(((depth > 0) & ((D & ((1 << np.clip(depth, 0, 62)) - 1)) != 0)).sum())
    return real_accumulate(D, ea, ws, EA, EW, g0, g1)


oracle.accumulate = accumulate
ext = __import__('kt_kernel').kt_kernel_ext
TOLERANT = bool(os.environ.get('KT_CONTRACT_V2_TOLERANT_FROM'))
if TOLERANT:
    assert hasattr(ext, 'contract_v2_tolerant_events'), 'KT_CONTRACT_V2_TOLERANT_FROM needs the apply_tolerant.py build'
    assert cfg.layer_idx >= int(os.environ['KT_CONTRACT_V2_TOLERANT_FROM']) > 39
report['tolerant_from'] = os.environ.get('KT_CONTRACT_V2_TOLERANT_FROM')
report['native_sha256'] = hashlib.sha256(open(ext.__file__, 'rb').read()).hexdigest()
tolerated = []
gen = torch.Generator().manual_seed(1103)
pool = [e for e in SAMPLE]
t = time.time()
# typical inputs first; then inputs with one 40x outlier group in row 0, which can push the down projection's input past the
# contract's 20-bit margin: kt-kernel then refuses (its truncation counter is process-wide and sticky), so that regime runs last
refused = False
for regime, rows in [('typical', r) for r in (1, 3, 4, 8)] + [('typical-b', r) for r in (1, 3, 4, 8)] + [('outlier', r) for r in (1, 3, 4, 8)]:
    if refused:
        break
    x = (torch.randn(rows, 5120, generator=gen) * 0.6).to(torch.bfloat16)
    if regime == 'outlier':
        x[0, :32] *= 40
    ids = torch.tensor([[pool[(r * 5 + j * 3 + rows + (7 if regime == 'typical-b' else 0)) % len(pool)] for j in range(3)] for r in range(rows)],
                       dtype=torch.int64)
    if rows >= 3:
        ids[1, 2] = -1   # a skipped route
    w = torch.rand(rows, 3, generator=gen, dtype=torch.float32) * 0.9 + 0.05
    xf = x.float().numpy()
    ref = np.zeros((rows, 3, 5120), np.float32)
    lost_terms[0] = 0
    for e in sorted(set(ids[ids >= 0].tolist())):
        tok, slot = np.nonzero(ids.numpy() == e)
        ref[tok, slot] = oracle.expert(xf[tok], *oracle_weights(e), w.numpy()[tok, slot])
    lost = lost_terms[0]
    qlen = torch.tensor([rows], dtype=torch.int32)
    out = torch.full((rows, 3, 5120), 7, dtype=torch.bfloat16)
    try:
        draft.cpu_infer.submit(draft.moe.forward_routes_task(qlen.data_ptr(), 3, ids.data_ptr(), w.data_ptr(), x.data_ptr(), out.data_ptr()))
        draft.cpu_infer.sync()
    except RuntimeError as exc:
        refused = 'truncation regime' in str(exc)
        check(f'{regime}, {rows} rows: kt-kernel refuses exactly when the oracle floors low bits', refused and lost > 0 and not TOLERANT,
              oracle_lost_terms=lost, native=str(exc)[:80])
        report.setdefault('forward', []).append(dict(regime=regime, rows=rows, oracle_lost_terms=lost, native='refused'))
        continue
    summed = torch.full((rows, 5120), 7, dtype=torch.bfloat16)
    draft.moe.forward(qlen.data_ptr(), 3, ids.data_ptr(), w.data_ptr(), x.data_ptr(), summed.data_ptr(), False)
    got = out.view(torch.int16).numpy().view(np.uint16)
    mism = int((got != bf16_bits(ref)).sum())
    skipped_zero = bool((got[ids.numpy() < 0] == 0).all())
    ref_sum = oracle.combine(ref, np.where(ids.numpy() >= 0, ids.numpy(), 10 ** 6))
    sum_mism = int((summed.view(torch.int16).numpy().view(np.uint16) != bf16_bits(ref_sum)).sum())
    entry = dict(regime=regime, rows=rows, routes=int((ids >= 0).sum()), experts=sorted(set(ids[ids >= 0].tolist())),
                 route_mismatches=mism, sum_mismatches=sum_mism, oracle_lost_terms=lost, nonzero_values=int((got != 0).sum()))
    if TOLERANT and lost > 0:
        # the truncation regime under KT_CONTRACT_V2_TOLERANT_FROM: kept, not refused; the floor-per-part sum may differ
        # from the oracle's floor-per-term sum in the last bits; the process-wide counter stays clean
        diff = np.abs(got.astype(np.int32) - bf16_bits(ref).astype(np.int32))
        entry.update(native='tolerated', max_bf16_code_diff=int(diff.max()), tolerant_events=int(ext.contract_v2_tolerant_events()),
                     truncation_events=int(ext.contract_v2_truncation_events()))
        check(f'{regime}, {rows} rows (oracle floors {lost} terms): draft layer {cfg.layer_idx} tolerated, global counter clean, '
              f'{mism} of {got.size} BF16 values differ from the oracle', skipped_zero and ext.contract_v2_truncation_events() == 0
              and ext.contract_v2_tolerant_events() > 0, **{k: entry[k] for k in ('max_bf16_code_diff', 'tolerant_events')})
        tolerated.append((rows, x, ids, w))
    else:
        check(f'{regime}, forward_routes, {rows} rows x top-3: per-route BF16 equals the contract v2 oracle', mism == 0 and skipped_zero and lost == 0,
              mismatches=mism, skipped_zero=skipped_zero, oracle_lost_terms=lost)
        check(f'{regime}, forward (ascending-id FP32 combine), {rows} rows: equals oracle.combine', sum_mism == 0, mismatches=sum_mism)
    report.setdefault('forward', []).append(entry)

if TOLERANT and tolerated:
    # a layer below the tolerant id (39, here loaded with the same draft stage) keeps the served rule: it refuses
    rows, x, ids, w = tolerated[0]
    used = sorted(set(ids[ids >= 0].tolist()))
    mask = torch.ones(128, dtype=torch.bool)
    mask[used] = False   # only the experts this input routes to are loaded
    token = amx._OWN_LOADER.set(kt_dspark_draft.loader_factory(cfg))
    try:
        strict = KTMoEWrapper(layer_idx=39, num_experts=128, num_experts_per_tok=3, gpu_experts_mask=mask, weight_path=DRAFT, **COMMON)
    finally:
        amx._OWN_LOADER.reset(token)
    strict.weight_base_key = f'model.layers.{STAGE}'
    strict.load_weights(torch.arange(128, dtype=torch.int64))
    qlen = torch.tensor([rows], dtype=torch.int32)
    out = torch.zeros((rows, 3, 5120), dtype=torch.bfloat16)
    try:
        strict.cpu_infer.submit(strict.moe.forward_routes_task(qlen.data_ptr(), 3, ids.data_ptr(), w.data_ptr(), x.data_ptr(), out.data_ptr()))
        strict.cpu_infer.sync()
        msg = 'not refused'
    except RuntimeError as exc:
        msg = str(exc)
    check('layer 39 (below the tolerant id), same input: refuses as served, and the process-wide counter is set',
          'truncation regime' in msg and ext.contract_v2_truncation_events() > 0, native=msg[:80],
          truncation_events=int(ext.contract_v2_truncation_events()))
report['truncation_events'] = int(ext.contract_v2_truncation_events())
report['tolerant_events'] = int(ext.contract_v2_tolerant_events()) if hasattr(ext, 'contract_v2_tolerant_events') else None
report['timings']['forward_and_oracle'] = round(time.time() - t, 1)
guard('end')
report['seconds'] = round(time.time() - T0, 1)
report['status'] = 'PASS' if all(c['ok'] for c in report['checks']) else 'FAIL'
with open(OUT, 'w') as f:
    json.dump(report, f, indent=1)
print(report['status'], report['timings'], flush=True)
os._exit(0 if report['status'] == 'PASS' else 1)
