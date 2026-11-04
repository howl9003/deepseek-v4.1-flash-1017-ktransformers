"""Why the stage-2 forward test reached the contract's truncation regime (Claude, 3 Nov 2026). CPU only, light reads.

1. Weight exponent spread per output row (max - min E8M0 code over the K groups) of gate/up/down, all 384 draft experts,
   against a sample of target experts (layers 0, 20, 39; experts 0-7) read from the target GGUF through its manifest.
2. The 4-row stage-2 input of test_draft_forward.py, regenerated from the same seeds, through the contract v2 oracle with
   every term below the row base counted (oracle.accumulate floors those; kt-kernel's K-part merge refuses them).
argv: out.json, oracle folder."""
import json
import os
import struct
import sys

import numpy as np
import torch

OUT, ORACLE = sys.argv[1], sys.argv[2]
sys.path.insert(0, ORACLE)
import contract_v2 as oracle  # noqa: E402

CKPT = '/scratch/aster/bench/05yuki-baseline-3991/draft-checkpoint'
TMAN = '/scratch/aster/bench/kt-native-pilot/evidence/expert-inventory/expert-ranges.json'
TGGUF = '/scratch/aster/models/gguf/DeepSeek-V4.1-Flash-1017-MXFP4.gguf'
report = {}


class Ckpt:
    def __init__(self, s):
        p = f'{CKPT}/mtp-original-stage-{s}.safetensors'
        with open(p, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            self.h, self.base = json.loads(f.read(n)), 8 + n
        self.fd, self.s = os.open(p, os.O_RDONLY), s

    def raw(self, name):
        a, b = self.h[name]['data_offsets']
        d = os.pread(self.fd, b - a, self.base + a)
        os.posix_fadvise(self.fd, self.base + a, b - a, os.POSIX_FADV_DONTNEED)
        return np.frombuffer(d, np.uint8).reshape(self.h[name]['shape'])


def spread_stats(codes_rows):
    sp = codes_rows.max(1).astype(int) - codes_rows.min(1).astype(int)
    return dict(max=int(sp.max()), p99=float(np.percentile(sp, 99)), mean=round(float(sp.mean()), 2),
                ge12=int((sp >= 12).sum()), ge16=int((sp >= 16).sum()), rows=int(sp.size))


ck = [Ckpt(s) for s in range(3)]
draft = {}
for s in range(3):
    for w in ('w1', 'w3', 'w2'):
        rows = np.concatenate([ck[s].raw(f'mtp.{s}.ffn.experts.{e}.{w}.scale') for e in range(128)])
        draft[f'stage{s}.{w}'] = spread_stats(rows)
        # the K-partition of the down projection: six slices of 12 groups; spread between slices' maxima per row
        if w == 'w2':
            parts = rows.reshape(rows.shape[0], 6, 12).max(2).astype(int)
            draft[f'stage{s}.w2.part_max_spread'] = spread_stats(parts)
report['draft_weight_code_spread_per_row'] = draft

man = json.loads(open(TMAN, 'rb').read())
rng = {(r['tensor'], r['expert']): r for r in man['ranges']}
fd = os.open(TGGUF, os.O_RDONLY)
target = {}
for layer in (0, 20, 39):
    for proj in ('gate', 'up', 'down'):
        rows = []
        for e in range(8):
            r = rng[(f'blk.{layer}.ffn_{proj}_exps.weight', e)]
            raw = os.pread(fd, r['bytes'], r['offset'])
            os.posix_fadvise(fd, r['offset'], r['bytes'], os.POSIX_FADV_DONTNEED)
            n, k = r['shape']
            rows.append(np.frombuffer(raw, np.uint8).reshape(-1, 17)[:, 0].reshape(n, k // 32))
        target[f'layer{layer}.{proj}'] = spread_stats(np.concatenate(rows))
os.close(fd)
report['target_weight_code_spread_per_row_sample'] = target

# 2. the failing input, regenerated: same generator, same draws as test_draft_forward.py up to rows=4
pool = report_sample = None
r = np.random.default_rng(20261103)
SAMPLE = sorted({0, 1, 63, 64, 126, 127} | set(r.choice(np.arange(2, 126), 6, replace=False).tolist()))
gen = torch.Generator().manual_seed(1103)
cases = {}
for rows in (1, 3, 4, 8):
    x = (torch.randn(rows, 5120, generator=gen) * 0.6).to(torch.bfloat16)
    x[0, :32] *= 40
    ids = torch.tensor([[SAMPLE[(q * 5 + j * 3 + rows) % len(SAMPLE)] for j in range(3)] for q in range(rows)], dtype=torch.int64)
    if rows >= 3:
        ids[1, 2] = -1
    w = torch.rand(rows, 3, generator=gen, dtype=torch.float32) * 0.9 + 0.05
    cases[rows] = (x.float().numpy(), ids.numpy(), w.numpy())

real_accumulate = oracle.accumulate
counts = []


def accumulate(D, ea, ws, EA, EW, g0=0, g1=None):
    m, n, Gp = D.shape
    e = ea[:, None, g0:g0 + Gp].astype(np.int64) + (ws[None, :, g0:g0 + Gp].astype(np.int64) - 127)
    B = EA[:, None].astype(np.int64) + EW[None, :].astype(np.int64) - oracle.MARGIN
    below = (e - B[..., None] < 0) & (D != 0)
    lost = (e - B[..., None] < 0) & ((D & ((1 << np.clip(B[..., None] - e, 0, 62)) - 1)) != 0)
    counts.append(dict(n=int(n), terms_below_base=int(below.sum()), terms_losing_bits=int(lost.sum()),
                       rows_affected=int(below.any(axis=(0, 2)).sum()),
                       act_spread=[int(ea[i].max()) - int(ea[i].min()) for i in range(m)],
                       max_depth=int((B[..., None] - e)[below].max()) if below.any() else 0))
    return real_accumulate(D, ea, ws, EA, EW, g0, g1)


oracle.accumulate = accumulate
out = {}
for stage in (0, 2):
    for rows in (4,):
        x, ids, w = cases[rows]
        for e in sorted(set(ids[ids >= 0].tolist())):
            p = f'mtp.{stage}.ffn.experts.{e}.'
            w13 = np.concatenate([ck[stage].raw(p + 'w1.weight'), ck[stage].raw(p + 'w3.weight')])
            s13 = np.concatenate([ck[stage].raw(p + 'w1.scale'), ck[stage].raw(p + 'w3.scale')])
            tok, slot = np.nonzero(ids == e)
            counts.clear()
            oracle.expert(x[tok], w13, s13, ck[stage].raw(p + 'w2.weight'), ck[stage].raw(p + 'w2.scale'), w[tok, slot])
            out[f'stage{stage}.rows{rows}.expert{e}'] = dict(tokens=tok.tolist(), gate_up=counts[0], down=counts[1])
report['oracle_terms_below_base'] = out
with open(OUT, 'w') as f:
    json.dump(report, f, indent=1)
print(json.dumps(report['draft_weight_code_spread_per_row'], indent=0)[:1500])
print(json.dumps(report['target_weight_code_spread_per_row_sample'])[:1200])
for k, v in out.items():
    print(k, 'gate_up below', v['gate_up']['terms_below_base'], 'down below', v['down']['terms_below_base'], 'lost', v['down']['terms_losing_bits'],
          'rows', v['down']['rows_affected'], 'act spread', v['down']['act_spread'], 'depth', v['down']['max_depth'])
