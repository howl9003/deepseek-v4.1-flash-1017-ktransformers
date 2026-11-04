"""Is the DSpark draft GGUF a lossless repack of the draft checkpoint's routed experts? (Claude, 3 Nov 2026)

Stdlib + NumPy only; no kt_kernel or SGLang code. For every expert projection range of the GGUF (3 stages x 128 experts x
gate/up/down = 1,152 ranges, from the draft's range manifest, whose digest is pinned):
  - the raw range's SHA256 equals the manifest's;
  - the ggml MXFP4 blocks [E8M0 e, 16 bytes qs] are decoded by ggml's own rule (element j = qs[j] & 15, element
    j + 16 = qs[j] >> 4), re-packed in the checkpoint's order (byte i = element 2i | element 2i+1 << 4), and the
    packed bytes and the e bytes are hashed against the checkpoint receipt's per-tensor SHA256
    (mtp.S.ffn.experts.E.w1/w3/w2 .weight and .scale) -- full coverage;
  - for a seeded sample of experts the same bytes are compared byte for byte with the tensors read from the
    checkpoint's safetensors files, whose own SHA256 must equal the receipt's (ties the receipt to the file).
argv: output JSON path."""
import hashlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

GGUF = Path('/scratch/aster/models/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf')
MANIFEST = Path('/scratch/aster/bench/kt-native-pilot/evidence/dspark-expert-inventory-v1/expert-ranges.json')
MANIFEST_SHA = 'b75712873df442ab6047ee4eefad8b0c2b79e0ab3048f444d456fce77bc0fb7d'
CKPT = Path('/scratch/aster/bench/05yuki-baseline-3991/draft-checkpoint')
PROJ = {'gate': 'w1', 'up': 'w3', 'down': 'w2'}
STAGES, EXPERTS = 3, 128
OUT = Path(sys.argv[1])
sha = lambda b: hashlib.sha256(b).hexdigest()


def drop(fd, off=0, n=0):
    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)


def ggml_to_checkpoint(raw, n, k):
    b = np.frombuffer(raw, np.uint8).reshape(-1, 17)
    if b.shape[0] != n * k // 32:
        raise ValueError('block count')
    qs = b[:, 1:]
    codes = np.empty((b.shape[0], 32), np.uint8)
    codes[:, :16] = qs & 15
    codes[:, 16:] = qs >> 4
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.reshape(n, k // 2).tobytes(), b[:, 0].reshape(n, k // 32).tobytes()


def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


t0 = time.time()
mbytes = MANIFEST.read_bytes()
assert sha(mbytes) == MANIFEST_SHA, 'manifest digest'
manifest = json.loads(mbytes)
assert manifest['file_size'] == GGUF.stat().st_size
ranges = {(r['tensor'], r['expert']): r for r in manifest['ranges']}
assert len(ranges) == STAGES * EXPERTS * 3 == len(manifest['ranges'])

receipt, headers, files = {}, {}, {}
for s in range(STAGES):
    p = CKPT / f'mtp-original-stage-{s}.safetensors'
    rc = json.loads((CKPT / f'mtp-original-stage-{s}.safetensors.receipt.json').read_text())
    assert rc['file_bytes'] == p.stat().st_size
    for t in rc['tensors']:
        receipt[t['name']] = t
    headers[s] = st_header(p)
    files[s] = p

rng = random.Random(20261103)
sample = {s: sorted({0, 1, 63, 64, 126, 127} | set(rng.sample(range(2, 126), 6))) for s in range(STAGES)}
report = dict(gguf=str(GGUF), gguf_bytes=GGUF.stat().st_size, manifest=str(MANIFEST), manifest_sha256=MANIFEST_SHA,
              checkpoint=str(CKPT), ranges=0, raw_sha_ok=0, weight_sha_ok=0, scale_sha_ok=0, shape_ok=0,
              sampled_direct=0, sampled_weight_bytes_equal=0, sampled_scale_bytes_equal=0, checkpoint_sha_ok=0,
              mismatches=[], sample={str(s): v for s, v in sample.items()})
gfd = os.open(GGUF, os.O_RDONLY)
cfd = {s: os.open(files[s], os.O_RDONLY) for s in range(STAGES)}
try:
    for s in range(STAGES):
        hdr, base = headers[s]
        for e in range(EXPERTS):
            for proj, wname in PROJ.items():
                r = ranges[(f'blk.{s}.ffn_{proj}_exps.weight', e)]
                n, k = r['shape']
                raw = os.pread(gfd, r['bytes'], r['offset'])
                drop(gfd, r['offset'], r['bytes'])
                report['ranges'] += 1
                if len(raw) != r['bytes'] or sha(raw) != r['sha256']:
                    report['mismatches'].append(('raw', s, e, proj)); continue
                report['raw_sha_ok'] += 1
                packed, scales = ggml_to_checkpoint(raw, n, k)
                wn, sn = f'mtp.{s}.ffn.experts.{e}.{wname}.weight', f'mtp.{s}.ffn.experts.{e}.{wname}.scale'
                rw, rs = receipt[wn], receipt[sn]
                if (rw['dtype'], rw['shape'], rs['dtype'], rs['shape']) == ('I8', [n, k // 2], 'F8_E8M0', [n, k // 32]):
                    report['shape_ok'] += 1
                else:
                    report['mismatches'].append(('shape', s, e, proj))
                if sha(packed) == rw['sha256']:
                    report['weight_sha_ok'] += 1
                else:
                    report['mismatches'].append(('weight', s, e, proj))
                if sha(scales) == rs['sha256']:
                    report['scale_sha_ok'] += 1
                else:
                    report['mismatches'].append(('scale', s, e, proj))
                if e in sample[s]:
                    report['sampled_direct'] += 1
                    for name, mine, key in ((wn, packed, 'sampled_weight_bytes_equal'), (sn, scales, 'sampled_scale_bytes_equal')):
                        a, b = hdr[name]['data_offsets']
                        theirs = os.pread(cfd[s], b - a, base + a)
                        drop(cfd[s], base + a, b - a)
                        if sha(theirs) == receipt[name]['sha256']:
                            report['checkpoint_sha_ok'] += 1
                        if theirs == mine:
                            report[key] += 1
                        else:
                            report['mismatches'].append(('direct', name))
finally:
    os.close(gfd)
    for fd in cfd.values():
        os.close(fd)
n = STAGES * EXPERTS * 3
d = report['sampled_direct']
report['status'] = 'LOSSLESS' if (not report['mismatches'] and report['raw_sha_ok'] == n and report['weight_sha_ok'] == n
                                  and report['scale_sha_ok'] == n and report['shape_ok'] == n and d == 3 * sum(len(v) for v in sample.values())
                                  and report['sampled_weight_bytes_equal'] == d and report['sampled_scale_bytes_equal'] == d
                                  and report['checkpoint_sha_ok'] == 2 * d) else 'NOT_LOSSLESS'
report['seconds'] = round(time.time() - t0, 1)
OUT.write_text(json.dumps(report, indent=1) + '\n')
print(report['status'], {k: v for k, v in report.items() if k not in ('mismatches', 'sample')}, report['mismatches'][:5])
