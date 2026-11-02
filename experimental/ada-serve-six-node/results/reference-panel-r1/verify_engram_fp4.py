"""Which FP8 to MXFP4 rule made NVIDIA's Engram tables? Sampled rows of nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 over HTTP against each candidate applied to our FP8 rows. No GPU. argv: index json of NVIDIA's repo."""
import json
import random
import struct
import sys
import urllib.request

import torch

sys.path.insert(0, '/scratch/aster/bench/reference-panel-r1')
import local_ckpt as L  # noqa: E402
import w4a4  # noqa: E402

REPO = 'https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4/resolve/main/'
wm = json.load(open(sys.argv[1]))['weight_map']
HEADERS = {}


def fetch(shard, start, stop):
    request = urllib.request.Request(REPO + shard, headers={'Range': f'bytes={start}-{stop - 1}', 'User-Agent': 'reference-panel-check'})
    with urllib.request.urlopen(request, timeout=120) as r:
        return r.read()


def header(shard):
    if shard not in HEADERS:
        n = struct.unpack('<Q', fetch(shard, 0, 8))[0]
        HEADERS[shard] = (8 + n, json.loads(fetch(shard, 8, 8 + n)))
    return HEADERS[shard]


def rule_ceil(amax):
    """The reference's fp4_quant_kernel: 2^ceil(log2(amax / 6))."""
    m, e = torch.frexp(amax.double() / 6)
    return e - (m == 0.5).long()


def rule_vllm(amax):
    """vLLM's UE8M0 path: biased exponent of amax after rounding bias at mantissa bit 21, minus 2."""
    import numpy as np
    bits = amax.float().numpy()
    raw = np.frombuffer(bits.tobytes(), dtype=np.uint32).copy().reshape(bits.shape)
    rounded = (raw + (1 << 21)) & 0xFF800000
    exp = ((rounded >> 23) & 0xFF).astype(np.int64) - 2
    return torch.from_numpy(exp - 127)


def rule_floor_up(amax):
    """floor(log2(amax)) - 2 + 1, the nearest alternative."""
    m, e = torch.frexp(amax.double())
    return e - 2


def encode(vals, exponent):
    scale = torch.exp2(exponent.double()).float()[..., None]
    y = vals / scale
    q = w4a4.to_e2m1(y.abs().clamp_max(6.0), True)
    table = torch.tensor(w4a4.VALUES)
    code = (q[..., None] == table).float().argmax(-1)
    sign = ((y < 0) & (code > 0)).long() * 8       # -0 becomes +0, as NVIDIA's export does
    nib = (code + sign).flatten(1)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8), (exponent + 127).to(torch.uint8)


ck = L.LocalCheckpoint()
rng = random.Random(1017)
report = {}
for layer in (1, 14):
    wname, sname = f'layers.{layer}.engram.embed.weight', f'layers.{layer}.engram.embed.scale'
    path, woff, _, shape = ck.locate(wname)
    spath, soff, _, _ = ck.locate(sname)
    rows = sorted(rng.sample(range(shape[0]), 48))
    base, meta = header(wm[wname])
    w_a = meta[wname]['data_offsets'][0] + base
    base2, meta2 = header(wm[sname])
    s_a = meta2[sname]['data_offsets'][0] + base2
    print(layer, meta[wname]['dtype'], meta[wname]['shape'], meta2[sname]['dtype'], meta2[sname]['shape'], flush=True)
    theirs_w, theirs_s = [], []
    mine = []
    with open(path, 'rb') as f, open(spath, 'rb') as g:
        for r in rows:
            theirs_w.append(fetch(wm[wname], w_a + r * 128, w_a + (r + 1) * 128))
            theirs_s.append(fetch(wm[sname], s_a + r * 8, s_a + (r + 1) * 8))
            f.seek(woff + r * 256)
            g.seek(soff + r * 8)
            mine.append((f.read(256), g.read(8)))
    w = torch.tensor([list(a) for a, _ in mine], dtype=torch.uint8).view(torch.float8_e4m3fn)
    s = torch.tensor([list(b) for _, b in mine], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    vals = w.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)
    want_w = torch.tensor([list(x) for x in theirs_w], dtype=torch.uint8)
    want_s = torch.tensor([list(x) for x in theirs_s], dtype=torch.uint8)
    for name, rule in (('ceil', rule_ceil), ('vllm', rule_vllm), ('floor', rule_floor_up)):
        amax = vals.abs().amax(-1)
        amax = amax.clamp_min(1e-30)
        cw, cs = encode(vals, rule(amax))
        report[f'{layer}-{name}'] = dict(scale_equal=float((cs == want_s).float().mean()), weight_equal=float((cw == want_w).float().mean()))
        print(layer, name, report[f'{layer}-{name}'], flush=True)
print(json.dumps(report))
