"""Check local_ckpt against the Hub: byte ranges of sampled tensors from the 1017 shards over HTTP, against what the loader returns. Experts, dense tensors, Engram rows. No GPU, no full download.
argv: output JSON."""
import hashlib
import json
import random
import struct
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, '/scratch/aster/bench/reference-panel-r1')
import local_ckpt as L  # noqa: E402

REPO = 'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/1bae0e1088993b9b82de08f59de5be5a34e42170/'
INDEX = '/scratch/aster/models/DeepSeek-V4.1-Flash-1017/model.safetensors.index.json'
HEADERS = {}


def fetch(shard, start, stop):
    request = urllib.request.Request(REPO + shard, headers={'Range': f'bytes={start}-{stop - 1}', 'User-Agent': 'reference-panel-check'})
    with urllib.request.urlopen(request, timeout=120) as r:
        data = r.read()
    assert len(data) == stop - start, (shard, len(data), stop - start)
    return data


def header(shard):
    if shard not in HEADERS:
        n = struct.unpack('<Q', fetch(shard, 0, 8))[0]
        HEADERS[shard] = (8 + n, json.loads(fetch(shard, 8, 8 + n)))
    return HEADERS[shard]


def hub_bytes(wm, name, start=0, length=None):
    shard = wm[name]
    base, meta = header(shard)
    t = meta[name]
    a, b = t['data_offsets']
    length = b - a if length is None else length
    return fetch(shard, base + a + start, base + a + start + length), t


def main(out):
    rng = random.Random(1017)
    wm = json.load(open(INDEX))['weight_map']
    ck = L.LocalCheckpoint()
    report = dict(gguf_sha_note='offsets read from the header', experts=[], dense=[], engram=[])
    print('gguf tensors', len(ck.offsets), [n for n in ck.offsets if n.startswith('blk.0.')][:30], flush=True)
    for layer, e in [(rng.randrange(40), rng.randrange(384)) for _ in range(9)] + [(0, 0), (39, 383), (1, 200)]:
        for proj in L.PROJ:
            packed, scales = ck.expert_projection(layer, proj, 'cpu')
            for part, mine in (('weight', packed[e]), ('scale', scales[e])):
                name = f'layers.{layer}.ffn.experts.{e}.{proj}.{part}'
                data, t = hub_bytes(wm, name)
                same = bytes(mine.numpy().tobytes()) == data
                report['experts'].append(dict(name=name, dtype=t['dtype'], shape=t['shape'], bytes=len(data), equal=same))
                assert same, name
            del packed, scales
        print('expert', layer, e, 'equal', flush=True)
    dense = [n for n in ck.weight_map if ck.weight_map[n] != '@gguf' and 'engram.embed' not in n and n in wm and '.ffn.experts.' not in n]
    for name in rng.sample(dense, 40) + ['norm.weight']:
        shard = wm[name]
        base, meta = header(shard)
        t = meta[name]
        size = t['data_offsets'][1] - t['data_offsets'][0]
        take = min(size, 1 << 20)
        data, _ = hub_bytes(wm, name, 0, take)
        path, offset, dtype, shape = ck.locate(name)
        with open(path, 'rb') as f:
            f.seek(offset)
            mine = f.read(take)
        report['dense'].append(dict(name=name, dtype=t['dtype'], local_dtype=dtype, shape=t['shape'], bytes=take, equal=mine == data))
        assert mine == data and t['dtype'] == dtype and t['shape'] == shape, name
    print('dense sample equal', len(report['dense']), flush=True)
    for layer in (1, 14):
        for part, width in (('weight', 256), ('scale', 8)):
            name = f'layers.{layer}.engram.embed.{part}'
            path, offset, dtype, shape = ck.locate(name)
            rows = rng.sample(range(shape[0]), 64)
            equal = 0
            with open(path, 'rb') as f:
                for row in rows:
                    data, t = hub_bytes(wm, name, row * width, width)
                    f.seek(offset + row * width)
                    equal += f.read(width) == data
            report['engram'].append(dict(name=name, rows=len(rows), equal=equal, shape=shape, hub_shape=t['shape'], hub_dtype=t['dtype']))
            assert equal == len(rows) and shape == t['shape'], name
        print('engram', layer, 'equal', flush=True)
    report['status'] = 'PASS'
    json.dump(report, open(out, 'x'), indent=1)
    print('PASS', len(report['experts']), 'expert tensors', len(report['dense']), 'dense', len(report['engram']), 'engram', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
