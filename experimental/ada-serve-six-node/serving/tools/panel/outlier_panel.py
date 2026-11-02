"""Where two panel captures differ most, read against the reference (Claude, 2 Nov 2026). Selected-token logprobs only. argv: capture A, capture B, reference part files in sequence order."""
import array
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def chosen(path):
    data = Path(path).read_bytes()
    n, top = struct.unpack('<qq', data[:16])
    v = array.array('f')
    v.frombytes(data[16 + 8 * n * top:])
    assert len(v) == n
    return torch.tensor(v, dtype=torch.float32)


def tail(d):
    d = d.abs()
    return {str(t): int((d > t).sum()) for t in (0.01, 0.1, 0.5, 1, 3, 5, 10)}


def main():
    a, b, parts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    refs = [load_file(p) for p in parts]
    r = -torch.cat([x['nll'] for x in refs]).float()
    lens = torch.cat([x['lens'] for x in refs]).tolist()
    x = torch.cat([chosen(a / f'seq-{i:04d}.bin') for i in range(len(lens))])
    y = torch.cat([chosen(b / f'seq-{i:04d}.bin') for i in range(len(lens))])
    assert x.numel() == y.numel() == r.numel()
    where = [(i, p) for i, k in enumerate(lens) for p in range(k)]
    worst = (y - x).abs().topk(12).indices.tolist()
    print(json.dumps(dict(positions=x.numel(), a_vs_b=tail(y - x), a_vs_reference=tail(x - r), b_vs_reference=tail(y - r),
                          rms_a_vs_reference=(x - r).square().mean().sqrt().item(), rms_b_vs_reference=(y - r).square().mean().sqrt().item(),
                          worst=[dict(conversation=where[j][0], position=where[j][1], a=x[j].item(), b=y[j].item(), reference=r[j].item()) for j in worst]), indent=1))


if __name__ == '__main__':
    main()
