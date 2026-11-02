"""Rows that come out context-free: the top three tokens are '<', newline and '#' in some order, and the real next token is improbable. Counts them in the reference and in each capture (Claude, 2 Nov 2026).
argv: capture dirs..., --, reference parts."""
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from score_panel import read_seq

PANEL = '/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors'
GENERIC = {30, 201, 5}
MISS = -5.0


def scored():
    data = Path(PANEL).read_bytes()
    n = struct.unpack('<Q', data[:8])[0]
    meta = json.loads(data[8:8 + n])
    a, b = meta['offsets']['data_offsets']
    offsets = torch.frombuffer(bytearray(data[8 + n + a: 8 + n + b]), dtype=torch.int64)
    a, b = meta['score']['data_offsets']
    score = torch.frombuffer(bytearray(data[8 + n + a: 8 + n + b]), dtype=torch.uint8)
    return offsets, score


def report(name, rows):
    rows = sorted(rows)
    print(name, len(rows), 'at token positions', sorted(r[1] for r in rows)[:30], flush=True)
    print(json.dumps(rows[:30]), flush=True)


def main():
    cut = sys.argv.index('--')
    captures, parts = sys.argv[1:cut], sys.argv[cut + 1:]
    refs = [load_file(p) for p in parts]
    pid = torch.cat([r['ids'] for r in refs]).long()
    plp = torch.cat([r['logprobs'] for r in refs]).float()
    nll = torch.cat([r['nll'] for r in refs]).float()
    lens = torch.cat([r['lens'] for r in refs]).tolist()
    offsets, score = scored()
    token = []
    for i, k in enumerate(lens):
        have = [t for t in range(1, int(offsets[i + 1] - offsets[i])) if score[int(offsets[i]) + t]]
        assert len(have) == k
        token.extend((i, t) for t in have)

    def blank(ids, chosen):
        hit = (ids[:, :3].unsqueeze(-1) == torch.tensor(sorted(GENERIC)).view(1, 1, 3)).any(-1).all(-1) & (chosen < MISS)
        return [token[j] for j in hit.nonzero().flatten().tolist()]
    report('reference', blank(pid, -nll))
    for capture in captures:
        ids, chosen = [], []
        for i, k in enumerate(lens):
            q, _, c = read_seq(Path(capture) / f'seq-{i:04d}.bin')
            ids.append(q[:, :3])
            chosen.append(c)
        report(capture[-30:], blank(torch.cat(ids).long(), torch.cat(chosen)))


if __name__ == '__main__':
    main()
