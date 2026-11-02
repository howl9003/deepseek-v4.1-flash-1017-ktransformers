"""Positions where a capture lands far from the reference, by the panel's KL (Claude, 2 Nov 2026). Same formula as score_panel.py. Prints the tail counts and the worst positions with each side's top tokens. argv: output JSON, capture dirs..., --, reference parts."""
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from score_panel import read_seq

PANEL = '/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors'


def scored():
    data = Path(PANEL).read_bytes()
    n = struct.unpack('<Q', data[:8])[0]
    meta = json.loads(data[8:8 + n])
    a, b = meta['offsets']['data_offsets']
    offsets = torch.frombuffer(bytearray(data[8 + n + a: 8 + n + b]), dtype=torch.int64)
    a, b = meta['score']['data_offsets']
    score = torch.frombuffer(bytearray(data[8 + n + a: 8 + n + b]), dtype=torch.uint8)
    return offsets, score


def main():
    cut = sys.argv.index('--')
    out, captures, parts = sys.argv[1], sys.argv[2:cut], sys.argv[cut + 1:]
    refs = [load_file(p) for p in parts]
    pid = torch.cat([r['ids'] for r in refs]).long()
    plp = torch.cat([r['logprobs'] for r in refs]).float()
    lens = torch.cat([r['lens'] for r in refs]).tolist()
    offsets, score = scored()
    report = {}
    for capture in captures:
        rows, o = [], 0
        for i, k in enumerate(lens):
            qid, qlp, _ = read_seq(Path(capture) / f'seq-{i:04d}.bin')
            p_id, p_lp = pid[o: o + k], plp[o: o + k]
            o += k
            if k == 0:
                continue
            match = p_id.unsqueeze(-1) == qid.unsqueeze(1)
            found = match.any(-1)
            q = (match.float() * qlp.exp().unsqueeze(1)).sum(-1)
            q = torch.where(found, q, qlp.exp().amin(-1, keepdim=True))
            P = p_lp.exp()
            p_rest = (1 - P.sum(-1)).clamp_min(1e-12)
            q_rest = (1 - q.sum(-1)).clamp_min(1e-12)
            kl = (P * (p_lp - q.clamp_min(1e-30).log())).sum(-1) + p_rest * (p_rest.log() - q_rest.log())
            token = [t for t in range(1, int(offsets[i + 1] - offsets[i])) if score[int(offsets[i]) + t]]
            for p in (kl > 1.5).nonzero().flatten().tolist():
                rows.append(dict(kl=round(kl[p].item(), 3), conversation=i, scored_position=p, token_position=token[p], reference_top=[(p_id[p, c].item(), round(p_lp[p, c].item(), 3)) for c in range(3)],
                                 ours_top=[(qid[p, c].item(), round(qlp[p, c].item(), 3)) for c in range(4)]))
        rows.sort(key=lambda r: -r['kl'])
        report[capture] = dict(over={str(t): sum(r['kl'] > t for r in rows) for t in (1.5, 2, 3, 5, 8)}, worst=rows[:14])
        print(capture[-30:], report[capture]['over'], flush=True)
        for r in rows[:14]:
            print(json.dumps(r), flush=True)
    json.dump(report, open(out, 'x'), indent=1)


if __name__ == '__main__':
    main()
