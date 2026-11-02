"""Score a served capture against reference parts with sudoingX's KL: the native top-128 support plus one tail bucket. Same formula as p4_kld.py's kl_vs_ref, but on what /generate returns, our top 256.
A native id outside our top 256 takes our smallest kept probability, an upper bound on Q there; the count is reported. argv: capture dir, output JSON, reference part files in sequence order. Needs panel_index.json beside the panel."""
import json
import math
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

INDEX = '/scratch/aster/bench/reference-panel-r1/panel-data/panel_index.json'


def read_seq(path):
    data = Path(path).read_bytes()
    n, top = struct.unpack('<qq', data[:16])
    a = 16 + 4 * n * top
    ids = torch.frombuffer(bytearray(data[16:a]), dtype=torch.int32).reshape(n, top)
    lp = torch.frombuffer(bytearray(data[a: a + 4 * n * top]), dtype=torch.float32).reshape(n, top)
    chosen = torch.frombuffer(bytearray(data[a + 4 * n * top:]), dtype=torch.float32)
    return ids, lp, chosen


def main():
    capture, out, parts = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    refs = [load_file(p) for p in parts]
    pid = torch.cat([r['ids'] for r in refs]).long()
    plp = torch.cat([r['logprobs'] for r in refs]).float()
    nll_ref = torch.cat([r['nll'] for r in refs]).float()
    lens = torch.cat([r['lens'] for r in refs]).tolist()
    index = json.load(open(INDEX))
    assert len(lens) == len(index)
    o, kls, agree, nll, missing, by = 0, [], [], [], 0, {}
    for i, k in enumerate(lens):
        qid, qlp, chosen = read_seq(capture / f'seq-{i:04d}.bin')
        assert qid.size(0) == k, (i, qid.size(0), k)
        p_id, p_lp = pid[o: o + k], plp[o: o + k]
        o += k
        match = p_id.unsqueeze(-1) == qid.unsqueeze(1)                # [k, 128, 256]
        found = match.any(-1)
        q = (match.float() * qlp.exp().unsqueeze(1)).sum(-1)
        q = torch.where(found, q, qlp.exp().amin(-1, keepdim=True))
        missing += int((~found).sum())
        P = p_lp.exp()
        p_rest = (1 - P.sum(-1)).clamp_min(1e-12)
        q_rest = (1 - q.sum(-1)).clamp_min(1e-12)
        kl = (P * (p_lp - q.clamp_min(1e-30).log())).sum(-1) + p_rest * (p_rest.log() - q_rest.log())
        hit = (qid[:, 0] == p_id[:, 0]).float()
        kls.append(kl)
        agree.append(hit)
        nll.append(-chosen)
        dom = index[i]['domain'].split('_')[0]
        by.setdefault(dom, []).append((kl.sum().item(), hit.sum().item(), k))
    kl, top1, ours = torch.cat(kls), torch.cat(agree), torch.cat(nll)
    report = dict(positions=int(kl.numel()), kld_mean=kl.mean().item(), kld_p99=kl.quantile(0.99).item(), top1_agreement=top1.mean().item(), nll_reference=nll_ref.mean().item(), nll_served=ours.mean().item(),
                  ppl_reference=math.exp(nll_ref.mean().item()), ppl_served=math.exp(ours.mean().item()), native_ids_outside_top256=missing,
                  by_domain={d: dict(kld=sum(x[0] for x in v) / sum(x[2] for x in v), top1=sum(x[1] for x in v) / sum(x[2] for x in v), positions=sum(x[2] for x in v)) for d, v in sorted(by.items())})
    json.dump(report, open(out, 'x'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
