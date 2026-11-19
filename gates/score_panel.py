"""Score a served capture against reference parts with sudoingX's KL: the native top-128 support plus one tail bucket. Same formula as p4_kld.py's kl_vs_ref, but on what /generate returns, our top 256.
A native id outside our top 256 takes our smallest kept probability, an upper bound on Q there; the count is reported. Arguments are capture directory, new output JSON, panel index and reference parts in sequence order."""
import hashlib
import json
import os
import math
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

def read_seq(path):
    data = Path(path).read_bytes()
    n, top = struct.unpack('<qq', data[:16])
    if n < 0 or top != 256 or len(data) != 16 + n * (8 * top + 4):
        raise ValueError(f"Invalid capture dimensions or byte count in {path}")
    if n == 0:      # a conversation with no scored position
        return torch.empty(0, top, dtype=torch.int32), torch.empty(0, top), torch.empty(0)
    a = 16 + 4 * n * top
    ids = torch.frombuffer(bytearray(data[16:a]), dtype=torch.int32).reshape(n, top)
    lp = torch.frombuffer(bytearray(data[a: a + 4 * n * top]), dtype=torch.float32).reshape(n, top)
    chosen = torch.frombuffer(bytearray(data[a + 4 * n * top:]), dtype=torch.float32)
    return ids, lp, chosen


def score_rows(pid, plp, o, k, qid, qlp):
    p_id, p_lp = pid[o: o + k], plp[o: o + k]
    match = p_id.unsqueeze(-1) == qid.unsqueeze(1)                # [k, 128, 256]
    found = match.any(-1)
    q = (match.float() * qlp.exp().unsqueeze(1)).sum(-1)
    q = torch.where(found, q, qlp.exp().amin(-1, keepdim=True))
    outside = int((~found).sum())
    P = p_lp.exp()
    p_rest = (1 - P.sum(-1)).clamp_min(1e-12)
    q_rest = (1 - q.sum(-1)).clamp_min(1e-12)
    kl = (P * (p_lp - q.clamp_min(1e-30).log())).sum(-1) + p_rest * (p_rest.log() - q_rest.log())
    hit = (qid[:, 0] == p_id[:, 0]).float()
    return kl, hit, outside


def main():
    capture, out, index_path, *parts = sys.argv[1:]
    capture = Path(capture)
    baseline = Path(os.environ['C4_PAIRED_CAPTURE']) if os.environ.get('C4_PAIRED_CAPTURE') else None
    baseline_pins = json.loads(Path(__file__).with_name('paired-captures.json').read_bytes())
    paired_kl = paired_nll = 0.0
    refs = [load_file(p) for p in parts]
    pid = torch.cat([r['ids'] for r in refs]).long()
    plp = torch.cat([r['logprobs'] for r in refs]).float()
    nll_ref = torch.cat([r['nll'] for r in refs]).float()
    lens = torch.cat([r['lens'] for r in refs]).tolist()
    index = json.loads(Path(index_path).read_bytes())
    assert len(lens) == len(index) == 199 and sum(lens) == 254846
    o, kls, agree, nll, missing, by = 0, [], [], [], 0, {}
    for i, k in enumerate(lens):
        qid, qlp, chosen = read_seq(capture / f'seq-{i:04d}.bin')
        assert qid.size(0) == k, (i, qid.size(0), k)
        kl, hit, outside = score_rows(pid, plp, o, k, qid, qlp)
        missing += outside
        if baseline:
            path = baseline / f"seq-{i:04d}.bin"
            if hashlib.sha256(path.read_bytes()).hexdigest() != baseline_pins[path.name]:
                raise ValueError(f"Paired capture differs at conversation {i}")
            aid, alp, achosen = read_seq(path)
            akl, _, _ = score_rows(pid, plp, o, k, aid, alp)
            paired_kl += (kl.double() - akl.double()).sum().item()
            paired_nll += (achosen.double() - chosen.double()).sum().item()
        o += k
        kls.append(kl)
        agree.append(hit)
        nll.append(-chosen)
        dom = index[i]['domain'].split('_')[0]
        by.setdefault(dom, []).append((kl.sum().item(), hit.sum().item(), k))
    kl, top1, ours = torch.cat(kls), torch.cat(agree), torch.cat(nll)
    report = dict(positions=int(kl.numel()), kld_mean=kl.mean().item(), kld_p99=kl.quantile(0.99).item(), top1_agreement=top1.mean().item(), nll_reference=nll_ref.mean().item(), nll_served=ours.mean().item(),
                  ppl_reference=math.exp(nll_ref.mean().item()), ppl_served=math.exp(ours.mean().item()), native_ids_outside_top256=missing,
                  by_domain={d: dict(kld=sum(x[0] for x in v) / sum(x[2] for x in v), top1=sum(x[1] for x in v) / sum(x[2] for x in v), positions=sum(x[2] for x in v)) for d, v in sorted(by.items())})
    if baseline:
        report.update(paired_mean_kl=paired_kl/o, paired_mean_nll=paired_nll/o)
        if not (abs(paired_kl/o)<=1e-7 and abs(paired_nll/o)<=1e-7):
            raise SystemExit("Paired panel differs from the reference")
    json.dump(report, open(out, 'x'), indent=1)
    print(json.dumps(report, indent=1))
    if not (0.009782 <= report['kld_mean'] <= 0.009784 and 0.47020 <= report['nll_served'] <= 0.47022 and report['top1_agreement'] == 0.9703232645988464):
        raise SystemExit('Panel score outside the measured gates')


if __name__ == '__main__':
    main()
