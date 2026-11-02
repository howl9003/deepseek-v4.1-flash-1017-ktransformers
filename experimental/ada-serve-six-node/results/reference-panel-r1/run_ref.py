"""DeepSeek's reference forward on this card, one layer at a time (Claude, 1 Nov 2026), through sudoingX's Streamer on local_ckpt. Two jobs:

  --pieces        the 24 pieces of 480 tokens our identity runs score. Writes nll.py's format, one list of log-probabilities a piece, so it pairs with any served-nll.json.
  --panel FILE    a panel from build_panel.py, sequences start .. start + count. Writes ids, logprobs (top 128), nll and lens as p4_kld.py --mode ref does, in FP32.

A panel run needs parts: Streamer keeps every sequence's stream on the device, about 41 KB a token, so a part must stay well under the card. --moe-chunk changes how tokens batch through the MoE, for the noise floor.
"""
import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, '/scratch/aster/bench/reference-panel-r1')
import local_ckpt as L  # noqa: E402
from dsv41q.stream import Streamer  # noqa: E402

TOPK = 128
TOOLS = '/scratch/aster/bench/05yuki-main-00fd12a-candidate/tools'
W4A4 = False


def walk(seqs, moe_chunk, max_layers):
    torch.set_default_dtype(torch.bfloat16)
    s = Streamer(L.CKPT, seqs)
    if W4A4:
        import w4a4
        w4a4.install(s.model)
    blk = None
    layers = max_layers or s.args.n_layers
    for layer in range(layers):
        t = time.time()
        blk = s.load(layer)
        loaded = time.time()
        s.run_layer(blk, moe_chunk_tokens=moe_chunk)
        torch.cuda.synchronize()
        print(f'layer {layer:2d}: load {loaded - t:.1f}s, forward {time.time() - loaded:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)
        if layer < layers - 1:
            del blk
            torch.cuda.empty_cache()
    return s, blk


def pieces(a):
    sys.path.insert(0, TOOLS)
    import nll_pieces
    seqs = nll_pieces.pieces(nll_pieces.tokenizer(L.CKPT))
    s, blk = walk(seqs, a.moe_chunk, a.max_layers)
    out = []
    for i, lg in s.iter_logits(blk):
        ids = torch.tensor(seqs[i], device=lg.device)
        lp = torch.log_softmax(lg[:-1], -1).gather(-1, ids[1:, None]).squeeze(-1)
        out.append(lp.double().cpu().tolist())
    json.dump(out, open(a.out, 'x'))
    n = sum(len(x) for x in out)
    print(f'{n} tokens in {len(out)} pieces, mean NLL {-sum(sum(x) for x in out) / n:.4f}', flush=True)


def floor(a, s, blk, seqs, span, score, offs):
    """The reference against an earlier reference part: p4_kld.py's KL (native top 128 plus one tail bucket) on the full distribution, for the noise floor. Writes per-position kl, agree and nll."""
    ref = load_file(a.against)
    lens = ref['lens'].tolist()
    assert len(lens) == len(seqs)
    o, kls, agree, nll = 0, [], [], []
    for k, lg in s.iter_logits(blk):
        i = span[k]
        sc = score[offs[i]: offs[i + 1]].bool().to(lg.device)
        pos = sc[1:].nonzero().squeeze(-1)
        assert pos.numel() == lens[k]
        lq = torch.log_softmax(lg[pos].float(), -1)
        pid = ref['ids'][o: o + lens[k]].to(lg.device).long()
        plp = ref['logprobs'][o: o + lens[k]].to(lg.device).float()
        o += lens[k]
        P, Q = plp.exp(), lq.gather(-1, pid).exp()
        p_rest, q_rest = (1 - P.sum(-1)).clamp_min(1e-12), (1 - Q.sum(-1)).clamp_min(1e-12)
        kls.append(((P * (plp - Q.clamp_min(1e-30).log())).sum(-1) + p_rest * (p_rest.log() - q_rest.log())).float().cpu())
        agree.append((lq.argmax(-1) == pid[:, 0]).float().cpu())
        tgt = torch.tensor(seqs[k], device=lg.device)[1:][pos]
        nll.append((-lq.gather(-1, tgt[:, None]).squeeze(-1)).float().cpu())
    assert not os.path.exists(a.out)
    kl, top1, n = torch.cat(kls), torch.cat(agree), torch.cat(nll)
    save_file({'kl': kl, 'agree': top1, 'nll': n, 'lens': ref['lens']}, a.out)
    print(f'{kl.numel()} positions: KL {kl.mean().item():.5f}, top-1 {top1.mean().item():.5f}, NLL {n.mean().item():.5f} against {ref["nll"].mean().item():.5f}', flush=True)


def panel(a):
    d = load_file(a.panel)
    toks, offs, score = d['tokens'].tolist(), d['offsets'].tolist(), d['score']
    stop = min(a.start + (a.count or len(offs)), len(offs) - 1)
    span = range(a.start, stop)
    seqs = [toks[offs[i]: offs[i + 1]] for i in span]
    s, blk = walk(seqs, a.moe_chunk, a.max_layers)
    if a.against:
        floor(a, s, blk, seqs, span, score, offs)
        return
    ids, lps, nll = [], [], []
    for k, lg in s.iter_logits(blk):
        i = span[k]
        sc = score[offs[i]: offs[i + 1]].bool().to(lg.device)
        pos = sc[1:].nonzero().squeeze(-1)              # position t predicts token t + 1
        lp = torch.log_softmax(lg[pos], -1)
        top = torch.topk(lp, TOPK, -1)
        ids.append(top.indices.int().cpu())
        lps.append(top.values.float().cpu())
        tgt = torch.tensor(seqs[k], device=lg.device)[1:][pos]
        nll.append((-lp.gather(-1, tgt[:, None]).squeeze(-1)).float().cpu())
    lens = torch.tensor([x.size(0) for x in ids], dtype=torch.int64)
    assert not os.path.exists(a.out)
    save_file({'ids': torch.cat(ids), 'logprobs': torch.cat(lps), 'nll': torch.cat(nll), 'lens': lens}, a.out)
    print(f'{int(lens.sum())} scored positions, sequences {a.start}..{stop}, mean NLL {torch.cat(nll).mean().item():.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pieces', action='store_true')
    ap.add_argument('--panel')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=0)
    ap.add_argument('--against')
    ap.add_argument('--moe-chunk', type=int, default=65536)
    ap.add_argument('--max-layers', type=int, default=0)
    ap.add_argument('--w4a4', action='store_true', help='NVFP4 W4A4 expert activations, see w4a4.py')
    ap.add_argument('--mxfp4-engram', action='store_true', help='Engram rows through NVIDIA FP8 to MXFP4 rule, see local_ckpt.mxfp4_row')
    a = ap.parse_args()
    global W4A4
    W4A4 = a.w4a4
    L.MXFP4_ENGRAM = a.mxfp4_engram
    t = time.time()
    (pieces if a.pieces else panel)(a)
    print(f'{time.time() - t:.0f} s', flush=True)


if __name__ == '__main__':
    main()
