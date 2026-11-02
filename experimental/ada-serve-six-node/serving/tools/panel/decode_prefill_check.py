"""Decode against prefill on a resident server (Claude, 2 Nov 2026). Same tokens as the FP4 check: 256 tokens (plain prefill) and 1,024 (streamed prefill), eight conversations each, 32 generated tokens, then a teacher-forced prefill of the same tokens.
Then repeats a few conversations against a saved panel capture, and asks for the token at the panel's context-free rows with the prompt cut just before it. Read-only against the server, stdlib only. argv: saved capture dir, output JSON."""
import array
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

PANEL = Path('/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors')
PANEL_SHA = '39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2'
TOP = 256
CHECKS = 8
STARTS = (256, 1024)
NEW = 32
REPEATS = (76, 136, 41, 152, 143, 150, 0)
BLANK = ((41, 104), (76, 101), (136, 64), (152, 61))     # rows that came out context-free in a panel capture


def panel():
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest() == PANEL_SHA
    data = PANEL.read_bytes()
    n = struct.unpack('<Q', data[:8])[0]
    meta = json.loads(data[8:8 + n])

    def part(name, code):
        a, b = meta[name]['data_offsets']
        v = array.array(code)
        v.frombytes(data[8 + n + a: 8 + n + b])
        return v
    return part('tokens', 'i'), part('offsets', 'q'), part('score', 'B')


def post(body):
    request = urllib.request.Request('http://127.0.0.1:18209/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def capture(ids):
    return post(dict(input_ids=ids, sampling_params=dict(max_new_tokens=1, temperature=0), return_logprob=True, logprob_start_len=0, top_logprobs_num=TOP))['meta_info']


def main():
    saved, out = Path(sys.argv[1]), Path(sys.argv[2])
    tokens, offsets, score = panel()
    count = len(offsets) - 1
    result = dict(decode_vs_prefill=[], repeats=[], blank_rows=[])
    for start in STARTS:
        long_enough = [c for c in range(count) if offsets[c + 1] - offsets[c] >= start]
        for i in long_enough[:: max(1, len(long_enough) // CHECKS)][:CHECKS]:
            ids = tokens[offsets[i]: offsets[i + 1]].tolist()[:start]
            decoded = post(dict(input_ids=ids, sampling_params=dict(max_new_tokens=NEW, temperature=0, ignore_eos=True), return_logprob=True, top_logprobs_num=8))
            made = decoded['output_ids'][:NEW]
            lp = [e[0] for e in decoded['meta_info']['output_token_logprobs']]
            forced = capture(ids + made)['input_token_logprobs']
            again = [forced[start + k][0] for k in range(NEW)]
            result['decode_vs_prefill'].append(dict(start=start, sequence=i, positions=NEW, equal=sum(a == b for a, b in zip(lp, again)), largest=max(abs(a - b) for a, b in zip(lp, again)), mean=sum(abs(a - b) for a, b in zip(lp, again)) / NEW))
            print(json.dumps(result['decode_vs_prefill'][-1]), flush=True)
    for i in REPEATS:
        ids = tokens[offsets[i]: offsets[i + 1]].tolist()
        selected = capture(ids)['input_token_logprobs']
        positions = [t for t in range(1, len(ids)) if score[offsets[i] + t]]
        data = (saved / f'seq-{i:04d}.bin').read_bytes()
        n, top = struct.unpack('<qq', data[:16])
        kept = array.array('f')
        kept.frombytes(data[16 + 8 * n * top:])
        now = array.array('f', [selected[t][0] for t in positions])
        assert len(now) == len(kept), (i, len(now), len(kept))
        result['repeats'].append(dict(sequence=i, positions=len(now), equal=sum(a == b for a, b in zip(now, kept)), largest=max(abs(a - b) for a, b in zip(now, kept))))
        print(json.dumps(result['repeats'][-1]), flush=True)
    for i, t in BLANK:      # the prompt stops just before the token, so this row has nothing after it
        ids = tokens[offsets[i]: offsets[i] + t].tolist()
        decoded = post(dict(input_ids=ids, sampling_params=dict(max_new_tokens=1, temperature=0, ignore_eos=True), return_logprob=True, top_logprobs_num=6))
        result['blank_rows'].append(dict(sequence=i, token_position=t, expected=tokens[offsets[i] + t], top=decoded['meta_info']['output_top_logprobs'][0]))
        print(json.dumps(result['blank_rows'][-1]), flush=True)
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
