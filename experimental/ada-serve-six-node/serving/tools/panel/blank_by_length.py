"""Does a context-free row depend on what follows it in the prompt? Asks a resident server for the same row with the prompt cut at growing lengths (Claude, 2 Nov 2026). Read-only against the server, stdlib only. argv: output JSON."""
import array
import json
import struct
import sys
import urllib.request
from pathlib import Path

PANEL = Path('/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors')
TOP = 6
ROWS = ((157, 47), (64, 69), (41, 104), (152, 61), (76, 101), (136, 64))
EXTRA = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)


def panel():
    data = PANEL.read_bytes()
    n = struct.unpack('<Q', data[:8])[0]
    meta = json.loads(data[8:8 + n])

    def part(name, code):
        a, b = meta[name]['data_offsets']
        v = array.array(code)
        v.frombytes(data[8 + n + a: 8 + n + b])
        return v
    return part('tokens', 'i'), part('offsets', 'q')


def post(body):
    request = urllib.request.Request('http://127.0.0.1:18209/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def main():
    tokens, offsets = panel()
    result = []
    for c, t in ROWS:
        ids = tokens[offsets[c]: offsets[c + 1]].tolist()
        seen = set()
        for extra in EXTRA:
            length = min(t + extra, len(ids))
            if length in seen:
                continue
            seen.add(length)
            meta = post(dict(input_ids=ids[:length], sampling_params=dict(max_new_tokens=1, temperature=0), return_logprob=True, logprob_start_len=0, top_logprobs_num=TOP))['meta_info']
            row = dict(conversation=c, token=t, length=length, selected=meta['input_token_logprobs'][t][0], top=[(e[1], round(e[0], 3)) for e in meta['input_top_logprobs'][t][:4]])
            result.append(row)
            print(json.dumps(row), flush=True)
    with open(sys.argv[1], 'x', encoding='utf-8') as f:
        json.dump(result, f, indent=1)


if __name__ == '__main__':
    main()
