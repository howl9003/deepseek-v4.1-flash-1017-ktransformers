"""Reference panel capture on the served build (Claude, 1 Nov 2026). No timing: scores sudoingX's public panel teacher-forced through /generate, keeps each position's top 256 for score_panel.py, then checks
that decode and prefill give the same bits on the same tokens, since the capture only exercises prefill. stdlib only: this runs under /usr/bin/python3. The status string is the one the frozen controller expects."""
import argparse
import array
import hashlib
import json
import os
import struct
import urllib.request
from pathlib import Path

PANEL = Path('/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors')
PANEL_SHA = '39e0db2f0b58aa432a512aed7b425e3feb1b424c9e02cc315be39bc9b7b71cf2'
TOP = 256
CHECKS = 8
START = 256
NEW = 32
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def panel():
    assert sha(PANEL) == PANEL_SHA
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--harness')
    parser.add_argument('--server-pid')
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {55}
    assert args.namespace == 'phase-g-g1v-adapanel-r1'
    launch = json.loads((args.trial / 'LAUNCH.json').read_text())
    assert launch['environment']['SGLANG_ROUTE3_ENGRAM'] == '1' and launch['environment']['SGLANG_SM89_ATTN_STAGED'] == '1'
    out = args.trial / ('supplemental-' + args.namespace)
    out.mkdir(exist_ok=False)
    result = dict(status='RUNNING', role='reference panel capture, not a timing run', top=TOP, conversations=0, positions=0, sequences=[], decode_vs_prefill=[])
    write(out / 'RESULT.json', result)
    try:
        tokens, offsets, score = panel()
        count = len(offsets) - 1
        for i in range(count):
            ids = tokens[offsets[i]: offsets[i + 1]].tolist()
            meta = capture(ids)
            selected, top = meta['input_token_logprobs'], meta['input_top_logprobs']
            assert len(selected) == len(ids) and len(top) == len(ids), (i, len(selected), len(top), len(ids))
            positions = [t for t in range(1, len(ids)) if score[offsets[i] + t]]     # token t, predicted from position t - 1
            keep_id, keep_lp, chosen = array.array('i'), array.array('f'), array.array('f')
            for t in positions:
                row = top[t]
                assert len(row) == TOP
                keep_id.extend(e[1] for e in row)
                keep_lp.extend(e[0] for e in row)
                chosen.append(selected[t][0])
            with (out / f'seq-{i:04d}.bin').open('xb') as f:
                f.write(struct.pack('<qq', len(positions), TOP))
                keep_id.tofile(f)
                keep_lp.tofile(f)
                chosen.tofile(f)
            result['conversations'] = i + 1
            result['positions'] += len(positions)
            if i % 10 == 0:
                write(out / 'RESULT.json', result)
        for i in range(CHECKS):
            ids = tokens[offsets[i * 20]: offsets[i * 20 + 1]].tolist()[:START]
            body = dict(input_ids=ids, sampling_params=dict(max_new_tokens=NEW, temperature=0, ignore_eos=True), return_logprob=True, top_logprobs_num=8)
            decoded = post(body)
            made = decoded['output_ids'][:NEW]
            lp = [e[0] for e in decoded['meta_info']['output_token_logprobs']]
            forced = capture(ids + made[:-1])['input_token_logprobs']
            again = [forced[START + k][0] for k in range(NEW)]
            equal = sum(a == b for a, b in zip(lp, again))
            result['decode_vs_prefill'].append(dict(sequence=i * 20, positions=NEW, equal=equal, largest=max(abs(a - b) for a, b in zip(lp, again))))
            write(out / 'RESULT.json', result)
        result.update(status='PERFORMANCE_COMPLETE_NOT_PR6_ACCEPTANCE')
        write(out / 'RESULT.json', result)
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        write(out / 'RESULT.json', result)
        raise


if __name__ == '__main__':
    main()
