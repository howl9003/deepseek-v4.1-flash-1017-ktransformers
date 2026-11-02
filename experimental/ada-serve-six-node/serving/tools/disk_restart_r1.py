"""Does a session survive a server restart? (Claude, 2 Nov 2026). Stage a: run each prompt, save its ids and reply. Restart the server. Stage b: send the next turn, which should read from disk, then clear the store and compute the same
turn cold; the two must be equal bit for bit. stdlib only, /usr/bin/python3. argv: a or b, working folder."""
import array
import json
from pathlib import Path
import struct
import sys
import time
import urllib.request

URL = 'http://127.0.0.1:18209'
PANEL = Path('/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors')
LENGTHS = (8192, 8192, 40960)
START = 250000
NEW = 32
EXTEND = 64


def tokens():
    data = PANEL.read_bytes()
    n = struct.unpack('<Q', data[:8])[0]
    meta = json.loads(data[8:8 + n])
    a, b = meta['tokens']['data_offsets']
    v = array.array('i')
    v.frombytes(data[8 + n + a: 8 + n + b])
    return v


def post(path, body=None):
    data = b'{}' if body is None else json.dumps(body).encode()
    request = urllib.request.Request(URL + path, data, {'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=1800) as response:
        raw = response.read()
    return json.loads(raw) if body is not None else raw.decode()


def run(ids):
    t0 = time.perf_counter()
    reply = post('/generate', dict(input_ids=ids, sampling_params=dict(max_new_tokens=NEW, temperature=0, ignore_eos=True), return_logprob=True, top_logprobs_num=8))
    meta = reply['meta_info']
    return dict(ids=reply['output_ids'], logprobs=meta['output_token_logprobs'], top=meta['output_top_logprobs'], cached=meta.get('cached_tokens'), details=meta.get('cached_tokens_details'), prefetch_wait=meta.get('storage_prefetch_wait'),
                first_s=meta.get('prefill_finished_time', 0) - meta.get('forward_entry_time', 0), wall_s=time.perf_counter() - t0)


def main():
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    saved = out / 'SESSIONS.json'
    if stage == 'a':
        assert not saved.exists()
        flat, start, rows = tokens(), START, []
        for length in LENGTHS:
            ids = flat[start: start + length].tolist()
            start += length
            first = run(ids)
            follow = ids + first['ids'] + flat[start: start + EXTEND].tolist()
            start += EXTEND
            rows.append(dict(length=length, follow=follow, first_s=first['first_s']))
        time.sleep(20)      # disk writes run behind the reply
        saved.write_text(json.dumps(rows))
        print('saved', [r['length'] for r in rows], [round(r['first_s'], 1) for r in rows])
        return
    rows, result = json.loads(saved.read_text()), []
    backs = [run(row['follow']) for row in rows]     # all reads first: the clear below takes every row with it
    print('reads done', [b['cached'] for b in backs], flush=True)
    post('/hicache/storage-backend/clear')
    for row, back in zip(rows, backs):
        post('/flush_cache')
        cold = run(row['follow'])
        result.append(dict(length=row['length'], from_disk={k: back[k] for k in ('cached', 'details', 'prefetch_wait', 'first_s', 'wall_s')}, cold={k: cold[k] for k in ('cached', 'details', 'prefetch_wait', 'first_s', 'wall_s')},
                           equal=dict(ids=back['ids'] == cold['ids'], logprobs=back['logprobs'] == cold['logprobs'], top=back['top'] == cold['top'])))
        print(json.dumps(result[-1]), flush=True)
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
