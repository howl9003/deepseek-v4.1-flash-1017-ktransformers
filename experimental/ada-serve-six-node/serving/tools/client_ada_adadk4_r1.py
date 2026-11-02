"""Serving gate with the disk tier on (Claude, 2 Nov 2026): the Engram gate client plus disk_stage below, which flushes the GPU and host tiers and checks that replies read back from disk equal the cold reply bit for bit.

Ada serving gate on the staggered spill (Claude, 31 Oct 2026): the tree r6 gate client on the arm 0 selectors, stagger and round worker on.

Host-reload gate on the final Phase G tree r6, prefill path A on (Claude, 30 Oct 2026). The first gate ran before G1p existed.

Phase G G1v arm 2 serving gate (Claude, 30 Oct 2026): the unchanged feature client on arm 2's serving load.

Its session, replay and queue stages are the replay fixtures and its KV stage is the frozen host-reload gate
(kv_checks.py). No timing and no numerical set here: the ledger ran on the numerical profile. The tier comparisons are
recorded for review, never turned into a tolerance.
"""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import urllib.request

HOST_RELOAD = 'eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


PANEL = Path('/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors')
LENGTHS = (1024, 8192, 40960, 131072)
EXTEND = 64
NEW = 32
SETTLE_S = 180


def tokens():
    data = PANEL.read_bytes()
    n = struct.unpack('<Q', data[:8])[0]
    meta = json.loads(data[8:8 + n])
    a, b = meta['tokens']['data_offsets']
    v = array.array('i')
    v.frombytes(data[8 + n + a: 8 + n + b])
    return v


def post(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request('http://127.0.0.1:18209' + path, data, {'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=1800) as response:
        raw = response.read()
    return json.loads(raw) if body is not None else raw.decode()


def run(ids):
    t0 = time.perf_counter()
    reply = post('/generate', dict(input_ids=ids, sampling_params=dict(max_new_tokens=NEW, temperature=0, ignore_eos=True), return_logprob=True, top_logprobs_num=8))
    meta = reply['meta_info']
    first = meta.get('prefill_finished_time', 0) - meta.get('forward_entry_time', 0)
    return dict(ids=reply['output_ids'], logprobs=meta['output_token_logprobs'], top=meta['output_top_logprobs'], cached=meta.get('cached_tokens'),
                details=meta.get('cached_tokens_details'), prefetch_wait=meta.get('storage_prefetch_wait'), first_s=first, wall_s=time.perf_counter() - t0)


def store(root):
    files = [p for p in root.rglob('*') if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def settle(root):
    """Writes to disk run behind the reply. Wait for the store to stop growing."""
    last, still, t0 = None, 0, time.perf_counter()
    while time.perf_counter() - t0 < SETTLE_S:
        now = store(root)
        still = still + 1 if now == last else 0
        if still >= 5:
            break
        last = now
        time.sleep(1)
    return dict(files=last[0], bytes=last[1], seconds=time.perf_counter() - t0)


def same(a, b):
    return dict(ids=a['ids'] == b['ids'], logprobs=a['logprobs'] == b['logprobs'], top=a['top'] == b['top'])


def summary(row):
    return {k: row[k] for k in ('cached', 'details', 'prefetch_wait', 'first_s', 'wall_s')}


def disk_stage(T, out, result):
    """The real case: a session's next turn, after the GPU and host tiers have gone. Compared bit for bit with the same turn computed cold."""
    root = Path(read(T / 'LAUNCH.json')['environment']['SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR'])
    flat, start = tokens(), 0
    rows = []
    result['disk'] = rows
    for length in LENGTHS:
        ids = (flat[start: start + length]).tolist()
        assert len(ids) == length, (start, len(ids))
        start += length
        first = run(ids)
        stored = settle(root)
        follow = ids + first['ids'] + flat[start: start + EXTEND].tolist()
        start += EXTEND
        message = post('/flush_cache?keep_storage=true')
        back = run(follow)
        warm = run(follow)
        post('/flush_cache')
        left = store(root)
        cold = run(follow)
        post('/flush_cache')
        rows.append(dict(length=length, stored=stored, left_after_clear=left, flush=message.strip()[:60], first=summary(first), cold=summary(cold), from_disk=summary(back), warm=summary(warm),
                         disk_equal=same(cold, back), warm_equal=same(cold, warm)))
        write(out / 'RESULT.json', result)
    result['disk_status'] = 'PASS' if all(all(r['disk_equal'].values()) and all(r['warm_equal'].values()) and ((r['from_disk']['details'] or {}).get('storage') or 0) >= 2 * 256 for r in rows) else 'FAIL'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--harness')
    parser.add_argument('--server-pid', required=True)
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {55}
    assert args.namespace == 'phase-g-g1v-adadk4-r1'
    T = args.trial
    out = T / ('supplemental-' + args.namespace)
    out.mkdir(exist_ok=False)
    assert sha(T / 'client-tools/kv_checks.py') == HOST_RELOAD
    environment = read(T / 'LAUNCH.json')['environment']
    assert environment.get('SGLANG_FP8_PATHA_DENSE', '0') == '0' and environment.get('SGLANG_SINGLE_ROUTE', '0') == '0'
    assert environment['KT_EXPERT_ARENA_SPILL_STAGGER'] == '1' and environment['KT_KV_HOT_DYN_WORKER'] == '1'
    assert environment['KT_GPU_STREAM_PATHA'] == '1'
    assert environment['SGLANG_FP8_ROUTE3_CUDA'] == '1' and environment['SGLANG_CACHE_TTL'] == '1' and environment['KT_PREFILL_SWA_LEND'] == '0'
    profile = read(T / 'EFFECTIVE_PROFILE.json')
    features = T / ('features-' + args.namespace)
    command = ['/usr/bin/python3', '-B', str(Path(__file__).resolve().parent / 'disk-client-r1/feature_client.py'), '--host', '127.0.0.1', '--port', '18209',
               '--model', 'default', '--context', str(profile['server_args']['context_length']),
               '--profile', str(T / 'EFFECTIVE_PROFILE.json'), '--source-root', str(T / 'recipe/third_party/sglang'),
               '--server-pid', args.server_pid, '--output', str(features), '--deadline', '1800',
               '--exclusive-caller-owned-server', '--execute']
    env = dict(os.environ, URL='http://127.0.0.1:18209', SGLANG_PORT='18209', PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONPATH', 'FEATURE_ORIGINS_CONFIG', 'FEATURE_ORIGINS_CONFIG_SHA256'):
        env.pop(key, None)
    result = dict(status='RUNNING', host_reload_sha256=HOST_RELOAD, command=command, numerical_acceptance='REVIEW_REQUIRED', pr6='HELD')
    write(out / 'RESULT.json', result)
    try:
        with (out / 'features.log').open('x') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1875)
        result['exit_code'] = run.returncode
        write(out / 'RESULT.json', result)
        assert run.returncode == 0, run.returncode
        report = read(features / 'report.json')
        assert report['status'] == 'MEASURED'
        kv = read(features / 'kv/KV_CHECKS.json')
        assert kv['status'] == 'PASS' and all(kv['checks'].values())
        result.update(status='ADA_SERVING_GATE_MEASURED_REVIEW_REQUIRED', kv_status=kv['status'], kv_checks=kv['checks'], kv_comparisons=kv['comparisons'],
                      history_and_lifecycle_checks=report['history_and_lifecycle_checks'],
                      all_replay_selected_ids_equal=report['all_replay_selected_ids_equal'],
                      replay_comparisons=[row['comparison'] for row in report['replays']], report_sha256=sha(features / 'report.json'))
        write(out / 'RESULT.json', result)
        disk_stage(T, out, result)
        write(out / 'RESULT.json', result)
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        write(out / 'RESULT.json', result)
        raise


if __name__ == '__main__':
    main()
