"""Ada serving gate on the staggered spill (Claude, 31 Oct 2026): the tree r6 gate client on the arm 0 selectors, stagger and round worker on.

Host-reload gate on the final Phase G tree r6, prefill path A on (Claude, 30 Oct 2026). The first gate ran before G1p existed.

Phase G G1v arm 2 serving gate (Claude, 30 Oct 2026): the unchanged feature client on arm 2's serving load.

Its session, replay and queue stages are the replay fixtures and its KV stage is the frozen host-reload gate
(kv_checks.py). No timing and no numerical set here: the ledger ran on the numerical profile. The tier comparisons are
recorded for review, never turned into a tolerance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HOST_RELOAD = 'eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--harness')
    parser.add_argument('--server-pid', required=True)
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {55}
    assert args.namespace == 'phase-g-g1v-adaygate-r1'
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
    command = ['/usr/bin/python3', '-B', str(T / 'client-tools/feature_client.py'), '--host', '127.0.0.1', '--port', '18209',
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
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        write(out / 'RESULT.json', result)
        raise


if __name__ == '__main__':
    main()
