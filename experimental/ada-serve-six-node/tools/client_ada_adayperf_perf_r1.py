"""Ada baseline performance (Claude, 31 Oct 2026): the frozen benchmark and 4K rows on the C1 Route 3 path with prefill path A.

Phase G G1p performance, on arm (Claude, 30 Oct 2026): the arm-2 r5 client; the off arm is arm-2 performance r5.

Phase G G1v arm 2 performance r5 (Claude, 30 Oct 2026): tree r4 (row-blocked fixed wo_a) and the paired 4K rows of
client_r19_D2_repeated_performance_r1.py, which the guide's performance set lists and r2 to r4 did not run.

Phase G G1v arm 2 performance (Claude, 30 Oct 2026): client_r19_C1_repeated_performance_r2.py with the arm-2
namespace and selector checks (path A dense and the single route on, Route 3 off). Original docstring follows.

Two rounds of the unchanged historical client, for C1 run-to-run spread.

No timestamps or routing observers. Full-model numerical qualification happens
in a different process before this timing load. Each round has a new namespace.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

P = Path(__file__).resolve().parents[1]
CLIENT_SHA = '4635dede1a4d69366ffe6291275e7cb0f02f9a1d44b231afb5cb7612b408d812'
BENCH_SHA = '1c361c2010d7f69a8f1a93e33f26e7622bb684bd43a2de51b29b63a00b063f40'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--harness')
    parser.add_argument('--server-pid')
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {55}
    assert args.namespace == 'phase-g-g1v-adayperf-r1'
    assert sha(P / 'tools/client_fast_r1.py') == CLIENT_SHA
    assert sha(P / 'source/bench.py') == BENCH_SHA
    launch = json.loads((args.trial / 'LAUNCH.json').read_text())
    assert launch['environment']['SGLANG_FP8_PATHA_EXPERT_STATIC'] == '1'
    assert launch['environment'].get('SGLANG_FP8_PATHA_DENSE', '0') == '0' and launch['environment'].get('SGLANG_SINGLE_ROUTE', '0') == '0'
    assert launch['environment']['SGLANG_FP8_ROUTE3_CUDA'] == '1'
    assert launch['environment']['KT_GPU_STREAM_PATHA'] == '1' and launch['environment']['KT_GPU_STREAM_PATHA_FLAGS_LOG'] == '0'
    assert launch['environment']['KT_HOT_DYNAMIC'] == '0' and launch['environment']['KT_KV_HOT_DYN_WORKER'] == '1' and launch['environment']['KT_EXPERT_ARENA_SPILL_STAGGER'] == '1'
    assert 'KT_ROUTING_DUMP' not in launch['environment']
    out = args.trial / ('supplemental-' + args.namespace)
    out.mkdir(exist_ok=False)
    result = dict(status='RUNNING', frozen_client_sha256=CLIENT_SHA, bench_sha256=BENCH_SHA,
                  static_mode=launch['environment']['SGLANG_FP8_PATHA_EXPERT_STATIC'], rounds=[], prefill4096_rounds=[],
                  limitation='Two consecutive same-profile rounds on one load; matched method, not an interleaved statistical study.')
    write(out / 'RESULT.json', result)
    try:
        for index in range(2):
            # Same frozen benchmark, two matched 4K inputs (seeds 100/101), separate log from the round.
            command4k = ['/usr/bin/python3', '-B', str(P / 'source/bench.py'),
                         '--url', 'http://127.0.0.1:18209', '--prefill-only', '--lengths', '4096,4096']
            path4k = out / f'prefill4096-round{index}.log'
            with path4k.open('x') as log:
                run4k = subprocess.run(command4k, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            assert run4k.returncode == 0, '4K prefill benchmark failed'
            seconds4k = [float(v) for v in re.findall(r'prefill, .*?4096-token prompt: ([0-9.]+) s =', path4k.read_text())]
            assert len(seconds4k) == 2 and all(v > 0 for v in seconds4k)
            result['prefill4096_rounds'].append(dict(index=index, command=command4k, seconds=seconds4k, log_sha256=sha(path4k)))
            write(out / 'RESULT.json', result)
            namespace = args.namespace + '-round' + str(index)
            target = args.trial / ('supplemental-' + namespace)
            assert not target.exists()
            command = ['/usr/bin/python3', '-B', str(P / 'tools/client_fast_r1.py'),
                       '--trial', str(args.trial), '--namespace', namespace]
            start = time.monotonic()
            with (out / ('round' + str(index) + '.log')).open('x') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
            assert run.returncode == 0, (index, run.returncode)
            receipt = json.loads((target / 'RESULT.json').read_text())
            assert receipt['status'] == 'PERFORMANCE_COMPLETE_NOT_PR6_ACCEPTANCE'
            assert receipt['post_timing_repeat']['text_equal'] and receipt['post_timing_repeat']['selected_logprobs_exact']
            historical, breadth = (target / 'historical.log').read_text(), (target / 'breadth.log').read_text()
            rows = [dict(workload=name, tok_s=float(rate)) for name, rate in re.findall(r'decode, (.*?):.*?= ([0-9.]+) tok/s', breadth)]
            prefill = [float(value) for value in re.findall(r'prefill, .*?: ([0-9.]+) s =', historical)]
            assert len(rows) == 6 and len(prefill) == 2
            result['rounds'].append(dict(index=index, namespace=namespace, command=command,
                seconds=time.monotonic()-start, client_receipt_sha256=sha(target / 'RESULT.json'),
                breadth=rows, breadth_median_tok_s=statistics.median(row['tok_s'] for row in rows),
                japanese_median_tok_s=receipt['japanese_median_tokens_per_second'],
                prefill35000_seconds=prefill, post_timing_repeat=receipt['post_timing_repeat']))
            write(out / 'RESULT.json', result)
        medians = [row['breadth_median_tok_s'] for row in result['rounds']]
        result.update(status='PERFORMANCE_COMPLETE_NOT_PR6_ACCEPTANCE',
                      breadth_round_medians_tok_s=medians, breadth_median_tok_s=statistics.median(medians),
                      breadth_run_to_run_range_tok_s=max(medians)-min(medians))
        write(out / 'RESULT.json', result)
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        write(out / 'RESULT.json', result)
        raise


if __name__ == '__main__':
    main()
