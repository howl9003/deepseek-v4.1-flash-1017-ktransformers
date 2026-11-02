"""B1b ordered gates: capture hot64, prove cold-placement identity, then ledger/quality."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import urllib.request
from compare_r19_exact_scores_r1 import compare
from client_r19_bitneutral_identity_r1 import replay_gate

B = Path('/scratch/aster/bench')
P = B / 'six-node-performance-r1'
N = B / 'feature-trial-B-contract-route3-numerics-six-r1/supplemental-contract-route3-numerics-r1'
H = B / 'feature-trial-B-r19-logical-hot-identity-six-r2/supplemental-r19-logical-hot-identity-r2'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def paired(reference, candidate):
    assert len(reference) == len(candidate) == 24 and all(len(row) == 479 for row in reference + candidate)
    differences = [a - b for old, new in zip(reference, candidate) for a, b in zip(old, new)]
    assert all(math.isfinite(x) for x in differences)
    pieces = [statistics.mean(a - b for a, b in zip(old, new)) for old, new in zip(reference, candidate)]
    mean = statistics.mean(pieces)
    se = statistics.stdev(pieces) / math.sqrt(len(pieces))
    return dict(mean_nll_increase=mean, paired_piece_standard_error=se, piece_means=pieces,
                criterion='Positive NLL degradation <= one paired-piece SE; no epsilon or multiplier.',
                pass_gate=mean <= se, positions=len(differences),
                differing_positions=sum(struct.pack('>d', float(a)) != struct.pack('>d', float(b))
                    for old, new in zip(reference, candidate) for a, b in zip(old, new)),
                max_absolute_logprob_delta=max(map(abs, differences)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--server-pid', required=True)
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    T = args.trial
    config = read(T / 'B1B_PAIR_CONFIG.json')
    arm = config['arm']
    assert arm in ('hot', 'cold') and os.sched_getaffinity(0) == {55}
    out = T / ('supplemental-' + args.namespace)
    out.mkdir(exist_ok=False)
    result = dict(status='RUNNING', arm=arm, performance_run=False, steps=[],
                  host_reload_gate='UNCHANGED_NOT_RERUN_IN_THIS_STEP', acceptance='HELD')
    save(out / 'RESULT.json', result)
    try:
        for path, wanted in config['references'].items():
            assert sha(Path(path)) == wanted, path
        profile = read(T / 'EFFECTIVE_PROFILE.json')
        env, server = profile['environment'], profile['server_args']
        assert env['KT_CONTRACT_V2_COMBINE_ORDER'] == 'logical' and env['KT_HOT_DYNAMIC'] == '0'
        assert server['enable_eplb'] is False and server['disable_radix_cache'] is True and server['max_running_requests'] == 1
        assert not server['enable_streaming_session'] and not server['enable_hierarchical_cache']
        assert sha(T / 'client-tools/kv_checks.py') == 'eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
        body = dict(text='Explain why the sky is blue in one sentence.', sampling_params=dict(max_new_tokens=8, temperature=0), return_logprob=True, top_logprobs_num=8)
        replies = []
        keys = ('text_equal', 'output_ids_equal', 'selected_logprob_ids_equal', 'selected_logprobs_exact')
        for i in range(2):
            request = urllib.request.Request('http://127.0.0.1:18209/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=180) as response:
                reply = json.load(response)
            save(out / f'replay-{i}.json', reply)
            replies.append(reply)
            if arm == 'cold':
                gate = replay_gate(read(H / f'replay-{i}.json'), reply)
                result['steps'].append(dict(name=f'replay-{i}-placement-invariance', gate=gate))
                save(out / 'RESULT.json', result)
                assert all(gate[k] for k in keys), 'Logical-order replay differs across placements'
        result['repeat'] = replay_gate(*replies)
        assert all(result['repeat'][k] for k in keys), 'Logical-order cold repeat changed'
        prepared = read(T / 'PREPARED.json')
        root = Path(prepared['canonical']['B']['root'])
        reference_receipt = read(N / 'served-nll-receipt.json')
        assert sha(N / 'served-nll.json') == 'ae50236af65eb21146ce8ae4aa19cc03883264401243884c728d4f9041640d6b'
        assert sha(root / 'tools/nll.py') == reference_receipt['upstream_sha256'] == 'c5b29236724b2f30579cde5269e3a08618f698b579b03b985e87000061b3a9d5'
        python = str(root / '.venv/bin/python')
        command = [python, str(args.harness / 'pr_acceptance/run_nll_recorded.py'), '--script', str(root / 'tools/nll.py'),
                   '--checkpoint', server['model_path'], '--output', str(out / 'served-nll.json'), '--receipt', str(out / 'served-nll-receipt.json')]
        child = dict(os.environ, URL='http://127.0.0.1:18209', SGLANG_PORT='18209', PYTHONDONTWRITEBYTECODE='1')
        for key in ('PYTHONPATH', 'FEATURE_ORIGINS_CONFIG', 'FEATURE_ORIGINS_CONFIG_SHA256'):
            child.pop(key, None)
        with (out / 'nll.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=child, timeout=7200)
        result['steps'].append(dict(name='numerical-set', command=command, exit_code=run.returncode))
        save(out / 'RESULT.json', result)
        assert run.returncode == 0
        receipt = read(out / 'served-nll-receipt.json')
        assert receipt['status'] == 'PASS' and receipt['input_ids_sha256'] == reference_receipt['input_ids_sha256']
        assert receipt['pieces'] == 24 and receipt['scored_positions'] == 11496 and receipt['scores_sha256'] == sha(out / 'served-nll.json')
        scores = read(out / 'served-nll.json')
        result['numerical_positions'] = 11496
        if arm == 'hot':
            result.update(status='B1B_LOGICAL_HOT_CAPTURE_COMPLETE',
                          next='Unload; independently run cold-rtr1 with identical source/native and compare exact before the ledger and quality gate.')
        else:
            hot_receipt = read(H / 'served-nll-receipt.json')
            assert hot_receipt['status'] == 'PASS' and hot_receipt['input_ids_sha256'] == receipt['input_ids_sha256'] and hot_receipt['scores_sha256'] == sha(H / 'served-nll.json')
            exact = compare(read(H / 'served-nll.json'), scores)
            save(out / 'EXACT_PLACEMENT_INVARIANCE.json', exact)
            result['placement_invariance'] = exact
            save(out / 'RESULT.json', result)
            assert exact['status'] == 'EXACT' and exact['compared_scores'] == 11496, 'Logical combine did not make the full numerical set placement invariant'
            ledger = paired(read(N / 'served-nll.json'), scores)
            ledger.update(accepted_sha256=sha(N / 'served-nll.json'), candidate_sha256=sha(out / 'served-nll.json'))
            save(out / 'PAIRED_R18_LEDGER.json', ledger)
            result['ledger'] = ledger
            save(out / 'RESULT.json', result)
            assert ledger['pass_gate'], 'Logical combine fails the unchanged one-SE paired-piece rule'
            quality = [python, str(args.harness / 'pr_acceptance/quality_tools/quality_probe.py'), '--checkpoint', server['model_path'],
                       '--arm', 'B', '--base-url', 'http://127.0.0.1:18209', '--output', str(out / 'quality')]
            with (out / 'quality.log').open('w') as log:
                run = subprocess.run(quality, stdout=log, stderr=subprocess.STDOUT, env=child, timeout=7200)
            result['steps'].append(dict(name='quality', command=quality, exit_code=run.returncode))
            save(out / 'RESULT.json', result)
            assert run.returncode == 0 and read(out / 'quality/report.json')['status'] == 'PASS', 'Frozen quality screen failed'
            result.update(status='B1B_LOGICAL_PAIR_AND_LEDGER_PASS_HELD', quality_report_sha256=sha(out / 'quality/report.json'))
        result.update(config_sha256=sha(T / 'B1B_PAIR_CONFIG.json'), effective_profile_sha256=sha(T / 'EFFECTIVE_PROFILE.json'))
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        save(out / 'RESULT.json', result)


if __name__ == '__main__':
    main()
