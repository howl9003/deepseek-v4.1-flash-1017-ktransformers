"""C2 repeat (Claude, 3 Nov 2026): client_ada_identity_r2.py on a second load of the C2 build, against the adac2a capture instead of C1:
replays, the 11,496 scores and the three long cases with their top 8 must equal it bit for bit.

Ada identity client with a rotation check (Claude, 1 Nov 2026): client_ada_identity_r1.py, first a streamed prefill on the fresh server against the same prefill after a decode
has swapped the resident rows and /flush_cache has cleared the pool. Their bits must agree: the check for the defect 05yuki fixed in 23dd4ea, our KT_PREFILL_LEND_EXPERT_SCALES.

Ada baseline identity client (Claude, 31 Oct 2026): the G1p client with the accepted C1 archive (through arm 0 r3, which equals it on all
11,496 scores) as reference. Prefill path A was proven against its off arm on the single route; the expert path does not read the dense
route, so the long cases here check flags and finite logprobs and record the bits, and an off arm compares them if this build is kept.

Phase G G1p identity client (Claude, 30 Oct 2026): prefill path A experts on the single route, on against off.

Both arms run on the arm-2 profile, tree r5. The numerical set and the two replays must equal arm 2's archive bit for bit, so
the off arm also proves trees r2 to r5 changed no served bits. Then 4K and two 35K prompts with eight scored continuation
tokens (client_r19_D2_identity_r1.py's cases): the off arm records them, the on arm must match them exactly, with the kernel's
flag word clear on all 40 layers of each request. Timing belongs to a separate load.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import urllib.request
from compare_r19_exact_scores_r1 import compare
from client_r19_bitneutral_identity_r1 import replay_gate

B = Path('/scratch/aster/bench')
ARM2 = B / 'ada-identity-c2-20261103-r1/supplemental-phase-g-g1v-adac2a-r1'  # C2, from adac2a
CASES = [('four-k', 4096, 100), ('35k-first', 35000, 0), ('35k-second', 35000, 1)]
KEYS = ('text_equal', 'output_ids_equal', 'selected_logprob_ids_equal', 'selected_logprobs_exact')
ROTATION_TEXT = 'Write a long, detailed essay in English on the history of bridges, with examples from several centuries.'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def post(body, timeout):
    request = urllib.request.Request('http://127.0.0.1:18209/generate', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def body_for(length, seed):
    rng = random.Random(seed)
    return dict(input_ids=[rng.randrange(1000, 129280) for _ in range(length)],
                sampling_params=dict(max_new_tokens=8, temperature=0, ignore_eos=True),
                return_logprob=True, logprob_start_len=-1, top_logprobs_num=8)


def long_identity(reference, candidate):
    gate = replay_gate(reference, candidate)

    def top(reply):
        rows = reply['meta_info']['output_top_logprobs']
        assert len(rows) == 8 and all(len(r) == 8 for r in rows)
        return [[r[0] for r in block] for block in rows], [[r[1] for r in block] for block in rows]
    a, ai = top(reference)
    b, bi = top(candidate)
    gate['top_logprobs_exact'] = compare(a, b)['status'] == 'EXACT'
    gate['top_logprob_ids_exact'] = ai == bi
    gate['continuation_length'] = len(candidate['output_ids']) == 8
    gate['finite_selected_logprobs'] = all(math.isfinite(r[0]) for r in candidate['meta_info']['output_token_logprobs'])
    return gate


def flush():
    request = urllib.request.Request('http://127.0.0.1:18209/flush_cache', data=b'', method='POST')
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.status


def rotation(out, server_log):
    """The fresh server's rows, then the rows a decode left: one prompt, one result."""
    body = body_for(2048, 7)
    first = post(body, 600)
    save(out / 'rotation-first.json', first)
    assert flush() == 200, 'first flush'
    start = server_log.stat().st_size
    turn = post(dict(text=ROTATION_TEXT, sampling_params=dict(max_new_tokens=640, temperature=0, ignore_eos=True)), 600)
    assert len(turn['output_ids']) == 640
    with server_log.open('rb') as stream:
        stream.seek(start)
        text = stream.read().decode(errors='replace')
    swaps = [int(x) for x in re.findall(r'dynamic round \d+: .*?(\d+) swaps', text)]
    assert flush() == 200, 'second flush'
    second = post(body, 600)
    save(out / 'rotation-second.json', second)
    gate = long_identity(first, second)
    gate.update(swaps_logged=swaps, moved=bool(swaps) and max(swaps) > 0)
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--server-pid', required=True)
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    T = args.trial
    assert os.sched_getaffinity(0) == {55}
    out = T / ('supplemental-' + args.namespace)
    out.mkdir(exist_ok=False)
    env = read(T / 'LAUNCH.json')['environment']
    on = env['KT_GPU_STREAM_PATHA'] == '1'
    assert env['KT_GPU_STREAM_PATHA'] in ('0', '1') and env['KT_GPU_STREAM_PATHA_FLAGS_LOG'] == '1'
    assert on and env.get('SGLANG_FP8_PATHA_DENSE', '0') == '0' and env.get('SGLANG_SINGLE_ROUTE', '0') == '0' and env['SGLANG_FP8_ROUTE3_CUDA'] == '1'
    assert sha(T / 'client-tools/kv_checks.py') == 'eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
    assert env['SGLANG_SM89_V41_KV'] == '1' and env['SGLANG_SM89_FP4_COMPRESSED_KV'] == '1', 'not the C2 build'
    assert '--enable-fp32-lm-head' in read(T / 'LAUNCH.json')['argv'], 'not the C2 build: BF16 head'
    result = dict(status='RUNNING', arm='on' if on else 'off', performance_run=False, cases=[], acceptance='HELD')
    save(out / 'RESULT.json', result)
    try:
        server_log = T / ('model-' + args.namespace) / 'server/launcher.log'
        result['rotation'] = rotation(out, server_log)
        save(out / 'RESULT.json', result)
        for i in range(2):
            reply = post(dict(text='Explain why the sky is blue in one sentence.', sampling_params=dict(max_new_tokens=8, temperature=0),
                              return_logprob=True, top_logprobs_num=8), 180)
            save(out / f'replay-{i}.json', reply)
            gate = replay_gate(read(ARM2 / f'replay-{i}.json'), reply)
            result[f'replay_{i}'] = gate
            save(out / 'RESULT.json', result)
            assert all(gate[k] for k in KEYS), f'replay {i} differs from C2'
        prepared = read(T / 'PREPARED.json')
        root = Path(prepared['canonical']['B']['root'])
        profile = read(T / 'EFFECTIVE_PROFILE.json')
        reference_receipt = read(ARM2 / 'served-nll-receipt.json')
        python = str(root / '.venv/bin/python')
        command = [python, str(args.harness / 'pr_acceptance/run_nll_recorded.py'), '--script', str(root / 'tools/nll.py'),
                   '--checkpoint', profile['server_args']['model_path'], '--output', str(out / 'served-nll.json'), '--receipt', str(out / 'served-nll-receipt.json')]
        child = dict(os.environ, URL='http://127.0.0.1:18209', SGLANG_PORT='18209', PYTHONDONTWRITEBYTECODE='1')
        for key in ('PYTHONPATH', 'FEATURE_ORIGINS_CONFIG', 'FEATURE_ORIGINS_CONFIG_SHA256'):
            child.pop(key, None)
        with (out / 'nll.log').open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=child, timeout=7200)
        assert run.returncode == 0, 'NLL recording failed'
        receipt = read(out / 'served-nll-receipt.json')
        assert receipt['status'] == 'PASS' and receipt['input_ids_sha256'] == reference_receipt['input_ids_sha256'] and receipt['scored_positions'] == 11496
        exact = compare(read(ARM2 / 'served-nll.json'), read(out / 'served-nll.json'))
        result['numerical_set_vs_c2'] = exact
        save(out / 'RESULT.json', result)
        assert exact['status'] == 'EXACT', 'numerical set differs from C2'
        for label, length, seed in CASES:
            body = body_for(length, seed)
            pin = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            start = server_log.stat().st_size
            answer = post(body, 1800)
            save(out / (label + '.json'), answer)
            assert answer['meta_info']['prompt_tokens'] == length
            gate = long_identity(read(ARM2 / (label + '.json')), answer)
            with server_log.open('rb') as stream:
                stream.seek(start)
                text = stream.read().decode(errors='replace')
            flags = re.findall(r'\[patha-prefill\] layer-check=(\d+) flags=(\d+)', text)
            gate['layer_flags'] = (len(flags) >= 40 and all(value == '0' for _, value in flags)) if on else not flags
            result['cases'].append(dict(case=label, prompt_tokens=length, request_sha256=pin, response_sha256=sha(out / (label + '.json')),
                                        gate=gate, flag_checks=len(flags), compared_against_c2=True))
            save(out / 'RESULT.json', result)
            assert all(v is True for k, v in gate.items() if k != 'numeric'), label + ' long-context identity failed'
        assert all(v is True for k, v in result['rotation'].items() if k not in ('numeric', 'swaps_logged')), 'rotation changed the result'
        result['status'] = 'ADA_NUMERICAL_EXACT_VS_C2'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        save(out / 'RESULT.json', result)


if __name__ == '__main__':
    main()
