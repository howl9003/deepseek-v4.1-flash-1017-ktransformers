"""Serving profile timing (Claude, 2 Nov 2026): the frozen benchmark for two rounds on the serving load, then one, two and four sessions at once, a follow-up turn in each of four sessions, and one chat call with a tool.
Rates here are the serving profile's, not the 32.98 performance profile's: radix cache, HiCache with the disk tier, four running requests. stdlib only, /usr/bin/python3."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import threading
import time
import urllib.request

P = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:18209'
CLIENT_SHA = '4635dede1a4d69366ffe6291275e7cb0f02f9a1d44b231afb5cb7612b408d812'
BENCH_SHA = '1c361c2010d7f69a8f1a93e33f26e7622bb684bd43a2de51b29b63a00b063f40'
PANEL = Path('/scratch/aster/bench/reference-panel-r1/panel-data/panel.safetensors')
PROMPTS = ['Explain how a hash map handles collisions, with examples.', 'Write a short story about a lighthouse keeper who finds a message in a bottle.',
           'Derive the quadratic formula step by step and show one worked example.', 'Write a Python function that merges overlapping intervals, with tests.']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def post(path, body):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def chat(prompt, tokens, row):
    """Time one streamed reply: to the first token, then the rate after it."""
    body = dict(model='default', max_tokens=tokens, temperature=1.0, top_p=1.0, stream=True, stream_options=dict(include_usage=True), ignore_eos=True,
                chat_template_kwargs=dict(enable_thinking=False), messages=[dict(role='user', content=prompt)])
    request = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    start, first, last, usage = time.perf_counter(), None, None, None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                continue
            data = json.loads(line[6:])
            usage = data.get('usage') or usage
            if data.get('choices') and (data['choices'][0]['delta'].get('content') or data['choices'][0]['delta'].get('reasoning_content')):
                last = time.perf_counter()
                first = first or last
    row.update(first_s=first - start, seconds=last - first, tokens=usage['completion_tokens'], start=start, end=last, tok_s=(usage['completion_tokens'] - 1) / (last - first))


def together(count, tokens):
    rows = [dict() for _ in range(count)]
    threads = [threading.Thread(target=chat, args=(PROMPTS[i % len(PROMPTS)], tokens, rows[i])) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = max(r['end'] for r in rows) - min(r['start'] for r in rows)
    return dict(sessions=count, tokens=tokens, wall_s=wall, total_tok_s=sum(r['tokens'] for r in rows) / wall, per_session_tok_s=[r['tok_s'] for r in rows], first_s=[r['first_s'] for r in rows])


def tokens():
    data = PANEL.read_bytes()
    n = struct.unpack('<Q', data[:8])[0]
    meta = json.loads(data[8:8 + n])
    a, b = meta['tokens']['data_offsets']
    v = array.array('i')
    v.frombytes(data[8 + n + a: 8 + n + b])
    return v


def turn(ids, row):
    t0 = time.perf_counter()
    reply = post('/generate', dict(input_ids=ids, sampling_params=dict(max_new_tokens=128, temperature=0, ignore_eos=True)))
    meta = reply['meta_info']
    row.update(ids=reply['output_ids'], seconds=time.perf_counter() - t0, cached=meta.get('cached_tokens'), prompt=meta.get('prompt_tokens'), first_s=meta.get('prefill_finished_time', 0) - meta.get('forward_entry_time', 0))


def follow_ups(size, count):
    """Four long sessions, each sending a second turn that extends its first. The second turn should read the first from cache."""
    flat, rows = tokens(), []
    first = [flat[i * size: (i + 1) * size].tolist() for i in range(count)]
    one = [dict() for _ in range(count)]
    threads = [threading.Thread(target=turn, args=(first[i], one[i])) for i in range(count)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    two = [dict() for _ in range(count)]
    threads = [threading.Thread(target=turn, args=(first[i] + one[i]['ids'] + flat[-64:].tolist(), two[i])) for i in range(count)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    for a, b in zip(one, two):
        rows.append(dict(first_turn={k: a[k] for k in ('seconds', 'cached', 'prompt', 'first_s')}, second_turn={k: b[k] for k in ('seconds', 'cached', 'prompt', 'first_s')}))
    return rows


def tool_call():
    body = dict(model='default', max_tokens=256, temperature=0, tools=[dict(type='function', function=dict(name='get_weather', description='Current weather for a city',
                parameters=dict(type='object', properties=dict(city=dict(type='string')), required=['city'])))],
                messages=[dict(role='user', content='What is the weather in Singapore right now? Use the tool.')])
    reply = post('/v1/chat/completions', body)
    message = reply['choices'][0]['message']
    return dict(finish=reply['choices'][0]['finish_reason'], tool_calls=message.get('tool_calls'), content=(message.get('content') or '')[:200], reasoning=(message.get('reasoning_content') or '')[:200], usage=reply.get('usage'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--harness')
    parser.add_argument('--server-pid')
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {55}
    assert args.namespace == 'phase-g-g1v-adasrr-r1'
    assert sha(P / 'tools/client_fast_r1.py') == CLIENT_SHA and sha(P / 'source/bench.py') == BENCH_SHA
    out = args.trial / ('supplemental-' + args.namespace)
    out.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rounds=[], sessions=[], limitation='Serving profile on one load; matched method for the frozen rounds, the session rows are new.')
    write(out / 'RESULT.json', result)
    try:
        for index in range(2):
            namespace = args.namespace + '-round' + str(index)
            target = args.trial / ('supplemental-' + namespace)
            assert not target.exists()
            command = ['/usr/bin/python3', '-B', str(P / 'tools/client_fast_r1.py'), '--trial', str(args.trial), '--namespace', namespace]
            with (out / ('round' + str(index) + '.log')).open('x') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
            assert run.returncode == 0, (index, run.returncode)
            receipt = json.loads((target / 'RESULT.json').read_text())
            historical, breadth = (target / 'historical.log').read_text(), (target / 'breadth.log').read_text()
            rows = [dict(workload=name, tok_s=float(rate)) for name, rate in re.findall(r'decode, (.*?):.*?= ([0-9.]+) tok/s', breadth)]
            prefill = [float(value) for value in re.findall(r'prefill, .*?: ([0-9.]+) s =', historical)]
            result['rounds'].append(dict(index=index, breadth=rows, breadth_median_tok_s=statistics.median(row['tok_s'] for row in rows), japanese_median_tok_s=receipt.get('japanese_median_tokens_per_second'),
                                         prefill35000_seconds=prefill, post_timing_repeat=receipt.get('post_timing_repeat')))
            write(out / 'RESULT.json', result)
        for count in (1, 2, 4):
            result['sessions'].append(together(count, 512))
            write(out / 'RESULT.json', result)
        result['follow_ups'] = follow_ups(16384, 4)
        write(out / 'RESULT.json', result)
        result['tool_call'] = tool_call()
        result.update(status='ADA_SERVE_TIMED_REVIEW_REQUIRED')
        write(out / 'RESULT.json', result)
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        write(out / 'RESULT.json', result)
        raise


if __name__ == '__main__':
    main()
