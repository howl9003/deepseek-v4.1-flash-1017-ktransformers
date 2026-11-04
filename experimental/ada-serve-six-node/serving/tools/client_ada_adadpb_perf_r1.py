"""Plain-gate batch check (Claude, 4 Nov 2026; owner: "try to fit dspark into the idle times ... if the cost can be reduced to nearly
free its a free win"). DSpark with the plain-step gate r2 (KT_DSPARK_PLAIN_GATE_MAX_BS=4: a batch runs plain when any request
votes plain) on the performance profile with four running requests. Two parts:

1. Tokens. Four greedy prompts (English prose, Japanese prose, code, JSON), 320 tokens each, no logprobs (a logprob request
   always speculates, so the identity client never compared a plain step's tokens). Each alone, then staggered 4 s apart,
   then all at once, the cache flushed before each. Saved here and compared offline with the same capture on served C2
   (run_pgtok_c2_r1.py): no C2 archive holds no-logprob greedy text.
2. Sessions. One, two and four chat sessions at once, two rounds (adasfp's method: 512 tokens, T=1, its prompts).

The gate's step counts come from the server log. stdlib only, /usr/bin/python3; importable (run_pgtok_c2_r1.py uses capture
and together)."""
import argparse
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:18209'
TOKEN_PROMPTS = [
    ('essay', 'Write a long, detailed essay in English on the history of bridges, with examples from several centuries.\n\n'),
    ('japanese', '日本の四季と、それぞれの季節の行事や食べ物について、詳しく説明してください。\n\n'),
    ('code', 'def merge_intervals(intervals):\n    """Merge overlapping intervals and return the merged list, sorted by start."""\n'),
    ('json', '{"employees": [\n  {"id": 1, "name": "Alice Smith", "department": "Engineering", "email": "alice.smith@example.com"},\n'),
]
TOKENS = 320
MODES = ('single', 'staggered', 'together')
SESSION_PROMPTS = ['Explain how a hash map handles collisions, with examples.', 'Write a short story about a lighthouse keeper who finds a message in a bottle.',
                   'Derive the quadratic formula step by step and show one worked example.', 'Write a Python function that merges overlapping intervals, with tests.']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def post(url, body, timeout=1800):
    request = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def flush(url=URL, tries=60):
    """SGLang answers 400 while requests are running or waiting (r5 hit that on served C2, 4 Nov): retry for a minute."""
    for attempt in range(tries):
        request = urllib.request.Request(url + '/flush_cache', data=b'', method='POST')
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                assert response.status == 200, 'flush failed'
                return
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or attempt == tries - 1:
                raise
        time.sleep(1)


def generate(url, text, row):
    body = dict(text=text, sampling_params=dict(max_new_tokens=TOKENS, temperature=0, ignore_eos=True))
    start = time.perf_counter()
    reply = post(url + '/generate', body)
    end = time.perf_counter()
    meta = reply['meta_info']
    row.update(output_ids=reply['output_ids'], text=reply['text'], start=start, end=end, seconds=end - start,
               prompt_tokens=meta.get('prompt_tokens'), completion_tokens=meta.get('completion_tokens'),
               spec={k: v for k, v in meta.items() if k.startswith('spec')})


def run_mode(url, mode):
    rows = [dict(name=name) for name, _ in TOKEN_PROMPTS]
    if mode == 'single':
        for (_, text), row in zip(TOKEN_PROMPTS, rows):
            flush(url)
            generate(url, text, row)
        return rows
    flush(url)
    threads = []
    for (_, text), row in zip(TOKEN_PROMPTS, rows):
        thread = threading.Thread(target=generate, args=(url, text, row))
        threads.append(thread)
        thread.start()
        if mode == 'staggered':
            time.sleep(4)
    for thread in threads:
        thread.join()
    assert all('output_ids' in row for row in rows), mode + ': a request failed'
    return rows


def capture(url=URL):
    return {mode: run_mode(url, mode) for mode in MODES}


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(reference, candidate):
    """Every candidate row against the reference's single row and the reference's row in the same mode."""
    out = {}
    for mode in MODES:
        for ref_single, ref_same, row in zip(reference['single'], reference[mode], candidate[mode]):
            out[f"{mode}/{row['name']}"] = dict(vs_single=first_difference(ref_single['output_ids'], row['output_ids']),
                                                vs_same_mode=first_difference(ref_same['output_ids'], row['output_ids']))
    out['exact_vs_single'] = all(v['vs_single'] is None for k, v in out.items() if isinstance(v, dict))
    return out


def chat(prompt, tokens, row, url=URL):
    """Time one streamed reply: to the first token, then the rate after it (adasfp's method)."""
    body = dict(model='default', max_tokens=tokens, temperature=1.0, top_p=1.0, stream=True, stream_options=dict(include_usage=True), ignore_eos=True,
                chat_template_kwargs=dict(enable_thinking=False), messages=[dict(role='user', content=prompt)])
    request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
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


def together(count, tokens, url=URL):
    rows = [dict() for _ in range(count)]
    threads = [threading.Thread(target=chat, args=(SESSION_PROMPTS[i % len(SESSION_PROMPTS)], tokens, rows[i], url)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = max(r['end'] for r in rows) - min(r['start'] for r in rows)
    return dict(sessions=count, tokens=tokens, wall_s=wall, total_tok_s=sum(r['tokens'] for r in rows) / wall, per_session_tok_s=[r['tok_s'] for r in rows], first_s=[r['first_s'] for r in rows])


def gate_lines(server_log, start):
    with server_log.open('rb') as stream:
        stream.seek(start)
        text = stream.read().decode(errors='replace')
    return re.findall(r'DSpark plain gate: .*', text)


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
    launch = json.loads((T / 'LAUNCH.json').read_text())
    env, argv = launch['environment'], launch['argv']
    assert env['KT_DSPARK_PLAIN_GATE'] == '1' and env['KT_DSPARK_PLAIN_GATE_MAX_BS'] == '4' and not env.get('KT_DSPARK_PLAIN_GATE_FORCE')
    assert argv[argv.index('--max-running-requests') + 1] == '4' and '--enable-fp32-lm-head' in argv
    server_log = T / ('model-' + args.namespace) / 'server/launcher.log'
    result = dict(status='RUNNING', performance_run=True, limitation='Performance profile with four running requests, not the serving profile; tokens compared offline against served C2.')
    write(out / 'RESULT.json', result)
    try:
        start = server_log.stat().st_size
        tokens = capture()
        write(out / 'tokens.json', tokens)
        result['tokens'] = {mode: [dict(name=r['name'], seconds=round(r['seconds'], 2), completion_tokens=r['completion_tokens'], spec=r['spec']) for r in rows]
                            for mode, rows in tokens.items()}
        result['tokens_gate_log'] = gate_lines(server_log, start)[-4:]
        write(out / 'RESULT.json', result)
        result['sessions'] = []
        for _ in range(2):
            for count in (1, 2, 4):
                result['sessions'].append(together(count, 512))
                write(out / 'RESULT.json', result)
        result['gate_log'] = gate_lines(server_log, start)[-6:]
        result['status'] = 'PERFORMANCE_COMPLETE_NOT_PR6_ACCEPTANCE'
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        write(out / 'RESULT.json', result)


if __name__ == '__main__':
    main()
