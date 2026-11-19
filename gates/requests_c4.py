import json
import os
import threading
import time
import urllib.error
import urllib.request

URL = os.environ.get('C4_URL', 'http://127.0.0.1:18209')

PROMPTS = ['Explain how a hash map handles collisions, with examples.', 'Write a short story about a lighthouse keeper who finds a message in a bottle.',
           'Derive the quadratic formula step by step and show one worked example.', 'Write a Python function that merges overlapping intervals, with tests.']


ORIGINAL_JSON_PROMPT_SHA256 = '1b04d1492124271d854181cbf035a2f7a25678fef2ce90dca8061849bd3136db'


def post(path, body):
    request = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def chat(prompt, tokens, row):
    """Time a streamed reply through its last content token."""
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
    if usage is None or usage.get("completion_tokens") != tokens or first is None or last is None:
        raise ValueError("Incomplete timed response")
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


TOKEN_PROMPTS = [
    ('essay', 'Write a long, detailed essay in English on the history of bridges, with examples from several centuries.\n\n'),
    ('japanese', '日本の四季と、それぞれの季節の行事や食べ物について、詳しく説明してください。\n\n'),
    ('code', 'def merge_intervals(intervals):\n    """Merge overlapping intervals and return the merged list, sorted by start."""\n'),
    ('json', '{"employees": [\n  {"id": 1, "name": "Alice Smith", "department": "Engineering", "email": "[contact removed]"},\n'),
]


def flush(tries=60):
    """Flush."""
    for attempt in range(tries):
        request = urllib.request.Request(URL + '/flush_cache', data=b'', method='POST')
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                assert response.status == 200, 'flush failed'
                return
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or attempt == tries - 1:
                raise
        time.sleep(1)


def generate(text, row):
    """Generate."""
    start = time.perf_counter()
    reply = post('/generate', dict(text=text, sampling_params=dict(max_new_tokens=320, temperature=0, ignore_eos=True)))
    meta = reply['meta_info']
    row.update(output_ids=reply['output_ids'], text=reply['text'], start=start, end=time.perf_counter(), seconds=time.perf_counter() - start,
               prompt_tokens=meta.get('prompt_tokens'), completion_tokens=meta.get('completion_tokens'), spec={k: v for k, v in meta.items() if k.startswith('spec')})


def run_mode(mode):
    rows = [dict(name=name) for name, _ in TOKEN_PROMPTS]
    if mode == 'single':
        for (_, text), row in zip(TOKEN_PROMPTS, rows):
            flush()
            generate(text, row)
        return rows
    flush()
    threads = []
    for (_, text), row in zip(TOKEN_PROMPTS, rows):
        threads.append(threading.Thread(target=generate, args=(text, row)))
        threads[-1].start()
        if mode == 'staggered':
            time.sleep(4)
    for thread in threads:
        thread.join()
    assert all('output_ids' in row for row in rows), mode + ': a request failed'
    return rows
