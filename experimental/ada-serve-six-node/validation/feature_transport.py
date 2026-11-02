"""Bounded transport and provenance gates for a caller-owned, exclusive server."""
import hashlib
import http.client
import json
import math
import os
import socket
from pathlib import Path
import threading
import time
import uuid

MAX_RESPONSE = 4 << 20
REQUIRED_SOURCES = (
    'python/sglang/srt/managers/scheduler.py',
    'python/sglang/srt/session/session_controller.py',
    'python/sglang/srt/session/streaming_session.py',
    'python/sglang/srt/mem_cache/base_prefix_cache.py',
    'python/sglang/srt/mem_cache/unified_radix_cache.py',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_inventory(root, expected):
    root = Path(root).resolve(strict=True)
    if not set(REQUIRED_SOURCES).issubset(expected):
        raise ValueError('source manifest omits session/cache implementation')
    result = {}
    for name, wanted in expected.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or len(wanted) != 64:
            raise ValueError('invalid source manifest path/hash')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise ValueError('source differs from selected profile: ' + name)
        result[name] = actual
    return result


def validate_profile(profile, *, model, context):
    if profile['kind'] not in ('session-only', 'combined-prefix'):
        raise ValueError('unknown profile kind')
    expected = dict(enable_streaming_session=True, max_running_requests=4,
                    prefill_max_requests=4, max_queued_requests=4, schedule_policy='fcfs',
                    enable_priority_scheduling=False, stream_interval=1,
                    tp_size=1, pp_size=1, speculative_algorithm=None, page_size=256)
    args = profile['server_args']
    for key, value in expected.items():
        if key not in args or args[key] != value or type(args[key]) is not type(value):
            raise ValueError('profile must specify ' + key + '=' + repr(value))
    if args['context_length'] != context or args['served_model_name'] != model:
        raise ValueError('CLI model/context differs from profile')
    if type(context) is not int or context < 4096:
        raise ValueError('context must hold the bounded session/queue fixtures')
    if type(args.get('max_total_tokens')) is not int or args['max_total_tokens'] < 4*4096:
        raise ValueError('explicit device token pool must fit four bounded fixtures')
    if type(args.get('random_seed')) is not int or type(args.get('chunked_prefill_size')) is not int:
        raise ValueError('explicit server seed and chunked-prefill size are required')
    env = profile['environment']
    if not env or any(not isinstance(k, str) or (v is not None and not isinstance(v, str)) for k,v in env.items()):
        raise ValueError('environment must map names to exact strings or null (absent)')
    if env.get('KT_STREAMING_SESSION_ADMISSION') != '1':
        raise ValueError('session admission must be explicitly enabled')
    if profile['kind'] == 'session-only':
        if args.get('disable_radix_cache') is not True or args.get('enable_hierarchical_cache') is not False:
            raise ValueError('session-only requires radix and HiCache off')
        if env.get('SGLANG_CACHE_TTL') != '0':
            raise ValueError('session-only must explicitly disable TTL')
    else:
        if args.get('disable_radix_cache') is not False or args.get('enable_hierarchical_cache') is not True:
            raise ValueError('combined-prefix requires radix and HiCache on')
    if profile.get('exclusive_caller_owned_server') is not True:
        raise ValueError('profile must declare exclusive caller-owned server')
    if not isinstance(args.get('model_path'),str) or not args['model_path'].startswith('/'):
        raise ValueError('explicit absolute model/checkpoint path required')
    return profile


def process_provenance(pid, root, port, expected_env):
    proc = Path('/proc') / str(pid)
    args = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if 'sglang.launch_server' not in args:
        raise ValueError('server PID is not the caller-owned launch_server process')
    if '--port' not in args or args[args.index('--port') + 1] != str(port):
        raise ValueError('server PID does not bind requested port')
    raw = dict(x.split('=', 1) for x in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in x)
    selected = {key: raw.get(key) for key in expected_env}
    if selected != expected_env:
        raise ValueError('server process environment differs from profile')
    python_root = str((Path(root) / 'python').resolve())
    if python_root not in [str(Path(x).resolve()) for x in raw.get('PYTHONPATH', '').split(':') if x]:
        raise ValueError('selected source root is absent from server PYTHONPATH')
    # PID reuse is detected by the process start-time field; do not dump unselected env.
    stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    observed = {k:v for k,v in raw.items() if k.startswith(('KT_','SGLANG_','DSV41_','SM89_'))}
    return dict(pid=pid, start_ticks=stat[19], cmdline=args, environment=selected,
                observed_selected_environment=observed,
                source_python_root=python_root)


class Stream:
    def __init__(self, ctx, label, body):
        self.ctx, self.label, self.body = ctx, label, dict(body, stream=True)
        self.lock = threading.RLock()
        self.state = dict(first_token=False, finished=False, status=None, errors=[], chunks=[])
        self.connection = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return dict(self.state, errors=list(self.state['errors']), chunks=list(self.state['chunks']))

    def close(self):
        if self.connection:
            self.ctx.release_connection(self.connection)

    def _run(self):
        start = time.monotonic()
        try:
            self.connection = self.ctx.connection()
            self.connection.request('POST', '/generate', json.dumps(self.body), {'Content-Type':'application/json'})
            response = self.connection.getresponse()
            with self.lock:
                self.state['status'] = response.status
            total = 0
            while True:
                self.ctx.check()
                line = response.readline(MAX_RESPONSE + 1)
                if not line:
                    break
                total += len(line)
                if total > MAX_RESPONSE:
                    raise ValueError('stream exceeds 4MiB')
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    break
                item = json.loads(data)
                with self.lock:
                    self.state['chunks'].append(dict(elapsed_s=time.monotonic()-start, value=item))
                    if item.get('meta_info', {}).get('completion_tokens', 0) > 0:
                        self.state['first_token'] = True
        except Exception as exc:
            with self.lock:
                self.state['errors'].append(repr(exc))
        finally:
            self.close()
            with self.lock:
                self.state['finished'] = True
            self.ctx.record_event(self.label, dict(request=self.body, stream=self.snapshot()))


class Context:
    def __init__(self, *, host, port, model, output, deadline_s):
        if host != '127.0.0.1':
            raise ValueError('local process/source provenance requires IPv4 loopback')
        if not 60 <= deadline_s <= 3600:
            raise ValueError('deadline must be 60..3600 seconds')
        self.host, self.port, self.model, self.output = host, port, model, Path(output)
        self.started = time.monotonic()
        self.deadline = self.started + deadline_s
        self.lock = threading.RLock()
        self.owned = set()
        self.known = set()
        self.streams = []
        self.connections = {}
        self.cleanup_mode = False

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def check(self):
        if self.remaining() <= 0:
            raise TimeoutError('absolute client deadline')

    def wait(self, seconds):
        if not math.isfinite(seconds) or seconds < 0 or seconds >= self.remaining():
            raise TimeoutError('wait exceeds absolute client deadline')
        time.sleep(seconds)
        self.check()

    def connection(self, maximum_seconds=None):
        self.check()
        budget = min(self.remaining(), maximum_seconds or self.remaining())
        absolute_end = time.monotonic() + budget
        c = http.client.HTTPConnection(self.host, self.port, timeout=min(90, budget))
        try:
            c.connect()
            self.check()
            connected_socket = c.sock
            budget = min(self.remaining(), absolute_end-time.monotonic())
            if budget <= 0:
                raise TimeoutError('connection deadline')
            timer = threading.Timer(budget, self._interrupt, args=(c, connected_socket))
            timer.daemon = True
            with self.lock:
                self.connections[c] = (timer, connected_socket)
            timer.start()
            return c
        except BaseException:
            c.close()
            raise

    @staticmethod
    def _interrupt(c, connected_socket):
        # Closing an HTTPConnection alone may leave its response.makefile reader
        # alive. shutdown interrupts a blocked/dribbling read at the absolute deadline.
        try:
            if connected_socket:
                connected_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        c.close()

    def release_connection(self, c):
        with self.lock:
            tracked = self.connections.pop(c, None)
        if tracked:
            timer, connected_socket = tracked
            timer.cancel()
            self._interrupt(c, connected_socket)
        else:
            c.close()

    def close_connections(self):
        with self.lock:
            connections = list(self.connections)
        for c in connections:
            self.release_connection(c)

    def record_event(self, label, data):
        item = dict(label=label, elapsed_s=time.monotonic()-self.started, data=data)
        with self.lock, (self.output/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, allow_nan=False) + '\n')

    def request(self, label, route, body=None, allow_reject=False, headers=None):
        if self.cleanup_mode and route not in ('/abort_request', '/close_session', '/v1/loads?include=all'):
            raise RuntimeError('cleanup cannot submit more work')
        if body and route in ('/generate', '/v1/completions'):
            if route == '/v1/completions':
                body = dict(body, rid=body.get('rid') or 'kv-'+uuid.uuid4().hex)
            if not body.get('rid'):
                raise ValueError('generation must use an owned explicit request ID')
            self.owned.add(body['rid'])
            self.known.add(body['rid'])
        short = label == 'overflow' or route == '/abort_request'
        absolute_end = time.monotonic()+min(self.remaining(),5 if short else self.remaining())
        c = self.connection(maximum_seconds=5 if short else None)
        start = time.monotonic()
        try:
            c.request('GET' if body is None else 'POST', route,
                      None if body is None else json.dumps(body),
                      dict({'Content-Type':'application/json'}, **(headers or {})))
            response = c.getresponse()
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise ValueError('response exceeds 4MiB')
            try:
                value = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                value = raw.decode(errors='replace')
            result = dict(status=response.status, value=value, wall_s=time.monotonic()-start)
            self.record_event(label, dict(route=route, request=body, headers=headers, response=result))
            if time.monotonic() >= absolute_end:
                raise TimeoutError('absolute request deadline')
            if body and route in ('/generate', '/v1/completions') and terminal_response(route,result):
                self.owned.discard(body['rid'])
            if response.status != 200 and not allow_reject:
                raise RuntimeError(f'{route}: HTTP {response.status}: {str(value)[:512]}')
            self.check()
            return result
        finally:
            self.release_connection(c)

    def start_stream(self, label, body):
        if self.cleanup_mode or not body.get('rid'):
            raise ValueError('stream requires explicit owned request ID before cleanup')
        self.owned.add(body['rid'])
        self.known.add(body['rid'])
        stream = Stream(self, label, body)
        self.streams.append(stream)
        return stream

    def abort(self, rid):
        if rid not in self.known:
            raise ValueError('refusing to abort an unowned request')
        if rid not in self.owned:
            return {'skipped':'already completed nonstream response', 'rid':rid}
        result = self.request('abort-owned', '/abort_request', {'rid':rid}, allow_reject=True)
        if result['status']==200:
            self.owned.discard(rid)
        return result

    def loads(self):
        return self.request('loads', '/v1/loads?include=all')['value']

    def idle(self, seconds=20):
        until = min(self.deadline, time.monotonic()+seconds)
        mark = time.time()
        while time.monotonic() < until:
            rows = self.loads().get('loads', [])
            if len(rows) != 1:
                raise ValueError('expected one TP-rank load snapshot')
            row = rows[0]
            if row['timestamp'] >= mark and row['num_running_reqs'] == row['num_waiting_reqs'] == 0:
                return row
            self.wait(.25)
        raise TimeoutError('server did not reach fresh idle state')

    def flush(self):
        self.idle()
        return self.request('flush-owned-idle-cache', '/flush_cache', {})

    def begin_cleanup(self, seconds=30):
        self.cleanup_mode = True
        self.deadline = time.monotonic() + seconds

    def close_streams(self):
        for stream in self.streams:
            stream.close()
        for stream in self.streams:
            stream.thread.join(timeout=min(1, self.remaining()))
        return [s.label for s in self.streams if s.thread.is_alive()]


def terminal_response(route, result):
    """Keep ambiguous/invalid replies owned for abort; never infer from HTTP alone."""
    value = result['value']
    if result['status']==503 and 'request queue is full' in json.dumps(value).lower():
        return True
    if result['status']!=200 or not isinstance(value,dict):
        return False
    if route=='/generate':
        return (value.get('meta_info',{}).get('finish_reason') or {}).get('type') in ('length','stop','abort')
    choices=value.get('choices')
    return (isinstance(choices,list) and bool(choices)
            and all(isinstance(x,dict) and x.get('finish_reason') in ('length','stop','tool_calls','content_filter') for x in choices))
