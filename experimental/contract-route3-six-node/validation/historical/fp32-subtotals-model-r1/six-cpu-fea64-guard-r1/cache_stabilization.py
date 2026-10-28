#!/usr/bin/env python3
"""Bounded, owned-cgroup file-cache stabilization before measured requests.

The final-server scope_exec forks a resident helper before executing the
server. It is born inside the systemd scope; cross-scope cgroup.procs migration
is not permitted on the target host. No reclaim occurs after the ready
memory-events baseline is taken.
"""

import argparse
import errno
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

GIB = 1024 ** 3
MIB = 1024 ** 2
HARD_CAP_BYTES = 416 * GIB
TARGET_BYTES = 392 * GIB
MAX_REQUEST_BYTES = 8 * GIB
MAX_ATTEMPTS = 6
HELPER_TIMEOUT_SECONDS = 30
TOTAL_TIMEOUT_SECONDS = 240
SETTLE_SECONDS = 20
TELEMETRY_TOLERANCE_BYTES = 256 * MIB
MIN_PROGRESS_BYTES = 256 * MIB


def _save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def _stat(sample):
    values = {}
    for line in sample['memory.stat'].splitlines():
        key, value = line.split()
        values[key] = int(value)
    required = ('anon', 'file', 'shmem', 'inactive_file', 'file_dirty', 'file_writeback')
    missing = set(required) - values.keys()
    if missing:
        raise RuntimeError('memory.stat missing required counters: ' + ','.join(sorted(missing)))
    return values


def _eligibility(sample):
    stat = _stat(sample)
    ordinary_unwritten = max(0, stat['file'] - stat['shmem']
                             - stat['file_dirty'] - stat['file_writeback'])
    return min(stat['inactive_file'], ordinary_unwritten)


def _v2_self_cgroup():
    rows = [line.partition('::')[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
            if line.startswith('0::')]
    if len(rows) != 1:
        raise RuntimeError('expected one cgroup-v2 self path')
    return (Path('/sys/fs/cgroup') / rows[0].lstrip('/')).resolve()


def _proc_identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        raw = (proc / 'stat').read_text()
        fields = raw.rsplit(') ', 1)[1].split()
        state, starttime = fields[0], int(fields[19])
        group = (proc / 'cgroup').read_text()
    except FileNotFoundError:
        return None
    rows = [line.partition('::')[2] for line in group.splitlines() if line.startswith('0::')]
    if len(rows) != 1:
        raise RuntimeError('resident helper has no unique cgroup-v2 path')
    cgroup = (Path('/sys/fs/cgroup') / rows[0].lstrip('/')).resolve()
    return {'pid': pid, 'state': state, 'starttime': starttime, 'cgroup': str(cgroup)}


def _helper_receipt(cgroup, out, *, require_alive=True):
    helper = json.loads((out / 'RECLAIM_HELPER.json').read_text())
    scope = json.loads((out / 'SCOPE.json').read_text())
    if (helper.get('status') != 'READY' or helper.get('nonce') != scope.get('reclaim_nonce')
            or helper.get('cgroup') != str(cgroup) or scope.get('cgroup') != str(cgroup)):
        raise RuntimeError('resident helper is not bound to exact server scope')
    pid, starttime = helper.get('pid'), helper.get('starttime')
    if type(pid) is not int or type(starttime) is not int:
        raise RuntimeError('resident helper PID/starttime malformed')
    live = _proc_identity(pid)
    if require_alive and (live is None or live['state'] == 'Z'):
        raise RuntimeError('resident helper died before reclaim')
    if live is not None and (live['starttime'] != starttime or
                            (live['state'] != 'Z' and live['cgroup'] != str(cgroup))):
        raise RuntimeError('resident helper PID reused or changed cgroup')
    return helper


def resident_helper(cgroup, out, nonce):
    """Born inside scope_exec's verified final-server cgroup, never migrated."""
    cgroup, out = cgroup.resolve(strict=True), out.resolve(strict=True)
    if _v2_self_cgroup() != cgroup:
        raise RuntimeError('resident helper did not inherit exact server scope')
    scope = json.loads((out / 'SCOPE.json').read_text())
    if scope.get('cgroup') != str(cgroup) or scope.get('reclaim_nonce') != nonce:
        raise RuntimeError('resident helper nonce/scope mismatch')
    main_pid = scope.get('pid')
    if type(main_pid) is not int or main_pid <= 0:
        raise RuntimeError('resident helper has no verified main launcher PID')
    main_identity = _proc_identity(main_pid)
    if main_identity is None or main_identity['cgroup'] != str(cgroup):
        raise RuntimeError('resident helper main launcher not in scope')
    main_starttime = main_identity['starttime']
    identity = _proc_identity(os.getpid())
    if identity is None or identity['cgroup'] != str(cgroup):
        raise RuntimeError('resident helper identity unavailable')
    _save(out / 'RECLAIM_HELPER.json', {'status': 'READY', 'pid': os.getpid(),
          'starttime': identity['starttime'], 'cgroup': str(cgroup), 'nonce': nonce})
    deadline = time.monotonic() + 7400
    for number in range(1, MAX_ATTEMPTS + 1):
        request_path = out / f'reclaim-request-{number}.json'
        while not request_path.exists():
            if time.monotonic() > deadline:
                raise RuntimeError('resident helper lifetime exceeded 7400s')
            main_live = _proc_identity(main_pid)
            if (main_live is None or main_live['state'] == 'Z'
                    or main_live['starttime'] != main_starttime):
                raise RuntimeError('final-server launcher ended before reclaim')
            time.sleep(0.2)
        request = json.loads(request_path.read_text())
        amount = request.get('requested_bytes')
        if (request.get('nonce') != nonce or request.get('number') != number
                or type(amount) is not int or not 0 < amount <= MAX_REQUEST_BYTES
                or _v2_self_cgroup() != cgroup):
            raise RuntimeError('resident helper request validation failed')
        try:
            (cgroup / 'memory.reclaim').write_text(f'{amount} swappiness=0\n')
            status, error = 'WRITE_COMPLETE', None
        except OSError as exc:
            status, error = ('EAGAIN' if exc.errno == errno.EAGAIN else 'FAILED'), exc.errno
        response = {'status': status, 'errno': error, 'pid': os.getpid(),
                    'starttime': identity['starttime'], 'cgroup': str(cgroup),
                    'nonce': nonce, 'number': number, 'requested_bytes': amount}
        _save(out / f'reclaim-response-{number}.json', response)
        if status == 'FAILED':
            raise RuntimeError(f'memory.reclaim failed errno={error}')
    # After the sixth request, wait for the parent to terminate the helper.
    while time.monotonic() <= deadline:
        time.sleep(0.2)
    raise RuntimeError('resident helper lifetime exceeded 7400s')


def _run_helper(cgroup, out, requested_bytes, number, *, timeout_seconds=HELPER_TIMEOUT_SECONDS):
    helper = _helper_receipt(cgroup, out)
    request = {'nonce': helper['nonce'], 'number': number,
               'requested_bytes': requested_bytes}
    response_path = out / f'reclaim-response-{number}.json'
    if response_path.exists():
        raise RuntimeError('stale reclaim response already exists')
    _save(out / f'reclaim-request-{number}.json', request)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if response_path.exists():
            result = json.loads(response_path.read_text())
            if (result.get('status') not in ('WRITE_COMPLETE', 'EAGAIN')
                    or result.get('nonce') != helper['nonce']
                    or result.get('pid') != helper['pid']
                    or result.get('starttime') != helper['starttime']
                    or result.get('cgroup') != str(cgroup)
                    or result.get('number') != number
                    or result.get('requested_bytes') != requested_bytes):
                raise RuntimeError('resident helper response mismatch or failure')
            return result
        _helper_receipt(cgroup, out)
        time.sleep(0.1)
    raise RuntimeError(f'reclaim helper PID {helper["pid"]} exceeded {timeout_seconds}s')


def _stop_helper(cgroup, out):
    helper = _helper_receipt(cgroup, out, require_alive=False)
    pid, starttime = helper['pid'], helper['starttime']
    for signum in (signal.SIGTERM, signal.SIGKILL):
        live = _proc_identity(pid)
        if live is None or live['state'] == 'Z':
            break
        if live['starttime'] != starttime or live['cgroup'] != str(cgroup):
            raise RuntimeError('resident helper PID identity changed before stop')
        os.kill(pid, signum)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            live = _proc_identity(pid)
            if live is None or live['state'] == 'Z':
                break
            if live['starttime'] != starttime or live['cgroup'] != str(cgroup):
                raise RuntimeError('resident helper PID identity changed during stop')
            time.sleep(0.1)
        if live is None or live['state'] == 'Z':
            break
    live = _proc_identity(pid)
    if live is not None and live['state'] != 'Z':
        raise RuntimeError('resident helper remained alive after bounded stop')
    if str(pid) in (cgroup / 'cgroup.procs').read_text().split():
        raise RuntimeError('resident helper remains in server cgroup')
    return {'pid': pid, 'starttime': starttime, 'status': 'STOPPED_BEFORE_CLIENT'}


def _check_scope(cgroup):
    if int((cgroup / 'memory.max').read_text()) != HARD_CAP_BYTES:
        raise RuntimeError('owned scope memory.max differs from 416 GiB')
    if int((cgroup / 'memory.swap.max').read_text()) != 0:
        raise RuntimeError('owned scope swap cap differs from zero')
    events = dict(line.split() for line in (cgroup / 'cgroup.events').read_text().splitlines())
    if events.get('populated') != '1':
        raise RuntimeError('owned scope is not populated')
    if not (cgroup / 'memory.reclaim').exists() or not (cgroup / 'cgroup.procs').exists():
        raise RuntimeError('owned scope lacks reclaim or process-placement interface')


def _unchanged_model_memory(before, after):
    old, new = _stat(before), _stat(after)
    current_drop = int(before['memory.current']) - int(after['memory.current'])
    ordinary_file_drop = ((old['file'] - old['shmem']) - (new['file'] - new['shmem']))
    tolerance = TELEMETRY_TOLERANCE_BYTES
    if abs(new['anon'] - old['anon']) > tolerance:
        raise RuntimeError('anon changed materially during cache stabilization')
    if abs(new['shmem'] - old['shmem']) > tolerance:
        raise RuntimeError('model shmem changed materially during cache stabilization')
    if current_drop > tolerance and ordinary_file_drop + tolerance < current_drop:
        raise RuntimeError('ordinary file-cache decline does not explain cgroup headroom')
    return {'memory_current_drop_bytes': current_drop,
            'ordinary_file_drop_bytes': ordinary_file_drop,
            'anon_delta_bytes': new['anon'] - old['anon'],
            'shmem_delta_bytes': new['shmem'] - old['shmem']}


def stabilize(cgroup, receipt_path, *, snapshot, check_memory, memory_events,
              verify_helper=_helper_receipt, write_reclaim=_run_helper,
              stop_helper=_stop_helper, sleep=time.sleep, clock=time.monotonic):
    """Return the final stable snapshot or raise after writing a FAILED receipt."""
    cgroup, receipt_path = Path(cgroup), Path(receipt_path)
    receipt = {'status': 'STARTED', 'cgroup': str(cgroup),
               'policy': {'memory_max_bytes': HARD_CAP_BYTES, 'swap_max_bytes': 0,
                          'target_bytes': TARGET_BYTES, 'max_request_bytes': MAX_REQUEST_BYTES,
                          'max_attempts': MAX_ATTEMPTS,
                          'helper_timeout_seconds': HELPER_TIMEOUT_SECONDS,
                          'total_timeout_seconds': TOTAL_TIMEOUT_SECONDS,
                          'settle_seconds': SETTLE_SECONDS,
                          'telemetry_tolerance_bytes': TELEMETRY_TOLERANCE_BYTES,
                          'swappiness': 0},
               'attempts': [], 'settle': [], 'settle_times_monotonic': []}
    started = clock()
    helper_stopped = False
    try:
        _check_scope(cgroup)
        receipt['helper_ready'] = verify_helper(cgroup, receipt_path.parent)
        before = snapshot(cgroup)
        check_memory(before, phase='LOADING')
        receipt['before'] = before
        _save(receipt_path, receipt)
        current = before
        for number in range(MAX_ATTEMPTS):
            remaining = int(current['memory.current']) - TARGET_BYTES
            if remaining <= 0:
                break
            eligibility = _eligibility(current)
            if eligibility < remaining:
                raise RuntimeError('insufficient eligible ordinary inactive file cache')
            amount = min(MAX_REQUEST_BYTES, eligibility)
            if clock() - started + HELPER_TIMEOUT_SECONDS + SETTLE_SECONDS > TOTAL_TIMEOUT_SECONDS:
                raise RuntimeError('cache stabilization total deadline exhausted')
            attempt = {'number': number + 1, 'requested_bytes': amount,
                       'eligibility_before_bytes': eligibility,
                       'memory_current_before_bytes': int(current['memory.current'])}
            receipt['attempts'].append(attempt)
            _save(receipt_path, receipt)
            try:
                attempt['helper'] = write_reclaim(cgroup, receipt_path.parent, amount, number + 1,
                                                   timeout_seconds=HELPER_TIMEOUT_SECONDS)
            except BaseException as exc:
                attempt['error'] = repr(exc)
                raise
            after = snapshot(cgroup)
            check_memory(after, phase='LOADING')
            attempt['after'] = after
            _unchanged_model_memory(before, after)
            progress = int(current['memory.current']) - int(after['memory.current'])
            if progress < MIN_PROGRESS_BYTES and int(after['memory.current']) > TARGET_BYTES:
                raise RuntimeError('reclaim made less than 256 MiB progress')
            current = after
            _save(receipt_path, receipt)
        if int(current['memory.current']) > TARGET_BYTES:
            raise RuntimeError('cache stabilization did not reach 392 GiB target')
        receipt['helper_stop'] = stop_helper(cgroup, receipt_path.parent)
        helper_stopped = True
        receipt['helper_stopped_monotonic'] = clock()
        _save(receipt_path, receipt)
        for index in range(2):
            if index:
                sleep(SETTLE_SECONDS)
            if clock() - started > TOTAL_TIMEOUT_SECONDS:
                raise RuntimeError('cache stabilization total deadline exhausted during settle')
            sample = snapshot(cgroup)
            check_memory(sample, phase='LOADING')
            if int(sample['memory.current']) > TARGET_BYTES:
                raise RuntimeError('cache charge rebounded above 392 GiB during settle')
            _unchanged_model_memory(before, sample)
            receipt['settle'].append(sample)
            receipt['settle_times_monotonic'].append(clock())
            _save(receipt_path, receipt)
        if memory_events(receipt['settle'][0]).get('max', 0) != memory_events(receipt['settle'][1]).get('max', 0):
            raise RuntimeError('memory.max events increased during idle settle')
        _check_scope(cgroup)
        final = receipt['settle'][-1]
        receipt['attribution'] = _unchanged_model_memory(before, final)
        receipt['ready_events'] = memory_events(final)
        receipt['status'] = 'PASS'
        receipt['elapsed_seconds'] = clock() - started
        _save(receipt_path, receipt)
        return final
    except BaseException as exc:
        receipt['status'] = 'FAILED'
        receipt['error'] = repr(exc)
        if not helper_stopped:
            try:
                receipt['helper_stop'] = stop_helper(cgroup, receipt_path.parent)
            except BaseException as stop_exc:
                receipt['helper_stop_error'] = repr(stop_exc)
        receipt['elapsed_seconds'] = clock() - started
        _save(receipt_path, receipt)
        if isinstance(exc, TimeoutError):
            raise RuntimeError('cache stabilization timeout') from exc
        raise


def validate_receipt(path, memory_before_client=None):
    """Validate a completed A/B stabilization receipt before later comparison."""
    path = Path(path)
    receipt = json.loads(path.read_text())
    expected_policy = {'memory_max_bytes': HARD_CAP_BYTES, 'swap_max_bytes': 0,
                       'target_bytes': TARGET_BYTES, 'max_request_bytes': MAX_REQUEST_BYTES,
                       'max_attempts': MAX_ATTEMPTS,
                       'helper_timeout_seconds': HELPER_TIMEOUT_SECONDS,
                       'total_timeout_seconds': TOTAL_TIMEOUT_SECONDS,
                       'settle_seconds': SETTLE_SECONDS,
                       'telemetry_tolerance_bytes': TELEMETRY_TOLERANCE_BYTES,
                       'swappiness': 0}
    if receipt.get('status') != 'PASS' or receipt.get('policy') != expected_policy:
        raise RuntimeError('cache stabilization status or policy mismatch')
    elapsed = receipt.get('elapsed_seconds')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed <= TOTAL_TIMEOUT_SECONDS:
        raise RuntimeError('cache stabilization elapsed time outside bound')
    before = receipt['before']
    cgroup = receipt['cgroup']
    helper_ready = receipt.get('helper_ready', {})
    if (helper_ready.get('status') != 'READY' or helper_ready.get('cgroup') != cgroup
            or type(helper_ready.get('pid')) is not int
            or type(helper_ready.get('starttime')) is not int
            or not helper_ready.get('nonce')):
        raise RuntimeError('resident helper identity missing from receipt')
    attempts = receipt['attempts']
    settled = receipt['settle']
    if len(attempts) > MAX_ATTEMPTS or len(settled) != 2:
        raise RuntimeError('cache stabilization attempt/settle count mismatch')
    times = receipt.get('settle_times_monotonic')
    stop_time = receipt.get('helper_stopped_monotonic')
    if (not isinstance(times, list) or len(times) != 2
            or not all(type(x) in (int, float) for x in times)
            or type(stop_time) not in (int, float)
            or not stop_time <= times[0] <= times[1]
            or times[1] - times[0] < SETTLE_SECONDS):
        raise RuntimeError('helper stop and settle ordering mismatch')
    previous = before
    if before.get('cgroup') != cgroup:
        raise RuntimeError('before snapshot belongs to another cgroup')
    for number, attempt in enumerate(attempts, 1):
        amount = attempt['requested_bytes']
        eligibility = _eligibility(previous)
        remaining = int(previous['memory.current']) - TARGET_BYTES
        if (attempt['number'] != number or type(amount) is not int
                or remaining <= 0 or eligibility < remaining
                or amount != min(MAX_REQUEST_BYTES, eligibility)
                or attempt['eligibility_before_bytes'] != eligibility
                or attempt['memory_current_before_bytes'] != int(previous['memory.current'])):
            raise RuntimeError('cache stabilization request outside policy')
        helper = attempt['helper']
        if (helper.get('number') != number or helper.get('requested_bytes') != amount
                or helper.get('status') not in ('WRITE_COMPLETE', 'EAGAIN')
                or helper.get('cgroup') != cgroup
                or any(helper.get(key) != helper_ready[key]
                       for key in ('pid', 'starttime', 'nonce'))):
            raise RuntimeError('cache stabilization helper receipt mismatch')
        after = attempt['after']
        if after.get('cgroup') != cgroup or int(after['memory.swap.current']) != 0:
            raise RuntimeError('reclaim snapshot cgroup/swap mismatch')
        _unchanged_model_memory(before, after)
        progress = int(previous['memory.current']) - int(after['memory.current'])
        if progress < MIN_PROGRESS_BYTES and int(after['memory.current']) > TARGET_BYTES:
            raise RuntimeError('reclaim attempt did not make minimum progress')
        previous = after
    if int(previous['memory.current']) > TARGET_BYTES:
        raise RuntimeError('cache stabilization did not reach target')
    final = settled[-1]
    for sample in settled:
        if sample.get('cgroup') != cgroup:
            raise RuntimeError('settle snapshot belongs to another cgroup')
        if int(sample['memory.current']) > TARGET_BYTES or int(sample['memory.swap.current']) != 0:
            raise RuntimeError('cache stabilization settle exceeded charge/swap target')
        events = {k: int(v) for k, v in
                  (line.split() for line in sample['memory.events'].splitlines())}
        if any(events.get(k, 0) for k in ('high', 'oom', 'oom_kill', 'oom_group_kill')):
            raise RuntimeError('cache stabilization settle had high/OOM events')
        _unchanged_model_memory(before, sample)
    final_events = {k: int(v) for k, v in
                    (line.split() for line in final['memory.events'].splitlines())}
    first_events = {k: int(v) for k, v in
                    (line.split() for line in settled[0]['memory.events'].splitlines())}
    if first_events.get('max', 0) != final_events.get('max', 0):
        raise RuntimeError('memory.max events increased during idle settle')
    if receipt['ready_events'] != final_events:
        raise RuntimeError('ready events differ from final settled snapshot')
    if receipt.get('attribution') != _unchanged_model_memory(before, final):
        raise RuntimeError('cache stabilization attribution mismatch')
    if receipt.get('helper_stop', {}).get('status') != 'STOPPED_BEFORE_CLIENT':
        raise RuntimeError('cache stabilization helper not stopped before client')
    if any(receipt['helper_stop'].get(key) != helper_ready[key]
           for key in ('pid', 'starttime')):
        raise RuntimeError('stopped helper differs from ready helper')
    if str(helper_ready['pid']) in final.get('cgroup.procs', '').split():
        raise RuntimeError('resident helper was still in cgroup at baseline')
    if memory_before_client is not None:
        if json.loads(Path(memory_before_client).read_text()) != final:
            raise RuntimeError('client memory baseline differs from stabilized snapshot')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resident', action='store_true')
    parser.add_argument('cgroup', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('nonce')
    args = parser.parse_args()
    try:
        if not args.resident:
            raise RuntimeError('resident helper mode required')
        resident_helper(args.cgroup, args.output, args.nonce)
    except BaseException as exc:
        print(json.dumps({'status': 'FAILED', 'error': repr(exc), 'pid': os.getpid()}),
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
