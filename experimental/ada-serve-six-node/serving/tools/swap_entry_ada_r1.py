"""llama-swap child for the SGLang build (Claude, 2 Nov 2026). Runs in the swap service, starts the serve controller in aster's user manager and waits on it.

The server needs the 416 GiB, all-eight-memory-node scope every bench run used, not the stack's 320 GiB slice, so it cannot be a child of the swap service. The user manager lingers and already owns every
bench scope. llama-swap's stop (SIGTERM) asks the controller to stop through its stop file, which keeps the normal cleanup. No guard lives here: admission, memory and provenance are the controller's.
"""
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BENCH = Path('/scratch/aster/bench')
TOOLS = BENCH / 'six-node-performance-r1/tools'
RUNS = BENCH / 'ada-serve-20261102-r4/runs'
STOP_WAIT_S = 330
os.environ.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
os.environ.setdefault('DBUS_SESSION_BUS_ADDRESS', f"unix:path={os.environ['XDG_RUNTIME_DIR']}/bus")


def user(*args, timeout=30):
    return subprocess.run(['systemctl', '--user', *args], capture_output=True, text=True, timeout=timeout)


def state(unit):
    return user('show', unit, '-p', 'ActiveState', '--value').stdout.strip()


def stop_unit(unit, run):
    (run / 'RESIDENT_STOP').touch()
    end = time.monotonic() + STOP_WAIT_S
    while time.monotonic() < end and state(unit) in ('active', 'activating', 'deactivating'):
        time.sleep(2)
    if state(unit) in ('active', 'activating', 'deactivating'):
        user('stop', unit, timeout=120)


def leftovers():
    """A swap service killed hard leaves the unit running. Stop it before starting another."""
    listed = user('list-units', '--no-legend', '--plain', 'ada-serve-*.service').stdout
    for line in listed.splitlines():
        unit = line.split()[0]
        stamp = unit[len('ada-serve-'):-len('.service')]
        print('stopping', unit, flush=True)
        stop_unit(unit, RUNS / stamp)


def main():
    leftovers()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run, unit = RUNS / stamp, f'ada-serve-{stamp}.service'
    RUNS.mkdir(parents=True, exist_ok=True)
    started = subprocess.run(['systemd-run', '--user', '--unit=' + unit, '--collect', '-p', 'MemoryMax=2G', '-p', 'MemorySwapMax=0', '-p', 'TimeoutStopSec=300',
                              '/usr/bin/python3', '-B', str(TOOLS / 'serve_ada_r1.py'), '--phase', 'model', '--run-dir', str(run)], capture_output=True, text=True, timeout=30)
    if started.returncode:
        sys.exit('could not start ' + unit + ': ' + started.stderr)
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(True))
    print('started', unit, 'run', run, flush=True)
    while state(unit) in ('active', 'activating'):
        if stopping:
            stop_unit(unit, run)
            return
        time.sleep(1)
    sys.exit('serve controller ended; see ' + str(run / 'RUN.json'))


if __name__ == '__main__':
    main()
