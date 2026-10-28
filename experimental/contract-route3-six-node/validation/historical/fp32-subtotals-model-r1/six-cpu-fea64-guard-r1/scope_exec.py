"""Verify an owned, capped, six-CPU-node / all-eight-memory-node scope before executing its payload."""

import ctypes
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time


out = Path(sys.argv[1])
cap_gib = int(sys.argv[2])
command = sys.argv[3:]
assert command and cap_gib in (64, 416)
cg = Path("/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split("::")[1])
assert int((cg / "memory.max").read_text()) == cap_gib * 1024**3
assert int((cg / "memory.swap.max").read_text()) == 0
(cg / "memory.oom.group").write_text("1")
assert (cg / "memory.oom.group").read_text().strip() == "1"
assert os.sched_getaffinity(0) == set(range(24)) | set(range(32,56))
lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
mode, mask = ctypes.c_int(), ctypes.c_ulong()
assert lib.get_mempolicy(ctypes.byref(mode), ctypes.byref(mask), 64, None, 0) == 0
assert mode.value == 3 and mask.value == 255, (mode.value, mask.value)
reclaim_server = os.environ.get('ACCEPT_RECLAIM_HELPER') == '1'
if reclaim_server:
    assert cap_gib == 416 and out.name == 'server', 'reclaim helper is final-server only'
nonce = secrets.token_hex(16) if reclaim_server else None
scope = {"cgroup": str(cg), "pid": os.getpid(),
         "policy": "interleave:0-7", "cap_gib": cap_gib, "command": command}
if reclaim_server:
    scope['reclaim_nonce'] = nonce
tmp = out / "SCOPE.tmp"
tmp.write_text(json.dumps(scope))
tmp.replace(out / "SCOPE.json")
if reclaim_server:
    helper_script = Path(__file__).resolve().parent / 'cache_stabilization.py'
    with (out / 'reclaim-helper.log').open('w') as log:
        helper = subprocess.Popen([sys.executable, str(helper_script), '--resident',
                                   str(cg), str(out), nonce], stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        ready = out / 'RECLAIM_HELPER.json'
        while not ready.exists() and time.monotonic() < deadline:
            if helper.poll() is not None:
                raise RuntimeError('resident reclaim helper exited before startup')
            time.sleep(0.1)
        if not ready.exists():
            raise RuntimeError('resident reclaim helper startup timed out')
        detail = json.loads(ready.read_text())
        assert detail['pid'] == helper.pid and detail['cgroup'] == str(cg)
        assert detail['nonce'] == nonce and detail['status'] == 'READY'
        assert helper.poll() is None, 'resident reclaim helper died after receipt'
    except BaseException:
        if helper.poll() is None:
            helper.kill()
        helper.wait(timeout=5)
        raise
    os.environ.pop('ACCEPT_RECLAIM_HELPER', None)
os.execv(command[0], command)
