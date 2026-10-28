"""Guarded build of the authenticated, unmodified Claude phase-2 package. No model."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback

P = Path('/scratch/aster/bench/six-node-performance-r1')
B = P.parent
PKG = P/'route3-phase2-l2-policy-r1/phase2-cuda'
OUT = P/'route3-phase2-build-r11'
G = 2**30
UUID = 'GPU-67d9f495-2d3a-1e93-a617-dd11ba547914'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cg = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
assert os.sched_getaffinity(0) == {55}
assert int((cg/'memory.max').read_text()) == 8*G
assert (cg/'memory.swap.max').read_text().strip() == '0'
events = (cg/'memory.events').read_text()
available = int(re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)[1])*1024
assert available >= 96*G
lock = (B/'.bench.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,power.limit', '--format=csv,noheader'], text=True).strip() == UUID+', 320.00 W'
assert sha(B/'feature-trial-B-contract-kernel-numerics-six-r3/client-tools/kv_checks.py') == 'eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
pin = json.loads((P/'ROUTE3_PHASE2_BUILD_PREPARATION_r11.json').read_text())
for rel, digest in pin['files'].items():
    assert sha(P/rel) == digest, rel
OUT.mkdir(exist_ok=False)
report = dict(status='BUILD_RUNNING', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              preparation_sha256=sha(P/'ROUTE3_PHASE2_BUILD_PREPARATION_r11.json'),
              package=str(PKG), no_model=True, selector_default_off=True)

def save():
    report.update(memory_peak_bytes=int((cg/'memory.peak').read_text()),
                  memory_events=(cg/'memory.events').read_text(), swap_bytes=int((cg/'memory.swap.current').read_text()))
    (OUT/'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')

save()
try:
    os.environ.update(CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='8.9', MAX_JOBS='1',
                      TORCH_EXTENSIONS_DIR=str(OUT/'extensions'), ROUTE3_EXT_NAME='route3_phase2_l2p0_r1',
                      ROUTE3_BUILD_VERBOSE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    headers = B/'05yuki-main-00fd12a-candidate/.venv/lib/python3.12/site-packages/nvidia/cu13/include'
    assert (headers/'cusparse.h').exists()
    os.environ['ROUTE3_L2_POLICY'] = '0'
    report['ROUTE3_L2_POLICY'] = 0
    os.environ.pop('CPATH', None)
    os.environ['CPLUS_INCLUDE_PATH'] = str(headers)
    report['include_path_fix'] = str(headers)
    report['cusparse_header_sha256'] = sha(headers/'cusparse.h')
    sys.path.insert(0, str(PKG))
    import route3_decode as candidate
    ext = candidate._load()
    binary = Path(ext.__file__)
    report.update(binary=str(binary), binary_sha256=sha(binary), smem_bytes=candidate.smem_bytes())
    assert report['smem_bytes'] == 38496
    tool = B/'05yuki-main-00fd12a-candidate/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump'
    if not tool.exists():
        tool = Path('/usr/local/cuda/bin/cuobjdump')
    report['cuobjdump'] = dict(path=str(tool), sha256=sha(tool))
    for flag, name in [('--dump-resource-usage', 'resources.txt'), ('--dump-sass', 'kernel.sass'), ('--dump-ptx', 'kernel.ptx')]:
        r = subprocess.run([str(tool), flag, str(binary)], capture_output=True, text=True, timeout=90)
        (OUT/name).write_text(r.stdout)
        (OUT/(name+'.stderr')).write_text(r.stderr)
        assert r.returncode == 0, (flag, r.stderr)
        report[name] = dict(sha256=sha(OUT/name), bytes=(OUT/name).stat().st_size)
    resources = (OUT/'resources.txt').read_text()
    sass = (OUT/'kernel.sass').read_text()
    assert 'QMMA.16832.F32.E4M3.E4M3' in sass
    assert 'FMUL' in sass and 'FFMA' in sass
    report['sass_counts'] = {s: sass.count(s) for s in ['QMMA.16832.F32.E4M3.E4M3', 'FMUL', 'FFMA', 'LDL', 'STL', 'LDSM', 'BAR.SYNC', 'LDGSTS', 'LDGDEPBAR']}
    report['resources'] = resources
    assert (cg/'memory.events').read_text() == events
    assert (cg/'memory.swap.current').read_text().strip() == '0'
    assert int(re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)[1])*1024 >= 64*G
    report['status'] = 'BUILD_PASS_SASS_REVIEW_REQUIRED_NO_IDENTITY_OR_PRICE_YET'
except BaseException as exc:
    report.update(status='BUILD_FAILED_STOPPED', error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(report['status'], flush=True)
