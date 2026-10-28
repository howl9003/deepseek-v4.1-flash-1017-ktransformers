"""Build an isolated default-off AVX2 quantizer; never install into serving runtime."""
from pathlib import Path
import fcntl, hashlib, json, os, re, shutil, signal, subprocess, time
P = Path(__file__).resolve().parent.parent
B = P.parent
OLD = P / 'contract-native-quant-option-r1'
NEW = P / 'contract-native-rebalance-r1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
lock = (B / '.bench.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and os.sched_getaffinity(0) == {55}
cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
assert int((cg / 'memory.max').read_text()) == 4 * 2**30
assert (cg / 'memory.swap.max').read_text().strip() == '0'
assert int(re.search(r'^MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)[1]) * 1024 >= 96 * 2**30
assert sha(B / 'feature-trial-B-contract-v2-model-r1/client-tools/kv_checks.py') == 'eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
old = read(OLD / 'RESULT.json')
assert old['status'] == 'PASS_PRIVATE_QUANT_OPTION_BUILD_NOT_RUNTIME_VALIDATED'
assert sha(Path(old['artifact']['path'])) == old['artifact']['sha256']
pin = read(P / 'ARENA_REBALANCE_PREPARATION_r1.json')
rel = Path('kt-kernel/operators/avx2/expert_arena_spill.hpp')
assert sha(OLD / 'src' / rel) == pin['original_sha256']
candidate = P / 'arena-rebalance-source-r1/expert_arena_spill.hpp'
assert sha(candidate) == pin['candidate_sha256']
proof = read(P / 'arena-rebalance-test-r1/RESULT.json')
assert proof['status'] == 'PASS_PHYSICAL_PAGE_PLACEMENT_NOT_RUNTIME'
assert proof['source']['candidate_sha256'] == sha(candidate) and len(proof['checks']) == 30
for flag in ('0','1'):
    proof=read(P / ('contract-speed-native-integration-r2-q'+flag) / 'RESULT.json')
    assert proof['status']=='PASS_NATIVE_QUANT_DP4A_ADAPTER_INTEGRATION_COMPONENT_ONLY'
    assert proof['native_sha256']==old['artifact']['sha256'] and all(c['bit_exact'] for c in proof['cases'])

events = (cg / 'memory.events').read_text()
signal.alarm(900)
NEW.mkdir(exist_ok=False)
report = dict(status='RUNNING', steps=[], installed=False, default_off=True, source_preparation_sha256=sha(P / 'ARENA_REBALANCE_PREPARATION_r1.json'))

def save():
    report.update(memory_peak=int((cg / 'memory.peak').read_text()), memory_events=(cg / 'memory.events').read_text(), swap=int((cg / 'memory.swap.current').read_text()))
    (NEW / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')

def run(command, label):
    start = time.monotonic()
    with (NEW / (label + '.log')).open('w') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    report['steps'].append(dict(label=label, exit=completed.returncode, seconds=time.monotonic()-start, command=command))
    save()
    assert completed.returncode == 0, label

try:
    shutil.copytree(OLD / 'src', NEW / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(candidate, NEW / 'src' / rel)
    changed = sorted(str(p.relative_to(NEW / 'src')) for p in (NEW / 'src').rglob('*') if p.is_file() and sha(p) != sha(OLD / 'src' / p.relative_to(NEW / 'src')))
    assert changed == [str(rel)], changed
    options = next(s for s in old['steps'] if s['label'] == 'configure')['command'][5:]
    assert options[0] == '-DCMAKE_BUILD_TYPE=Release'
    run(['cmake', '-S', str(NEW / 'src/kt-kernel'), '-B', str(NEW / 'cmake'), *options], 'configure')
    run(['cmake', '--build', str(NEW / 'cmake'), '--target', 'kt_kernel_ext', '-j', '1'], 'compile')
    binary, = list((NEW / 'cmake').glob('kt_kernel_ext*.so'))
    report.update(artifact=dict(path=str(binary), sha256=sha(binary)), changed_sources=changed)
    assert events == (cg / 'memory.events').read_text() and (cg / 'memory.swap.current').read_text().strip() == '0'
    report['status'] = 'PASS_PRIVATE_REBALANCE_BUILD_NOT_RUNTIME_VALIDATED'
except BaseException as exc:
    report.update(status='FAIL', error=repr(exc))
    raise
finally:
    save()
