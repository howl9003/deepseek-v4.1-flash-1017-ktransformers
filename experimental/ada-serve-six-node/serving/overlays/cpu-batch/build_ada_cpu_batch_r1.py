"""Native build for the CPU window batch (Claude, 3 Nov 2026): the served r16 KT tree plus apply_shared_x.py and apply_cpu_batch.py,
selectors KT_CONTRACT_V2_SHARED_X, KT_CONTRACT_V2_FUSED_PHASES and KT_CONTRACT_V2_MERGE_IN_PARTS, all default off. CPU only, on spare cores, nothing served reads this tree."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, time

B = Path('/scratch/aster/bench')
G = B / 'phase-g-g0-native-20261030-r1'
S = B / 'ada-native-r16-20261101-r1'
N = B / 'ada-native-cpu-batch-20261103-r1'
M = B / 'six-node-performance-r1/ada-spill-r1/shared-x-r1'
C = B / 'six-node-performance-r1/ada-spill-r1/cpu-batch-r1'
ENV = B / 'phase-g-g0-environment-20261030-r3/venv'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
N.mkdir(exist_ok=False)
report = dict(status='RUNNING', steps=[], default_off=True, base=str(S))


def save():
    (N / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')


def run(cmd, label, env=None):
    start = time.monotonic()
    with (N / (label + '.log')).open('w') as log:
        result = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, timeout=3000)
    report['steps'].append(dict(label=label, exit=result.returncode, seconds=time.monotonic() - start))
    save()
    assert result.returncode == 0, label


try:
    shutil.copytree(S / 'ktransformers', N / 'ktransformers', symlinks=True)
    kt = N / 'ktransformers/kt-kernel'
    header = kt / 'operators/avx2/contract_v2-moe.hpp'
    report['header_before'] = sha(header)
    subprocess.run(['/usr/bin/python3', '-B', str(M / 'apply_shared_x.py'), str(header)], check=True)
    subprocess.run(['/usr/bin/python3', '-B', str(C / 'apply_cpu_batch.py'), str(header)], check=True)
    report['header_after'] = sha(header)
    command = json.loads((G / 'NATIVE_RESULT.json').read_text())['steps'][0]['command']
    index = command.index('-B')
    command[2] = str(kt)
    command[index + 1] = str(N / 'kt-build')
    env = dict(os.environ, PATH=str(ENV / 'bin') + ':' + os.environ['PATH'])
    run(command, 'configure', env)
    run([ENV / 'bin/cmake', '--build', N / 'kt-build', '--target', 'kt_kernel_ext', '-j', '3'], 'build', env)
    binary, = list((N / 'kt-build').glob('kt_kernel_ext*.so'))
    report.update(status='NATIVE_CPU_BATCH_BUILD_PASS', artifact=dict(path=str(binary), sha256=sha(binary)))
except BaseException as exc:
    report.update(status='FAIL', error=repr(exc))
    raise
finally:
    save()
