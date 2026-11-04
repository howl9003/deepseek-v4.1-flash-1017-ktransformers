"""Native build for the DSpark CPU draft (Claude, 3 Nov 2026): the served r16 KT tree plus apply_tolerant.py, selector
KT_CONTRACT_V2_TOLERANT_FROM, default off. CPU only, on the two spare cores, new folder; nothing served reads this tree."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, time

B = Path('/scratch/aster/bench')
G = B / 'phase-g-g0-native-20261030-r1'
S = B / 'ada-native-r16-20261101-r1'
N = B / 'ada-native-dspark-tolerant-20261103-r1'
C = B / 'six-node-performance-r1/kernel-steer-r1/route3-mma-r1/dspark-cpu-r1'
ENV = B / 'phase-g-g0-environment-20261030-r3/venv'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
N.mkdir(exist_ok=False)
report = dict(status='RUNNING', steps=[], default_off=True, base=str(S), base_artifact_sha256=sha(next((S / 'kt-build').glob('kt_kernel_ext*.so'))))


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
    files = [kt / 'operators/avx2/contract_v2-moe.hpp', kt / 'ext_bindings.cpp']
    report['before'] = {f.name: sha(f) for f in files}
    subprocess.run(['/usr/bin/python3', '-B', '-I', str(C / 'apply_tolerant.py'), str(kt)], check=True)
    report['after'] = {f.name: sha(f) for f in files}
    command = json.loads((G / 'NATIVE_RESULT.json').read_text())['steps'][0]['command']
    index = command.index('-B')
    command[2] = str(kt)
    command[index + 1] = str(N / 'kt-build')
    env = dict(os.environ, PATH=str(ENV / 'bin') + ':' + os.environ['PATH'], MAX_JOBS='2')
    run(command, 'configure', env)
    run([ENV / 'bin/cmake', '--build', N / 'kt-build', '--target', 'kt_kernel_ext', '-j', '2'], 'build', env)
    binary, = list((N / 'kt-build').glob('kt_kernel_ext*.so'))
    report.update(status='NATIVE_DSPARK_TOLERANT_BUILD_PASS', artifact=dict(path=str(binary), sha256=sha(binary)))
except BaseException as exc:
    report.update(status='FAIL', error=repr(exc))
    raise
finally:
    save()
