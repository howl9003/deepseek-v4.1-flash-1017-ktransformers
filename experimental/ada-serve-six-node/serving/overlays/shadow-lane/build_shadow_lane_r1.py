"""Native build of the KT shadow lane (shadow-lane-r1, Claude, 4 Nov 2026): the DSpark tolerant tree
(ada-native-dspark-tolerant-20261103-r1, kt_ext 57e25283) plus apply_shadow_lane.py, selectors KT_SHADOW_LANE and
KT_SHADOW_LANE_FROM, default off. First test_lane.cpp against the patched cpu_backend: its pool is built from the
unbound constructors, so every thread stays on the caller's taskset cores and never runs on a served worker core (a
user unit cannot hold threads to a CPU set on fea01: no cpuset delegation). CPU only, new folder; nothing served
reads this tree. Run with CUDA_VISIBLE_DEVICES='' under taskset -c <two spare cores>, at a time with no timing run."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, time

B = Path('/scratch/aster/bench')
G = B / 'phase-g-g0-native-20261030-r1'
S = B / 'ada-native-dspark-tolerant-20261103-r1'
N = B / 'ada-native-shadow-lane-20261104-r1'
C = B / 'six-node-performance-r1/kernel-steer-r1/route3-mma-r1/dspark-cpu-r1/shadow-lane-r1'
ENV = B / 'phase-g-g0-environment-20261030-r3/venv'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert len(os.sched_getaffinity(0)) <= 2, 'run under taskset with two spare cores'
N.mkdir(exist_ok=False)
report = dict(status='RUNNING', steps=[], default_off=True, base=str(S), base_artifact_sha256=sha(next((S / 'kt-build').glob('kt_kernel_ext*.so'))),
              overlay_sha256=sha(C / 'apply_shadow_lane.py'), test_sha256=sha(C / 'test_lane.cpp'))


def save():
    (N / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')


def run(cmd, label, env=None, timeout=3000):
    start = time.monotonic()
    with (N / (label + '.log')).open('w') as log:
        result = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    report['steps'].append(dict(label=label, exit=result.returncode, seconds=time.monotonic() - start))
    save()
    assert result.returncode == 0, label


try:
    shutil.copytree(S / 'ktransformers', N / 'ktransformers', symlinks=True)
    kt = N / 'ktransformers/kt-kernel'
    subprocess.run(['/usr/bin/python3', '-B', '-I', str(C / 'apply_shadow_lane.py'), str(kt)], check=True)
    report['patched'] = {str(p.relative_to(kt)): sha(p) for p in sorted((kt / 'cpu_backend').glob('*')) if p.is_file()}
    env = dict(os.environ, PATH=str(ENV / 'bin') + ':' + os.environ['PATH'], MAX_JOBS='2')
    run(['/usr/bin/g++', '-std=c++20', '-O2', '-pthread', '-I', kt / 'cpu_backend', C / 'test_lane.cpp', kt / 'cpu_backend/worker_pool.cpp',
         kt / 'cpu_backend/task_queue.cpp', '-lhwloc', '-lnuma', '-o', N / 'test_lane'], 'test-build', env)
    test_env = dict(env, KT_SHADOW_LANE='1')
    run([N / 'test_lane', '0,1', '2', '200'], 'test-lane', test_env, timeout=900)  # unbound threads on the taskset cores
    assert 'LANE_TEST_PASS' in (N / 'test-lane.log').read_text(), 'lane test'
    command = json.loads((G / 'NATIVE_RESULT.json').read_text())['steps'][0]['command']
    index = command.index('-B')
    command[2] = str(kt)
    command[index + 1] = str(N / 'kt-build')
    run(command, 'configure', env)
    run([ENV / 'bin/cmake', '--build', N / 'kt-build', '--target', 'kt_kernel_ext', '-j', '2'], 'build', env)
    binary, = list((N / 'kt-build').glob('kt_kernel_ext*.so'))
    report.update(status='NATIVE_SHADOW_LANE_BUILD_PASS', artifact=dict(path=str(binary), sha256=sha(binary)))
except BaseException as exc:
    report.update(status='FAIL', error=repr(exc))
    raise
finally:
    save()
