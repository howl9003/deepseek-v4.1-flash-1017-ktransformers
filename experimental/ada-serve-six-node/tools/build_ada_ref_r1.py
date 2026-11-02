"""Native build with the route partials read in place (Claude, 31 Oct 2026): the staggered-spill KT tree plus one default-off environment selector.

KT_CONTRACT_V2_ROUTE_REF=1: the merge reads each route's int64 partial where the down projection left it, instead of a per-route copy. That copy is 57 KiB a route on each
part's main thread, after the workers have finished: the serial gap before merge in the worker traces, 1.2-1.6 ms a token. Both buffers hold the same integers and live to the
end of the forward, so no arithmetic changes. Unset, the copy runs as accepted. CPU only."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, time

B = Path('/scratch/aster/bench')
G = B / 'phase-g-g0-native-20261030-r1'
S = B / 'ada-native-stagger-20261031-r1'
N = B / 'ada-native-ref-20261031-r1'
ENV = B / 'phase-g-g0-environment-20261030-r3/venv'
HEADER_BEFORE = '38fe6e01261f1e0e0845f032224d451d4d39a17e0935e68b7dbd7c1fc089b538'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
N.mkdir(exist_ok=False)
report = dict(status='RUNNING', steps=[], default_off=True, base=str(S))


def save():
    (N / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')


def run(cmd, label, env=None):
    start = time.monotonic()
    with (N / (label + '.log')).open('w') as log:
        result = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1500)
    report['steps'].append(dict(label=label, exit=result.returncode, seconds=time.monotonic() - start))
    save()
    assert result.returncode == 0, label


EDITS = [
    ('''  int64_t* route_acc_ = nullptr;
''', '''  // KT_CONTRACT_V2_ROUTE_REF=1: the merge reads a route's partial in place, not from a copy made serially on the part's main thread after the workers finished.
  // The down projection's buffer lives to the end of the forward and holds the same integers. Off: the copy, as accepted.
  static bool route_ref() {
    static const bool enabled = [] {
      const char* value = std::getenv("KT_CONTRACT_V2_ROUTE_REF");
      return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
  }
  std::vector<const int64_t*> route_ref_;  // where the merge reads each (token, slot)
  int64_t* route_acc_ = nullptr;
'''),
    ('''      route_acc_ = s.acc.data(); route_ea_ = s.ea.data(); route_expert_ = s.expert.data();
    }

    int activated_expert = 0;
''', '''      route_acc_ = s.acc.data(); route_ea_ = s.ea.data(); route_expert_ = s.expert.data();
      if (route_ref_.size() < (size_t)qlen * k) route_ref_.resize((size_t)qlen * k);
    }

    int activated_expert = 0;
'''),
    ('''        std::memcpy(route_acc_ + r * config_.hidden_size, down_bc_[e]->acc + (size_t)row * config_.hidden_size, sizeof(int64_t) * config_.hidden_size);
''', '''        const int64_t* partial = down_bc_[e]->acc + (size_t)row * config_.hidden_size;
        if (route_ref()) route_ref_[r] = partial;
        else {
          std::memcpy(route_acc_ + r * config_.hidden_size, partial, sizeof(int64_t) * config_.hidden_size);
          route_ref_[r] = route_acc_ + r * config_.hidden_size;
        }
'''),
    ('''      acc[p] = tp->route_acc_ + r * c.hidden_size;
''', '''      acc[p] = tp->route_ref_[r];
'''),
]

try:
    shutil.copytree(S / 'ktransformers', N / 'ktransformers', symlinks=True)
    kt = N / 'ktransformers/kt-kernel'
    assert sha(kt / 'operators/avx2/expert_arena_spill.hpp') == HEADER_BEFORE
    header = kt / 'operators/avx2/contract_v2-moe.hpp'
    report['moe_header_before'] = sha(header)
    text = header.read_text()
    for old, new in EDITS:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    header.write_text(text)
    report['moe_header_after'] = sha(header)
    command = json.loads((G / 'NATIVE_RESULT.json').read_text())['steps'][0]['command']
    index = command.index('-B')
    command[2] = str(kt)
    command[index + 1] = str(N / 'kt-build')
    env = dict(os.environ, PATH=str(ENV / 'bin') + ':' + os.environ['PATH'])
    run(command, 'configure', env)
    run([ENV / 'bin/cmake', '--build', N / 'kt-build', '--target', 'kt_kernel_ext', '-j', '8'], 'build', env)
    binary, = list((N / 'kt-build').glob('kt_kernel_ext*.so'))
    report.update(status='NATIVE_REF_BUILD_PASS', artifact=dict(path=str(binary), sha256=sha(binary)))
except BaseException as exc:
    report.update(status='FAIL', error=repr(exc))
    raise
finally:
    save()
