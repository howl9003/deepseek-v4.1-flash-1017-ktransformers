"""Launch Phase G G1v arm 0 only after fresh unchanged admission (Claude, 30 Oct 2026).

dispatch_r19_C1_static_on_identity_r2.py without the C1-specific A3 receipt and the profiler-permission
check (arm 0 runs no profiler). One foreground invocation, no schedule, no cache clearing or guard change.
The controller reacquires the shared lock and repeats its own admission and occupancy checks.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess
P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument('--execute', action='store_true'); args = ap.parse_args()
expected = hashlib.sha256((P / 'tools/controller_phase_g_g1v_adayperf_r1.py').read_bytes()).hexdigest()
code = r'''from pathlib import Path
import datetime,fcntl,hashlib,json,subprocess,sys
B=Path('/scratch/aster/bench');P=B/'six-node-performance-r1';T=B/'ada-perf-engram-20261101-r1'
unit='phase-g-g1v-adayperf-controller-r1.service';controller=P/'tools/controller_phase_g_g1v_adayperf_r1.py'
assert hashlib.sha256(controller.read_bytes()).hexdigest()==EXPECTED
assert not (T/'RUN-phase-g-g1v-adayperf-r1.json').exists(),'Never repeat the prepared namespace'
assert not (T/'PHASE_G_ADAYPERF_LAUNCH_OWNER.json').exists(),'Launch ownership already exists'
assert subprocess.check_output(['systemctl','--user','show',unit,'-p','LoadState','--value'],text=True).strip()=='not-found'
lock=(B/'.bench.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(P/'tools'));import controller_phase_g_g1v_adayperf_r1 as ctl
prepared=ctl.verify_prepared(T);ctl.check_cache(T)
R=B/'fp32-subtotals-model-r1';pins=ctl.read(R/'workflow/HARNESS_PINS.json')
mods=ctl.load_frozen(B/'eight-node-ab-tools/retry_v9',B/'eight-node-ab-tools/pr-acceptance-retry-v9.manifest.json',pins['B']['manifest_sha256'],a_harness=B/'eight-node-ab-tools/retry_v7',a_manifest=B/'eight-node-ab-tools/pr-acceptance-retry-v7.manifest.json',a_manifest_sha256=pins['A']['manifest_sha256'])
host=mods.run_ab.host();gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
power=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,power.limit','--format=csv,noheader'],text=True).strip()
assert power=='GPU-67d9f495-2d3a-1e93-a617-dd11ba547914, 320.00 W'
ready=host['memavailable']>=480*2**30 and min(v['free_plus_clean'] for v in host['nodes'].values())>=51*2**30 and host['cma_guard']['pass_gate'] and not gpu
result=dict(status='READY' if ready else 'HELD',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),global_gib=host['memavailable']/2**30,nodes_gib={k:v['free_plus_clean']/2**30 for k,v in host['nodes'].items()},cma=host['cma_guard'],gpu_compute=gpu,power=power,trial=str(T),controller_sha256=EXPECTED,owner='Claude takeover, owner request 30 Oct 2026')
fcntl.flock(lock,fcntl.LOCK_UN)
if not EXECUTE or not ready:
 print(json.dumps(result));sys.exit(0 if ready else 3)
cmd=['systemd-run','--user','--unit='+unit,'--remain-after-exit','-p','MemoryMax=2G','-p','MemorySwapMax=0','-p','RuntimeMaxSec=7200','-p','TimeoutStopSec=420','/usr/bin/python3','-B',str(controller),'--phase','model']
x=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
result.update(status='DISPATCHED' if x.returncode==0 else 'DISPATCH_FAILED',unit=unit,command=cmd,returncode=x.returncode,stderr=x.stderr,invocation_id=subprocess.check_output(['systemctl','--user','show',unit,'-p','InvocationID','--value'],text=True).strip())
with (T/'PHASE_G_ADAYPERF_LAUNCH_OWNER.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result));assert x.returncode==0
'''.replace('EXPECTED', repr(expected)).replace('EXECUTE', repr(args.execute))
x = subprocess.run(['ssh', 'fea01', 'python3', '-B', '-'], input=code, text=True, capture_output=True, timeout=180)
out = P / 'phase-g-r1/g1v-adayperf-r1'
out.mkdir(parents=True, exist_ok=True)
import datetime
name = ('LAUNCH' if args.execute else 'ADMISSION_CHECK') + datetime.datetime.now(datetime.timezone.utc).strftime('-%Y%m%dT%H%M%SZ')
with (out / (name + '.stdout')).open('x', encoding='utf-8') as f: f.write(x.stdout)
with (out / (name + '.stderr')).open('x', encoding='utf-8') as f: f.write(x.stderr)
print(x.stdout, x.stderr); raise SystemExit(x.returncode)
