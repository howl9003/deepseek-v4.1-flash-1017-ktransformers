#!/usr/bin/env python3
"""Run one bounded preparation/model scope for an already reviewed feature programme.

controller_v8 (Claude, 26 Oct 2026) = controller_v7 with ROUTE3_NUMERICS_PREPARED_r1.json, receipt RUN-contract-route3-numerics-r1.json, output namespace
model-r7/ and model_client_clean_r18.py --namespace r7 (supplemental-contract-route3-numerics-r1/, benchmark-r7/, features-r7/): RUN-r6 reached readiness
and passed the settle, then its client died on RUN-r4's preserved supplemental/ and benchmark/ directories.

controller_v7 (Claude, 26 Oct 2026) = controller_v5 (the full-suite controller) with: MODEL_PREPARED_v7.json (the
Python max_len fix manifest), receipt RUN-r6.json, output namespace model-r6/, launch_payload_v15.py, and a read-only
post-readiness wait (post_ready_settle) before the unchanged frozen stabilize(): RUN-r5 aborted in the frozen settle
(two samples 20 s apart must both stay under 416 GiB) because the owned cgroup's anon charge was still growing
right after /health; RUN-r4 passed the same guard by under 60 MiB. The guard package and its pins are untouched.

Default is plan only. Root executes cache then model after A/B acceptance. No
canonical mutation, service unload/restore, publication, or numerical acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error

sys.path[:0]=['/scratch/aster/bench/fp32-subtotals-model-r1','/scratch/aster/bench/fp32-subtotals-model-r1/workflow']
from trial_common import BASE,GIB,inventory,source_inventory,load_frozen,read,require,require_programme,save,sha,verify_pair
import module_origins_fast_r1 as module_origins
from server_info_profile import split_server_info

R=Path('/scratch/aster/bench/fp32-subtotals-model-r1')
P=Path('/scratch/aster/bench/six-node-performance-r1')
HERE=R/'workflow'

def verify_prepared(trial,*,allow_private_hint_writes=False):
    p=read(trial/'ROUTE3_NUMERICS_PREPARED_r1.json')
    require(p['status']=='PREPARED_NO_RUNTIME' and p['trial']==str(trial),'preparation mismatch')
    require(sha(P/'source/bench.py')=='1c361c2010d7f69a8f1a93e33f26e7622bb684bd43a2de51b29b63a00b063f40','historical benchmark absent or changed')
    require(read(P/'CLIENT_PREFLIGHT_r1.json')['status']=='PASS_CLIENT_PROTOCOL_MOCKED_ONLY','client preflight missing')
    for name,wanted in p['pinned_files'].items():require(sha(trial/name)==wanted,'prepared file changed: '+name)
    for root,key in [('recipe/third_party/sglang','source_after'),('packages','native_after'),('recipe/engram-adapter','adapter_after'),('client-tools','client_sha256')]:
        require(inventory(trial/root)==p[key],'private source changed: '+root)
    original=Path(p['source_input'])
    require(inventory(original/'sglang')==p['source_input_inventory']['sha256'],'validated SGLang runtime changed')
    require(inventory(original/'packages')==p['native_input_inventory']['sha256'],'validated native runtime changed')
    for name,wanted in p['tool_sha256'].items():require(sha(R/name)==wanted,'tool changed: '+name)
    for name,wanted in p['shared_input_sha256'].items():require(sha(name)==wanted,'shared input changed: '+name)
    require(sha(p['setup_environment']['path'])==p['setup_environment']['sha256'],'setup changed')
    require(sha(HERE/'HARNESS_PINS.json')==p['guard_pins_sha256'],'frozen guard pins changed')
    require(sha(trial/'client-tools/kv_checks.py')=='eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4','host reload gate changed')
    return p

def check_cache(trial):
    receipt=read(trial/'CACHE_PREPARED.json')
    require(receipt['status']=='PASS' and receipt['prepared_sha256']==sha(trial/'PREPARED.json'),
            'private cache preparation missing')
    require(receipt['gpu_execution'] is False and receipt['library_loaded'] is False,'wrong cache preparation')
    for rel,h in receipt['files'].items():require(sha(trial/'.cache/sglang'/rel)==h,'private fusedMoE cache changed')
    return receipt

def unloaded(life):
    require(not life.gpu_pids(),'GPU occupied')
    loaded=[x['id'] for x in life.http('/v1/models')['data'] if x.get('status',{}).get('value')=='loaded']
    require(not loaded,'Qwen/model remains loaded; this workflow never unloads it')

def headroom(ab,trial,record,*,fresh):
    timeout,available,node=(600,480,51) if fresh else (300,128,16)
    deadline=time.monotonic()+timeout;rows=[]
    record['prelaunch_headroom' if fresh else 'post_cleanup_headroom']=rows
    while time.monotonic()<deadline:
        h=ab.host();minimum=min(n['free_plus_clean'] for n in h['nodes'].values())
        rows.append({'utc':h['utc'],'memavailable':h['memavailable'],
                     'minimum_node_free_plus_clean':minimum,'cma_guard':h['cma_guard']})
        save(trial/record['receipt_name'],record)
        if h['memavailable']>=available*GIB and minimum>=node*GIB and h['cma_guard']['pass_gate'] is True:
            ab.check_general_headroom(h);return
        time.sleep(10)
    raise TimeoutError('unchanged v7 host/node headroom requirement was not met')

def group_members(pgid):
    live=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':live.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return live

def stop_client(client):
    if client is None:return
    # Also signal the owned group after leader completion, so a timed-out scoring
    # child cannot outlive the client. The group was created by this controller.
    for sig,wait in [(signal.SIGTERM,10),(signal.SIGKILL,5)]:
        try:os.killpg(client.pid,sig)
        except ProcessLookupError:pass
        try:client.wait(timeout=wait)
        except subprocess.TimeoutExpired:continue
        until=time.monotonic()+5
        while time.monotonic()<until:
            if not group_members(client.pid):return
            time.sleep(.1)
    raise RuntimeError('owned client process group did not terminate within the bound')

def check_runtime_configuration(profile,classified):
    expected=profile['server_args'];actual=classified['server_args']
    require(actual==expected,'resolved server args differ: '+str({k:(expected.get(k),actual.get(k)) for k in set(expected)|set(actual) if expected.get(k)!=actual.get(k)}))
    # These existing scheduler overrides are declared before model startup.
    scheduler={k:v for k,v in expected.items() if k!='custom_sigquit_handler'}
    scheduler.update(disable_chunked_prefix_cache=True,pp_max_micro_batch_size=expected['max_running_requests'],disaggregation_decode_retraction_backup='cpu_tensor')
    require(classified['scheduler_args']==scheduler,'scheduler resolved arguments differ: '+str({k:(scheduler.get(k),classified['scheduler_args'].get(k)) for k in set(scheduler)|set(classified['scheduler_args']) if scheduler.get(k)!=classified['scheduler_args'].get(k)}))
    require(expected['cuda_graph_config']['prefill']['backend']=='disabled','prefill graph unexpectedly configured')
    require(profile['environment']['SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS']=='0','tail graph unexpectedly configured')
    require(expected['enable_decoder_swa_bounded_replay'] is False,'approximate bounded decoder replay unexpectedly enabled')
    effective=classified['effective']
    require(effective['max_total_num_tokens']==effective['token_capacity']==1048576,'full KV capacity differs')
    require(effective['token_capacity_swa']>=512,'SWA cannot hold chunk and page')
    require(effective['world_size']==1 and effective['effective_max_running_requests_per_dp']==4,'parallelism differs')
    require(1 in effective['decode_cuda_graph_bs'] and max(effective['decode_cuda_graph_bs'])==4 and min(effective['decode_cuda_graph_bs'])==1, 'four-request CUDA graph capacity differs')
    require(classified['observed_only']['decode_cuda_graph_memory_gb']>0,'decode graph not allocated')


def route3_native_origin_gate(trial, cg, output, scheduler):
    binary=trial/'packages/route3_phase2_l2p0_r1.so'
    expected='51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649'
    require(sha(binary)==expected,'Route3 binary changed')
    proc=Path('/proc')/str(scheduler)
    require(proc.exists(),'scheduler disappeared')
    group=(proc/'cgroup').read_text().strip().split('::')[1].lstrip('/')
    require(Path('/sys/fs/cgroup')/group==cg,'Route3 scheduler outside owned scope')
    maps=[line.split()[-1] for line in (proc/'maps').read_text().splitlines() if 'route3_phase2_l2p0_r1.so' in line]
    require(maps and set(maps)=={str(binary)},'Route3 extension not mapped from pinned candidate')
    receipt=read(trial/'route3-binary-origins'/(str(scheduler)+'.json'))
    require(receipt==dict(pid=scheduler,binary=str(binary),binary_sha256=expected,
                         loader='direct_pinned_binary_no_compiler',policy=0),'Route3 load receipt differs')
    save(output/'ROUTE3_NATIVE_ORIGIN.json',dict(status='PASS',scheduler_pid=scheduler,
         binary_sha256=expected,mapped_path=str(binary),receipt=receipt))

def effective_gate(trial,cg,mods,output):
    output.mkdir(exist_ok=True)
    profile=read(trial/'PROFILE.json');profile['server_args']=read(trial/'PROFILE_RESOLVED.json');info=mods.run_ab.http('/get_server_info')
    save(output/'server_info.json',info)
    classified=split_server_info(info)
    save(output/'SERVER_INFO_SPLIT.json',classified)
    check_runtime_configuration(profile,classified)
    save(trial/'EFFECTIVE_PROFILE.json',profile)
    scheduler=mods.run_ab.runtime(cg,output/'runtime-before.json')
    runtime=read(output/'runtime-before.json')
    scheduler_process=next(p for p in runtime['processes'] if p['pid']==scheduler)
    service=scheduler_process.get('service_thread_affinity', {})
    save(output/'SERVICE_AFFINITY.json',service)
    require(service.get('count',0)>0 and service.get('cpu_mask_counts')=={'7,15,23,39,47,55':service.get('count')}, 'scheduler service threads are not confined to the six reserved service CPUs')
    parents=[p for p in runtime['processes'] if '-m sglang.launch_server ' in p['cmdline']]
    require(len(parents)==1,'missing owned launch parent')
    pid=parents[0]['pid']
    raw=dict(x.split('=',1) for x in (Path('/proc')/str(pid)/'environ').read_bytes().decode().split('\0') if '=' in x)
    wanted=profile['environment']
    require({k:raw.get(k) for k in wanted}==wanted,'effective parent environment differs')
    executable=Path(read(trial/'LAUNCH.json')['argv'][0]).resolve()
    python_pids=[]
    for process in runtime['processes']:
        try:
            if (Path('/proc')/str(process['pid'])/'exe').resolve(strict=True)==executable:
                python_pids.append(process['pid'])
        except FileNotFoundError:continue
    require(scheduler in python_pids,'owned scheduler does not use the selected Python runtime')
    origins=module_origins.gate_ready(trial/'ORIGINS_CONFIG.json',
        expected_sha256=sha(trial/'ORIGINS_CONFIG.json'),owned_python_pids=python_pids,
        scheduler_pid=scheduler,cgroup=cg)
    save(output/'MODULE_ORIGINS.json',origins)
    route3_native_origin_gate(trial,cg,output,scheduler)
    return pid

from settle_before_reclaim_v2 import wait as post_ready_settle, recording_snapshot

def run_scope(args,mods,prepared,record):
    trial=args.trial;ab,life=mods.run_ab,mods.scope_lifecycle
    kind=args.phase;out=trial/('cache-scope' if kind=='cache' else 'model-contract-route3-numerics-r1/server');out.mkdir(parents=True,exist_ok=False)
    unit='codex-sm89-'+trial.name+'-'+kind+'-contract-route3-numerics-r1.scope'
    record.update(unit=unit,status='STARTING',scope_output=str(out))
    cap=64 if kind=='cache' else 416
    payload=P/'tools/launch_route3_numerics_r1.py'
    shell='set -Eeuo pipefail\nsource "$1"\nexec /usr/bin/python3 -B "$2" --trial "$3" --phase "$4"\n'
    command=['systemd-run','--user','--scope','--quiet','--unit='+unit,
        '-p',f'MemoryMax={cap}G','-p','MemorySwapMax=0','/usr/bin/python3','-B',
        str(R/'six-cpu-fea64-guard-r1/scope_exec.py'),str(out),str(cap),'/bin/bash','-c',shell,
        'feature-payload',prepared['setup_environment']['path'],str(payload),str(trial),kind]
    env={k:v for k,v in os.environ.items() if not k.startswith(('KT_','SGLANG_','DSV41_','SM89_','FEATURE_','ACCEPT_'))}
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='' if kind=='cache' else '0')
    if kind=='model':env['ACCEPT_RECLAIM_HELPER']='1'
    proc=client=None;client_log=None;cg=None;ready=None;resident=False
    try:
        with (out/'launcher.log').open('w') as log,(out/'memory.jsonl').open('w') as memory:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                  preexec_fn=ab.policy,start_new_session=True)
            record['launcher_pid']=proc.pid;deadline=time.monotonic()+7200
            while True:
                require(resident or time.monotonic()<deadline,'bounded phase deadline exceeded')
                if cg is None and (out/'SCOPE.json').exists():
                    cg=life.checked_scope_path(read(out/'SCOPE.json')['cgroup'],unit)
                if cg is not None:
                    if cg.exists():
                        sample=ab.snapshot(cg);memory.write(__import__('json').dumps(sample)+'\n');memory.flush()
                        ab.check_memory(sample,phase='LOADING' if ready is None else 'BENCHMARKING',ready_events=ready)
                    elif proc.poll() is None:raise RuntimeError('owned cgroup disappeared')
                if proc.poll() is not None:
                    require(kind=='cache' and proc.returncode==0,'owned payload failed or server exited')
                    record['cache_prepared']=check_cache(trial);record['status']='CACHE_PREPARED';break
                if resident:
                    require(cg is not None and cg.exists(),'resident scope disappeared')
                    if (trial/'RESIDENT_STOP').exists():
                        record['residency']['status']='STOP_REQUESTED';break
                    ab.http('/health',timeout=5)
                    record['residency']['last_health_utc']=ab.now()
                    save(trial/record['receipt_name'],record)
                    time.sleep(30)
                    continue
                if kind=='model' and ready is None and cg is not None:
                    text=(out/'launcher.log').read_text(errors='replace')
                    if 'The server is fired up' in text:
                        try:ab.http('/health',timeout=2)
                        except (urllib.error.URLError,TimeoutError,ConnectionError):
                            time.sleep(2);continue
                        # v7: let post-readiness allocation finish before the frozen settle (read-only wait).
                        record['post_ready_settle']=post_ready_settle(cg,ab,proc,out/'POST_READY_SETTLE.json')
                        require(time.monotonic()<deadline,'startup deadline exceeded during the post-readiness wait')
                        # Resident helper is stopped by stabilization before the
                        # origin PID inventory and ready baseline are captured.
                        settled=mods.cache_stabilization.stabilize(cg,out/'CACHE_STABILIZATION.json',
                            snapshot=recording_snapshot(ab.snapshot,out/'CACHE_SAMPLES.jsonl'),check_memory=ab.check_memory,memory_events=ab.memory_events)
                        launch_pid=effective_gate(trial,cg,mods,out)
                        ab.http('/health',timeout=2);require(proc.poll() is None,'server exited after settling')
                        require(time.monotonic()<deadline,'startup deadline exceeded during stabilization')
                        ready=ab.memory_events(settled);save(out/'memory-before-client.json',settled)
                        record.update(status='CLIENT',ready_events=ready,ready_utc=ab.now())
                        client_log=(out/'client.log').open('w')
                        client=subprocess.Popen(['/usr/bin/python3','-B',str(P/'tools/client_route3_numerics_r1.py'),
                            '--trial',str(trial),'--harness',str(args.harness),'--server-pid',str(launch_pid),'--namespace','contract-route3-numerics-r1'],
                            stdout=client_log,stderr=subprocess.STDOUT,env=env,
                            preexec_fn=lambda:os.sched_setaffinity(0,{55}),start_new_session=True)
                        record['client_pid']=client.pid;deadline=time.monotonic()+17000
                elif client is not None and client.poll() is not None:
                    require(client.returncode==0,'supplemental client failed')
                    result=read(trial/'supplemental-contract-route3-numerics-r1/RESULT.json')
                    require(result['status']=='TOOLS_COMPLETE_PENDING_NUMERICAL_REVIEW','client incomplete')
                    final=ab.snapshot(cg);ab.check_memory(final,phase='BENCHMARKING',ready_events=ready)
                    save(out/'memory-after-client.json',final)
                    effective_gate(trial,cg,mods,out/'after')
                    check_cache(trial)
                    record.update(status='TOOLS_COMPLETE_PENDING_NUMERICAL_REVIEW',
                                  client_receipt_sha256=sha(trial/'supplemental-contract-route3-numerics-r1/RESULT.json'))
                    if not args.keep_loaded:break
                    # Finish and snapshot validation before entering idle residency.
                    # Failures still take the original finally/cleanup path.
                    stop_client(client);client=None
                    client_log.close();client_log=None
                    verify_prepared(trial,allow_private_hint_writes=True)
                    record['validated_sources_unchanged']=True
                    record['private_hint_outputs']=inventory(trial/'hints')
                    record['validation_finished_utc']=ab.now()
                    record['residency']={'status':'ACTIVE','cgroup':str(cg),
                        'launch_pid':launch_pid,'since_utc':ab.now(),
                        'stop_file':str(trial/'RESIDENT_STOP'),
                        'memory_cap_gib':416,'swap_max_bytes':0,
                        'health_interval_seconds':30,'automatic_restart':False}
                    require(not (trial/'CONTRACT_NUMERICAL_TOOLS_COMPLETE_r1.json').exists(),'validation snapshot exists')
                    save(trial/'CONTRACT_NUMERICAL_TOOLS_COMPLETE_r1.json',record)
                    resident=True
                    print('DIAGNOSTIC_COMPLETE_MODEL_RESIDENT',flush=True)
                save(trial/record['receipt_name'],record);time.sleep(2)
    except BaseException as exc:
        record.update(status='FAIL',error=repr(exc));raise
    finally:
        failures=[]
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
        try:stop_client(client)
        except BaseException as exc:failures.append('client: '+repr(exc))
        try:
            if client_log is not None:client_log.close()
        except BaseException as exc:failures.append('client log: '+repr(exc))
        if proc is not None:
            try:
                record['cleanup']=life.stop_scope(unit,proc,cg)
                if resident:record['residency']['status']='STOPPED'
                require(record['cleanup']['empty'] is True,'owned scope not empty')
            except BaseException as exc:failures.append('scope: '+repr(exc))
        try:
            verify_prepared(trial,allow_private_hint_writes=True)
            record['private_hint_outputs']=inventory(trial/'hints')
            record['validated_sources_unchanged']=True
        except BaseException as exc:failures.append('source: '+repr(exc))
        record['finished_utc']=ab.now()
        if failures:record.update(status='FAIL',cleanup_errors=failures)
        save(trial/record['receipt_name'],record)
        if failures:raise RuntimeError('; '.join(failures))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cache','model'],required=True)
    p.add_argument('--keep-loaded',action='store_true');a=p.parse_args()
    a.trial=BASE/'feature-trial-B-contract-route3-numerics-six-r1';a.harness=BASE/'eight-node-ab-tools/retry_v9'
    pins=read(HERE/'HARNESS_PINS.json')
    mods=load_frozen(a.harness,BASE/'eight-node-ab-tools/pr-acceptance-retry-v9.manifest.json',pins['B']['manifest_sha256'],
        a_harness=BASE/'eight-node-ab-tools/retry_v7',a_manifest=BASE/'eight-node-ab-tools/pr-acceptance-retry-v7.manifest.json',a_manifest_sha256=pins['A']['manifest_sha256'])
    sys.path.insert(0,str(R/'six-cpu-fea64-guard-r1'))
    import importlib.util
    import six_cpu_runtime
    spec=importlib.util.spec_from_file_location('six_private_cache',R/'six-cpu-fea64-guard-r1/cache_stabilization.py')
    private_cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(private_cache)
    mods.cache_stabilization=private_cache
    ab=mods.run_ab;ab.policy=six_cpu_runtime.policy;ab.runtime=six_cpu_runtime.runtime
    from fea_reserve import with_fea_reserve
    ab.check_memory=with_fea_reserve(ab.check_memory)
    prepared=verify_prepared(a.trial)
    receipt='CACHE_RUN.json' if a.phase=='cache' else 'RUN-contract-route3-numerics-r1.json'
    require(not (a.trial/receipt).exists(),'phase already ran; preserve evidence')
    if a.phase=='model':
        cache=read(a.trial/'CACHE_RUN.json');require(cache['status']=='CACHE_PREPARED' and cache['cleanup']['empty'],'cache scope incomplete')
        check_cache(a.trial)
    record=dict(status='PREPARING',receipt_name=receipt,phase=a.phase,started_utc=ab.now(),
        prepared_sha256=sha(a.trial/'ROUTE3_NUMERICS_PREPARED_r1.json'),guard_provenance=mods.provenance,scope_cap_gib=64 if a.phase=='cache' else 416,
        numerical_acceptance='REVIEW_REQUIRED',host_reload_gate_unchanged=True,pr6='HELD',cpu_nodes=[0,1,2,4,5,6],shared_memory_nodes=list(range(8)),fea_reserved_gib=64,ready_target_gib=392)
    import fcntl
    with (BASE/'.bench.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def interrupted(*_):raise KeyboardInterrupt('owner requested stop')
        signal.signal(signal.SIGTERM,interrupted)
        try:
            unloaded(mods.scope_lifecycle)
            record['hardware_before']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,power.limit,driver_version','--format=csv,noheader'],text=True).strip()
            require(record['hardware_before'].startswith('GPU-67d9f495-2d3a-1e93-a617-dd11ba547914, 320.00 W'),'GPU or power differs')
            boot=read(Path(prepared['canonical']['A']['campaign'])/'BOOT.json')
            record['boot']=mods.boot_gate.capture(Path(boot['preboot_receipt']),ab.host());save(a.trial/'BOOT.json',record['boot'])
            headroom(ab,a.trial,record,fresh=True)
            run_scope(a,mods,prepared,record)
        except BaseException as exc:
            record.update(status='FAIL',error=repr(exc));raise
        finally:
            if record.get('launcher_pid'):
                failures=[]
                try:unloaded(mods.scope_lifecycle)
                except BaseException as exc:failures.append('GPU/service: '+repr(exc))
                try:headroom(ab,a.trial,record,fresh=False)
                except BaseException as exc:failures.append('headroom: '+repr(exc))
                if failures:record.update(status='FAIL',post_cleanup_errors=failures)
            record['finished_utc']=ab.now();save(a.trial/receipt,record)
    print(record['status'],flush=True)
    require(record['status']!='FAIL','phase or cleanup failed')
if __name__=='__main__':main()
