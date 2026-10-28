"""Read-only review of the frozen historical workload; no numerical promotion."""
from pathlib import Path
import datetime,hashlib,json,re,statistics,subprocess
B=Path('/scratch/aster/bench');P=B/'six-node-performance-r1'
T=B/'feature-trial-B-contract-route3-fast-six-r1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=read(T/'RUN-contract-route3-perf-r1.json');clientpath=T/'supplemental-contract-route3-perf-r1/RESULT.json';client=read(clientpath)
assert run.get('finished_utc'), 'Do not review a live or unfinished controller'
unit=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show','codex-route3-performance-r1.service','-p','MainPID','-p','Result','-p','InvocationID'],text=True).splitlines() if '=' in x)
assert unit['MainPID']=='0', 'Benchmark owner still active'
prep=read(T/'ROUTE3_PERF_PREPARED_r1.json');accepted=read(P/'ROUTE3_NUMERICAL_REVIEW_r1.json')
baseline=read(B/'feature-trial-B-contract-route3-numerics-six-r1/ROUTE3_NUMERICS_PREPARED_r1.json')
audit=read(P/'MATCHED_ROUTE3_PERFORMANCE_PROFILE_r1.json')
server=T/'model-contract-route3-perf-r1/server';before=read(server/'memory-before-client.json');after=read(server/'memory-after-client.json')
events=lambda s:{k:int(v) for k,v in (line.split() for line in s.splitlines())}
e0=events(before['memory.events']);e1=events(after['memory.events'])
hist=(clientpath.parent/'historical.log').read_text();breadth=(clientpath.parent/'breadth.log').read_text()
rates=[float(x) for x in re.findall(r'decode p\d+:.*= ([0-9.]+) tok/s',hist)]
prefill=[float(x) for x in re.findall(r'prefill, .*?: ([0-9.]+) s =',hist)]
broad=[dict(workload=name,tok_s=float(rate)) for name,rate in re.findall(r'decode, (.*?):.*?= ([0-9.]+) tok/s',breadth)]
gate='eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
gates=dict(
 exact_owner=unit['InvocationID']==read(P/'ROUTE3_PERFORMANCE_OWNER_r1.json')['invocation'],owner_success=unit['Result']=='success',
 run_complete=run['status']=='PERFORMANCE_COMPLETE_NOT_PR6_ACCEPTANCE',
 client_complete=client['status']=='PERFORMANCE_COMPLETE_NOT_PR6_ACCEPTANCE',
 authenticated_client=sha(clientpath)==run.get('client_receipt_sha256'),
 authenticated_manifest=sha(T/'ROUTE3_PERF_PREPARED_r1.json')==run['prepared_sha256'],
 clean_scope=run.get('cleanup',{}).get('empty') is True,
 no_outer_cleanup_errors=not run.get('cleanup_errors') and not run.get('post_cleanup_errors'),
 validated_source_unchanged=run.get('validated_sources_unchanged') is True,
 numerical28_pass=accepted['status']=='CHECKS_PASS_HELD_FOR_REVIEW' and len(accepted['gates'])==28 and all(accepted['gates'].values()),
 same_source_as_numerical=prep['source_after']==baseline['source_after'],
 same_native_as_numerical=prep['native_after']==baseline['native_after'],
 frozen_host_reload=sha(T/'client-tools/kv_checks.py')==gate==prep['host_reload_gate_sha256'],
 historical_client_hash=sha(P/'source/bench.py')==client['bench_sha256']=='1c361c2010d7f69a8f1a93e33f26e7622bb684bd43a2de51b29b63a00b063f40',
 four_japanese_two_prefill_six_breadth=len(rates)==4 and len(prefill)==2 and len(broad)==6,
 both_clients_success=len(client['cases'])==2 and all(c['returncode']==0 for c in client['cases']),
 reported_rates_match=rates==client['japanese_tokens_per_second'] and statistics.median(rates)==client['japanese_median_tokens_per_second'],
 model_cap_and_reserve=run['scope_cap_gib']==416 and run['fea_reserved_gib']==64,
 six_nodes=run['cpu_nodes']==[0,1,2,4,5,6],
 no_swap=int(before['memory.swap.current'])==int(after['memory.swap.current'])==0,
 no_client_memory_events=e0==e1==run['ready_events'],
 final_host_reserve=after['host']['memavailable']>=64*2**30,
 recorded_historical_env_matches=audit['unchanged_recorded_env_count']==90)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 status='PERFORMANCE_CHECKS_PASS_HELD' if all(gates.values()) else 'PERFORMANCE_REVIEW_FAILED_HELD',gates=gates,
 failed=[k for k,v in gates.items() if not v],unit=unit,
 evidence={str(p):sha(p) for p in (T/'RUN-contract-route3-perf-r1.json',clientpath,T/'ROUTE3_PERF_PREPARED_r1.json',P/'ROUTE3_NUMERICAL_REVIEW_r1.json',P/'MATCHED_ROUTE3_PERFORMANCE_PROFILE_r1.json',server/'memory-before-client.json',server/'memory-after-client.json',clientpath.parent/'historical.log',clientpath.parent/'breadth.log')},
 breadth_median_tok_s=statistics.median(row['tok_s'] for row in broad),japanese_tok_s=rates,japanese_median_tok_s=statistics.median(rates),prefill35000_seconds=prefill,breadth=broad,
 post_timing_repeat=client['post_timing_repeat'],
 memory=dict(final_charge_gib=int(after['memory.current'])/2**30,whole_run_peak_gib=int(after['memory.peak'])/2**30,
 final_available_gib=after['host']['memavailable']/2**30,swap_bytes=int(after['memory.swap.current']),ready_events=e0,after_events=e1),
 allocation_warning='Read this run launcher.log for warnings; no claim of zero allocation pressure.',
 limitations=['This performance profile differs explicitly from numerical acceptance: historical hot/stream/cache/concurrency settings. The separate28-gate numerical run is the acceptance evidence.',
 'Six CPU nodes42workers vs historical8nodes56workers; math selectors and approximate replay exclusion are explicit additional differences. Historical unrecorded environment and power/unlock state remain unknown.',
 'Single completed campaign; no statistical cross-run speedup claim. Post-timing repeated8token probe is descriptive, not a substitute for the full gates.'],
 pr6='HELD_NO_PUSH_NO_PUBLISH')
for suffix in ('','after'):
    native=read(server/suffix/'ROUTE3_NATIVE_ORIGIN.json')
    assert native['status']=='PASS' and native['binary_sha256']=='51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649'
result['route3_native_origins_verified']=True
target=P/'ROUTE3_PERFORMANCE_REVIEW_r1.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','failed','breadth_median_tok_s','breadth','japanese_median_tok_s','prefill35000_seconds','memory')}))
assert all(gates.values()),result['failed']
