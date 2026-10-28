"""Full head candidate review; preserve frozen checks and new-arithmetic scope."""
from pathlib import Path
import hashlib,json,math,statistics,subprocess,sys
B=Path('/scratch/aster/bench');P=B/'six-node-performance-r1';R=B/'fp32-subtotals-model-r1';T=B/'feature-trial-B-contract-route3-numerics-six-r1';ns='contract-route3-numerics-r1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=read(T/('RUN-'+ns+'.json'));assert run.get('finished_utc')
target=P/'ROUTE3_NUMERICAL_REVIEW_r1.json';assert not target.exists(),'preserve existing review'
subprocess.run([sys.executable,'-B',str(P/'tools/review_fast_numerical_run.py'),'--trial',str(T),'--namespace',ns,'--output',str(target)],check=True,stdout=subprocess.DEVNULL)
d=read(target);g=d['gates'];d['frozen_base_gates']=dict(g)
exact=lambda c:c.get('output_ids_equal') is True and c.get('max_selected_logprob_delta')==0
f=T/('sequence-regression-'+ns)/'RESULT.json';s=read(f) if f.exists() else {}
g['rotation_regression_complete']=s.get('status')=='PASS_CAPTURED_ROTATION_REGRESSION_FULL_ACCEPTANCE_STILL_REQUIRED' and not s.get('cleanup_errors')
g['rotation_preserves_new_cold_exact']=exact(s.get('cold_before_after',{}));g['rotation_retained_exact']=exact(s.get('cold_retained',{}))
g['new_cold_context_declared']=s.get('cold_context')=='route3_default_fixed_splitk_dense_with_accepted_k3_moe'
env=read(T/'EFFECTIVE_PROFILE.json')['environment']
g['clean_opt_in_candidate']=env.get('SGLANG_DSV41_FIXED_LM_HEAD')=='1' and env.get('KT_PREFILL_LEND_EXPERT_SCALES')=='1' and not any('OBSERVER' in k for k in env)
g['selected_speed_options_enabled']=all(env.get(k)=='1' for k in ('SGLANG_FP8_ROUTE3_CUDA','KT_CONTRACT_V2_GPU_DP4A','KT_CONTRACT_V2_QUANT_AVX2')) and all(env.get(k)=='0' for k in ('SGLANG_FP8_FIXED_CUDA_CORE','SGLANG_FP8_FIXED_DECODE_TRANSPOSE','SGLANG_FP8_FIXED_DECODE_GEOMETRY','ROUTE3_L2_POLICY'))
manifest=T/'ROUTE3_NUMERICS_PREPARED_r1.json';prep=read(manifest)
g['prepared_identity']=sha(manifest)==run.get('prepared_sha256')==read(P/'ROUTE3_NUMERICS_RUNTIME_r1.json')['manifest_sha256']
g['candidate_sources_identified']=all(sha(T/'recipe/third_party/sglang'/name)==prep['source_after'][name] for name in prep['kernel_revision']['source_changes'])
g['placement_option_enabled']=env.get('KT_EXPERT_ARENA_REBALANCE_NODE0')=='1'
g['resource_contract_retained']=run.get('scope_cap_gib')==416 and run.get('fea_reserved_gib')==64 and run.get('cpu_nodes')==[0,1,2,4,5,6]
candidate=T/('supplemental-'+ns)/'served-nll.json';accepted=B/'feature-trial-B-contract-kernel-numerics-six-r3/supplemental-contract-kernel-numerics-r3/served-nll.json'
g['accepted_contract_nll_inputs_authenticated']=False;g['paired_piece_nll_increase_within_one_se']=False
if candidate.exists():
    cr=read(candidate.with_name('served-nll-receipt.json'));ar=read(accepted.with_name('served-nll-receipt.json'))
    g['accepted_contract_nll_inputs_authenticated']=(cr['status']==ar['status']=='PASS' and cr['input_ids_sha256']==ar['input_ids_sha256'] and cr['scores_sha256']==sha(candidate) and ar['scores_sha256']==sha(accepted)=='142cbd40c85ba50e878d4074f667d3ed45448e387e6589b45e835fc34abb3153')
    c,a=read(candidate),read(accepted);assert len(c)==len(a)==24 and all(len(x)==479 for x in c+a)
    diffs=[av-cv for aa,cc in zip(a,c) for av,cv in zip(aa,cc)];assert all(math.isfinite(x) for x in diffs)
    pieces=[statistics.mean(av-cv for av,cv in zip(aa,cc)) for aa,cc in zip(a,c)]
    mean=statistics.mean(pieces);se=statistics.stdev(pieces)/math.sqrt(len(pieces))
    g['paired_piece_nll_increase_within_one_se']=g['accepted_contract_nll_inputs_authenticated'] and mean<=se
    d['accepted_contract_comparison']=dict(positions=len(diffs),piece_means=pieces,mean_nll_increase=mean,paired_piece_standard_error=se,criterion='positive degradation no greater than one paired piece SE; no epsilon or post-result multiplier',different_positions=sum(x!=0 for x in diffs),max_absolute_logprob_delta=max(map(abs,diffs)),mean_absolute_logprob_delta=statistics.mean(map(abs,diffs)),candidate_sha256=sha(candidate),accepted_sha256=sha(accepted))
server=T/('model-'+ns)/'server'
g['route3_native_origins_before_after']=all(read(server/suffix/'ROUTE3_NATIVE_ORIGIN.json')['status']=='PASS' and read(server/suffix/'ROUTE3_NATIVE_ORIGIN.json')['binary_sha256']=='51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649' for suffix in ('','after'))
g['route3_native_binary_unchanged']=sha(T/'packages/route3_phase2_l2p0_r1.so')=='51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649'
d.update(rotation_regression=s,pr6='HELD_NO_PUBLISH',unpassed_checks=[k for k,v in g.items() if not v],review_script_sha256=sha(Path(__file__)))
d['status']='CHECKS_PASS_HELD_FOR_REVIEW' if all(g.values()) else 'CHECKS_INCOMPLETE_OR_FAILED_HELD'
d['limitations'].append('Route3 fixed split-K dense arithmetic has its own cold context relative to K3 CUDA-core dense; old r4 lifecycle-only byte-identity assertions are not reused. All frozen gates and quality threshold remain unchanged; Fable paired-piece NLL comparison is additional. This run is not the historical fast performance profile.')
target.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps({k:d.get(k) for k in ('status','unpassed_checks','reference_nll','accepted_contract_comparison')}))
