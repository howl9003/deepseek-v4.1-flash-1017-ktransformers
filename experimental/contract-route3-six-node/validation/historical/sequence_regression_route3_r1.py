"""New arithmetic cold context; exact rotation/retained regression and frozen acceptance follow."""
from pathlib import Path
import argparse,json,signal,sys
ap=argparse.ArgumentParser();ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--server-pid',type=int,required=True);ap.add_argument('--namespace',required=True)
a=ap.parse_args();T=a.trial;sys.path.insert(0,str(T/'client-tools'))
from feature_transport import Context,source_inventory,validate_profile,process_provenance,digest
from feature_client import body,summary,compare
read=lambda p:json.loads(p.read_text())
profile=read(T/'EFFECTIVE_PROFILE.json');validate_profile(profile,model='default',context=profile['server_args']['context_length'])
assert profile['environment']['KT_PREFILL_LEND_EXPERT_SCALES']=='1'
assert profile['environment']['SGLANG_DSV41_FIXED_LM_HEAD']=='1'
assert all(profile['environment'].get(k)=='1' for k in ('SGLANG_FP8_ROUTE3_CUDA','KT_CONTRACT_V2_GPU_DP4A','KT_CONTRACT_V2_QUANT_AVX2','KT_EXPERT_ARENA_REBALANCE_NODE0'))
assert profile['environment']['SGLANG_FP8_FIXED_CUDA_CORE']=='0'
assert not any('OBSERVER' in k for k in profile['environment'])
assert 'FEATURE_FAST_REPLAY_OBSERVER_DIR' not in profile['environment']
history=read(T.parent/'feature-trial-B-contract-v2-model-r1/append-probe-r1/RESULT.json')['turns'][2]['full_input']
assert len(history)==538
out=T/('sequence-regression-'+a.namespace);out.mkdir(exist_ok=False)
ctx=Context(host='127.0.0.1',port=18209,model='default',output=out,deadline_s=300)
report=dict(status='RUNNING',cases=[],history_sha256=digest(history),performance_valid=False);sid=None
def save():(out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
def exact(x):return x['output_ids_equal'] is True and x['max_selected_logprob_delta']==0
def cold(label):
    ctx.flush();value=summary(ctx.request(label,'/generate',body(history)))
    assert value['cached_tokens']==0 and value['prompt_tokens']==538
    report['cases'].append(dict(label=label,result=value));save();return value
def expired(*_):raise TimeoutError('bounded sequence regression')
signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGALRM,expired);signal.alarm(300);save()
try:
    report['source_before']=source_inventory(T/'recipe/third_party/sglang',profile['source_sha256'])
    report['process_before']=process_provenance(a.server_pid,T/'recipe/third_party/sglang',18209,profile['environment'])
    baseline=cold('cold-before-rotation')
    expected=read(T.parent/'feature-trial-B-contract-fast-trace-six-r1/supplemental-contract-fast-trace-r1/RESULT.json')['cases'][0]['result']
    report['vs_accepted_initial']=compare(baseline,expected);report['cold_context']='route3_default_fixed_splitk_dense_with_accepted_k3_moe';save()
    inputs=read(T/'INPUTS.json');ctx.flush();payload=body(inputs['512'],tokens=128)
    payload.update(stream=True,return_logprob=False)
    stream=ctx.start_stream('rotate-resident-512x128',payload)
    try:
        while True:
            snap=stream.snapshot()
            assert not snap['errors'],snap['errors']
            if snap['finished']:break
            ctx.wait(.05)
    finally:stream.close()
    snap=stream.snapshot();assert snap['status']==200 and snap['chunks'] and not snap['errors']
    final=snap['chunks'][-1]['value'];meta=final['meta_info']
    assert meta['prompt_tokens']==512 and meta['cached_tokens']==0 and meta['completion_tokens']==128 and len(final['output_ids'])==128
    after=cold('cold-after-rotation');report['cold_before_after']=compare(baseline,after);save();assert exact(report['cold_before_after'])
    ctx.flush();sid=ctx.request('retained-open','/open_session',dict(capacity_of_str_len=100000,streaming=True,timeout=180))['value']
    prefix=ctx.request('retained-prefix','/generate',body(history[:512],sid,tokens=0));assert prefix['status']==200 and prefix['value']['output_ids']==[]
    retained=summary(ctx.request('retained-append','/generate',body(history[512:],sid)))
    assert retained['cached_tokens']==512 and retained['prompt_tokens']==538
    report['retained']=retained;report['cold_retained']=compare(after,retained);save();assert exact(report['cold_retained'])
    ctx.request('retained-close','/close_session',dict(session_id=sid));sid=None
    assert source_inventory(T/'recipe/third_party/sglang',profile['source_sha256'])==report['source_before']
    assert process_provenance(a.server_pid,T/'recipe/third_party/sglang',18209,profile['environment'])==report['process_before']
    report['status']='PASS_CAPTURED_ROTATION_REGRESSION_FULL_ACCEPTANCE_STILL_REQUIRED'
except BaseException as exc:
    report.update(status='FAIL',error=repr(exc));raise
finally:
    ctx.begin_cleanup();errors=[]
    for rid in list(ctx.owned):
        try:ctx.abort(rid)
        except BaseException as exc:errors.append(repr(exc))
    if sid:
        try:ctx.request('cleanup-close','/close_session',dict(session_id=sid))
        except BaseException as exc:errors.append(repr(exc))
    ctx.close_connections();signal.alarm(0);report['cleanup_errors']=errors
    if errors:report['status']='FAIL'
    save()
assert report['status']=='PASS_CAPTURED_ROTATION_REGRESSION_FULL_ACCEPTANCE_STILL_REQUIRED'
print(report['status'])
