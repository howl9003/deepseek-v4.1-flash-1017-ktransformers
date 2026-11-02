"""r19 fast-profile identity only. No timing, tolerance waiver, or acceptance promotion."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,urllib.request
from compare_r19_exact_scores_r1 import compare

B=Path('/scratch/aster/bench');P=B/'six-node-performance-r1'
N=B/'feature-trial-B-contract-route3-numerics-six-r1/supplemental-contract-route3-numerics-r1'
F=B/'feature-trial-B-contract-route3-fast-six-r1/supplemental-contract-route3-perf-r1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def selected(reply):
    rows=reply['meta_info']['output_token_logprobs']
    return {'text':reply['text'],'output_ids':reply['output_ids'],'scores':[r[0] for r in rows],'score_ids':[r[1] for r in rows]}
def replay_gate(reference,candidate):
    a,b=selected(reference),selected(candidate)
    numeric=compare(a['scores'],b['scores'])
    return {'text_equal':a['text']==b['text'],'output_ids_equal':a['output_ids']==b['output_ids'],
            'selected_logprob_ids_equal':a['score_ids']==b['score_ids'],'selected_logprobs_exact':numeric['status']=='EXACT','numeric':numeric}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial',required=True,type=Path);ap.add_argument('--harness',required=True,type=Path)
    ap.add_argument('--server-pid',required=True);ap.add_argument('--namespace',required=True);a=ap.parse_args()
    assert os.sched_getaffinity(0)=={55}
    T=a.trial;out=T/('supplemental-'+a.namespace);out.mkdir(exist_ok=False)
    result=dict(status='RUNNING',diagnostic_timing=False,performance_run=False,host_reload_gate='UNCHANGED_NOT_RERUN_IN_THIS_COMPONENT',steps=[])
    save(out/'RESULT.json',result)
    try:
        pins=read(T/'R19_IDENTITY_REFERENCES.json')
        assert len(pins['files'])==4 and {item['path'] for item in pins['files']}=={str(N/name) for name in ('served-nll.json','served-nll-receipt.json')}|{str(F/name) for name in ('repeat-0.json','repeat-1.json')}
        for item in pins['files']:
            assert sha(Path(item['path']))==item['sha256'],item['path']
        profile=read(T/'EFFECTIVE_PROFILE.json');env=profile['environment'];server=profile['server_args']
        assert server['disable_radix_cache'] is True and server['max_running_requests']==1
        assert server['enable_streaming_session'] is False and server['enable_hierarchical_cache'] is False
        assert env['KT_PREFILL_SWA_LEND']=='1' and env.get('KT_WORKER_TIMESTAMPS','0')=='0'
        assert env.get('KT_ROUTING_DUMP','').startswith(str(T/'routing-dump/'))
        assert sha(T/'client-tools/kv_checks.py')=='eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4'
        accepted=read(F/'repeat-0.json')
        assert all(replay_gate(accepted,read(F/'repeat-1.json'))[k] for k in ('text_equal','output_ids_equal','selected_logprob_ids_equal','selected_logprobs_exact'))
        body={'text':'Explain why the sky is blue in one sentence.','sampling_params':{'max_new_tokens':8,'temperature':0},'return_logprob':True,'top_logprobs_num':8}
        replies=[]
        for i in range(2):
            request=urllib.request.Request('http://127.0.0.1:18209/generate',json.dumps(body).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=180) as response:r=json.load(response)
            save(out/f'replay-{i}.json',r);replies.append(r)
            gate=replay_gate(accepted,r);result['steps'].append({'name':f'replay-{i}-vs-accepted','gate':gate});save(out/'RESULT.json',result)
            assert all(gate[k] for k in ('text_equal','output_ids_equal','selected_logprob_ids_equal','selected_logprobs_exact')),'replay differs from accepted fast fixture'
        repeat=replay_gate(replies[0],replies[1]);result['repeat']=repeat
        assert all(repeat[k] for k in ('text_equal','output_ids_equal','selected_logprob_ids_equal','selected_logprobs_exact'))
        prepared=read(T/'PREPARED.json');root=Path(prepared['canonical']['B']['root'])
        nll=root/'tools/nll.py';reference=read(N/'served-nll-receipt.json')
        assert sha(nll)==reference['upstream_sha256']=='c5b29236724b2f30579cde5269e3a08618f698b579b03b985e87000061b3a9d5'
        assert sha(N/'served-nll.json')==reference['scores_sha256']=='ae50236af65eb21146ce8ae4aa19cc03883264401243884c728d4f9041640d6b'
        command=[str(root/'.venv/bin/python'),str(a.harness/'pr_acceptance/run_nll_recorded.py'),'--script',str(nll),'--checkpoint',server['model_path'],'--output',str(out/'served-nll.json'),'--receipt',str(out/'served-nll-receipt.json')]
        child=dict(os.environ,URL='http://127.0.0.1:18209',SGLANG_PORT='18209',PYTHONDONTWRITEBYTECODE='1')
        for key in ('PYTHONPATH','FEATURE_ORIGINS_CONFIG','FEATURE_ORIGINS_CONFIG_SHA256'):child.pop(key,None)
        with (out/'nll.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=child,timeout=7200)
        result['steps'].append({'name':'numerical-set','command':command,'exit_code':r.returncode});save(out/'RESULT.json',result)
        assert r.returncode==0,'NLL recording failed'
        rec=read(out/'served-nll-receipt.json')
        assert rec['status']=='PASS' and rec['input_ids_sha256']==reference['input_ids_sha256']
        assert rec['pieces']==24 and rec['scored_positions']==11496 and rec['scores_sha256']==sha(out/'served-nll.json')
        exact=compare(read(N/'served-nll.json'),read(out/'served-nll.json'));save(out/'EXACT_NUMERICAL_SET.json',exact)
        result['numerical_set']=exact;save(out/'RESULT.json',result)
        assert exact['status']=='EXACT' and exact['compared_scores']==11496,'bit-neutral numerical identity failed; do not time or waive'
        result.update(status='R19_BITNEUTRAL_IDENTITY_PASS_TIMING_NOT_RUN',reference_manifest_sha256=sha(T/'R19_IDENTITY_REFERENCES.json'),
                      effective_profile_sha256=sha(T/'EFFECTIVE_PROFILE.json'),source_prepared_sha256=sha(T/'PREPARED.json'),
                      next='Collect routing dump and unload; qualify exact candidate inventories before a fresh matched timing namespace. No timing in this process.')
    except BaseException as exc:
        result.update(status='FAIL',error=repr(exc));raise
    finally:save(out/'RESULT.json',result)

if __name__=='__main__':main()
