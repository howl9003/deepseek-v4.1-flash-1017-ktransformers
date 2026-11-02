"""Separate full-model timing, reference scoring and unchanged feature checks.

v2 (Claude, 26 Oct 2026): --namespace NAME writes supplemental-NAME/, benchmark-NAME/ and features-NAME/ instead of the
fixed supplemental/, benchmark/ and features/ that RUN-r4 left behind (RUN-r6 died on that collision). Steps unchanged."""
import argparse,os,subprocess,sys
from pathlib import Path
R=Path('/scratch/aster/bench/fp32-subtotals-model-r1');sys.path.insert(0,str(R/'workflow'))
from trial_common import read,save,sha,require
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--harness',type=Path,required=True);p.add_argument('--server-pid',required=True);p.add_argument('--namespace',required=True);a=p.parse_args()
T=a.trial;out=T/('supplemental-'+a.namespace);out.mkdir(exist_ok=False)
prepared=read(T/'PREPARED.json');profile=read(T/'EFFECTIVE_PROFILE.json')
root=Path(prepared['canonical']['B']['root']);baseline=Path(prepared['canonical']['A']['campaign'])
python=str(root/'.venv/bin/python');package=a.harness/'pr_acceptance'
reference=read(baseline/'reference/REFERENCE.json')
require(reference['status']=='PASS' and reference['scores_sha256']==sha(baseline/'reference/ref.json'),'reference changed')
commands=[
    ('precheck',['/usr/bin/python3','-B',str(R/'replay-audit-r1/precheck_clean_model_v2.py'),'--trial',str(T),'--server-pid',a.server_pid,'--namespace',a.namespace],960),
    ('benchmark',['/usr/bin/python3','-B',str(R/'benchmark_v2.py'),'--trial',str(T),'--out','benchmark-'+a.namespace],7200),
    ('nll',[python,str(package/'run_nll_recorded.py'),'--script',str(root/'tools/nll.py'),'--checkpoint',profile['server_args']['model_path'],'--output',str(out/'served-nll.json'),'--receipt',str(out/'served-nll-receipt.json')],7200),
    ('compare',[python,str(root/'tools/nll.py'),'compare',str(baseline/'reference/ref.json'),str(out/'served-nll.json')],120),
    ('quality',[python,str(package/'quality_tools/quality_probe.py'),'--checkpoint',profile['server_args']['model_path'],'--arm','B','--base-url','http://127.0.0.1:18209','--output',str(out/'quality')],7200),
    ('features',['/usr/bin/python3','-B',str(T/'client-tools/feature_client.py'),'--host','127.0.0.1','--port','18209','--model','default','--context',str(profile['server_args']['context_length']),'--profile',str(T/'EFFECTIVE_PROFILE.json'),'--source-root',str(T/'recipe/third_party/sglang'),'--server-pid',a.server_pid,'--output',str(T/('features-'+a.namespace)),'--deadline','1800','--exclusive-caller-owned-server','--execute'],1875)]
env=dict(os.environ,URL='http://127.0.0.1:18209',SGLANG_PORT='18209',PYTHONDONTWRITEBYTECODE='1')
for key in ('PYTHONPATH','FEATURE_ORIGINS_CONFIG','FEATURE_ORIGINS_CONFIG_SHA256'):env.pop(key,None)
record=dict(status='RUNNING',steps=[],numerical_acceptance='REVIEW_REQUIRED',pr6='HELD',timing_separate=True)
save(out/'RESULT.json',record)
try:
    for name,argv,timeout in commands:
        row=dict(name=name,argv=argv,timeout_s=timeout);record['steps'].append(row);save(out/'RESULT.json',record)
        with (out/(name+'.log')).open('w') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=timeout)
        row['exit_code']=result.returncode;save(out/'RESULT.json',record)
        require(result.returncode==0,name+' failed')
        if name=='nll':
            receipt=read(out/'served-nll-receipt.json')
            require(receipt['status']=='PASS' and receipt['input_ids_sha256']==reference['input_ids_sha256'],'NLL inputs differ')
            require(receipt['scores_sha256']==sha(out/'served-nll.json'),'NLL score bytes differ')
        elif name=='quality':require(read(out/'quality/report.json')['status']=='PASS','quality failed')
        elif name=='features':require(read(T/('features-'+a.namespace)/'report.json')['status']=='MEASURED','feature protocol incomplete')
        row['status']='RECORDED';save(out/'RESULT.json',record)
    record['status']='TOOLS_COMPLETE_PENDING_NUMERICAL_REVIEW'
except BaseException as exc:
    record.update(status='FAIL',error=repr(exc));raise
finally:save(out/'RESULT.json',record)
