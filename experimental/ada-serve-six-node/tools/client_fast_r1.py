"""Exact historical performance protocol, followed by a separately labelled numerical probe."""
from pathlib import Path
import argparse,hashlib,json,math,os,re,statistics,subprocess,time,urllib.request
P=Path(__file__).resolve().parent.parent
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--harness');ap.add_argument('--server-pid');ap.add_argument('--namespace',default='perf-r1');a=ap.parse_args()
 assert os.sched_getaffinity(0)=={55}
 out=a.trial/('supplemental-'+a.namespace);out.mkdir(exist_ok=False)
 bench=P/'source/bench.py';assert hashlib.sha256(bench.read_bytes()).hexdigest()=='1c361c2010d7f69a8f1a93e33f26e7622bb684bd43a2de51b29b63a00b063f40'
 result={'status':'RUNNING','profile':'Historical fast, six nodes. Not PR6 numerical acceptance.','bench_sha256':hashlib.sha256(bench.read_bytes()).hexdigest(),'cases':[]}
 save(out/'RESULT.json',result)
 try:
  for name,flags in [('historical',[]),('breadth',['--other-text'])]:
   cmd=['/usr/bin/python3','-B',str(bench),'--url','http://127.0.0.1:18209',*flags]
   start=time.monotonic()
   with (out/(name+'.log')).open('w') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
   result['cases'].append(dict(name=name,command=cmd,returncode=run.returncode,seconds=time.monotonic()-start));save(out/'RESULT.json',result)
   assert run.returncode==0,name
  text=(out/'historical.log').read_text();breadth=(out/'breadth.log').read_text()
  rates=[float(v) for v in re.findall(r'decode p\d+:.*= ([0-9.]+) tok/s',text)]
  assert len(rates)==4 and len(re.findall(r'prefill,',text))==2
  assert len(re.findall(r'decode,',breadth))==6
  result['japanese_tokens_per_second']=rates;result['japanese_median_tokens_per_second']=statistics.median(rates)
  # Preserve the historical timing protocol: no extra requests or flush before it.
  # Dynamic expert placement may change rounding; this probe records it without calling it acceptance.
  body={'text':'Explain why the sky is blue in one sentence.','sampling_params':{'max_new_tokens':8,'temperature':0},'return_logprob':True,'top_logprobs_num':8}
  replies=[]
  for i in range(2):
   req=urllib.request.Request('http://127.0.0.1:18209/generate',json.dumps(body).encode(),{'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=180) as response:r=json.load(response)
   save(out/('repeat-'+str(i)+'.json'),r);replies.append(r)
   assert all(math.isfinite(row[0]) for row in r['meta_info']['output_token_logprobs'])
  result['post_timing_repeat']={'text_equal':replies[0].get('text')==replies[1].get('text'),'selected_logprobs_exact':replies[0]['meta_info']['output_token_logprobs']==replies[1]['meta_info']['output_token_logprobs'],'full_acceptance':False}
  result['status']='PERFORMANCE_COMPLETE_NOT_PR6_ACCEPTANCE';save(out/'RESULT.json',result)
 except BaseException as exc:
  result.update(status='FAIL',error=repr(exc));save(out/'RESULT.json',result);raise
if __name__=='__main__':main()
