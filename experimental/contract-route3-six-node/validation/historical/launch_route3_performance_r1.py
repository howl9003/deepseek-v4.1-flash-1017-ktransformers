"""Only execute the pinned isolated performance payload inside the owner scope."""
from pathlib import Path
import argparse,json,os,subprocess,sys,hashlib
P=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--phase',choices=['model'],required=True);a=ap.parse_args()
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prepared=read(a.trial/'ROUTE3_PERF_PREPARED_r1.json');assert sha(a.trial/'LAUNCH.json')==prepared['launch_sha256']
plan=read(a.trial/'LAUNCH.json');env={k:v for k,v in os.environ.items() if not k.startswith(('KT_','SGLANG_','DSV41_','SM89_','FEATURE_'))}
for k,v in plan['environment'].items():
 if v is None:env.pop(k,None)
 else:env[k]=v
subprocess.run([plan['argv'][0],'-B',str(P/'tools/profile_route3_performance_r1.py'),str(a.trial)],env=env,check=True,timeout=180)
os.chdir(a.trial);os.execve(plan['argv'][0],plan['argv'],env)
