# Phase G G1v arm 0 launch (Claude, 30 Oct 2026): launch_r19_C1_static_on_identity_r2.py with the arm-0 manifest and
# profile, and PATHA_/ROUTE3_ added to the scrubbed ambient prefixes (the plan dictionary sets them).
"""Only execute the pinned isolated performance payload inside the owner scope."""
from pathlib import Path
import argparse,json,os,subprocess,sys,hashlib
P=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--trial',type=Path,required=True);ap.add_argument('--phase',choices=['model'],required=True);a=ap.parse_args()
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prepared=read(a.trial/'PHASE_G_G1V_ADADK4_PREPARED_r1.json');assert sha(a.trial/'LAUNCH.json')==prepared['launch_sha256']
plan=read(a.trial/'LAUNCH.json');env={k:v for k,v in os.environ.items() if not k.startswith(('KT_','SGLANG_','DSV41_','SM89_','FEATURE_','PATHA_','ROUTE3_'))}
for k,v in plan['environment'].items():
 if v is None:env.pop(k,None)
 else:env[k]=v
subprocess.run([plan['argv'][0],'-B',str(P/'tools/profile_phase_g_g1v_adadk4_r1.py'),str(a.trial)],env=env,check=True,timeout=300)
os.chdir(a.trial);os.execve(plan['argv'][0],plan['argv'],env)
