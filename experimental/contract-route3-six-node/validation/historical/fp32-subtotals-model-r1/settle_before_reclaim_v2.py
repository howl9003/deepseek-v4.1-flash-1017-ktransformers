"""Stricter read-only startup settling; the frozen reclaim guard is unchanged."""
import json,time
from pathlib import Path

def stable_window(rows,now,*,start,minimum_seconds=180,window=120,tolerance=64*1024**2):
 if now-start<minimum_seconds:return False
 earlier=[row for row in rows if now-row['monotonic']>=window]
 if not earlier:return False
 first=earlier[-1]['monotonic']
 values=[row['anon'] for row in rows if row['monotonic']>=first]
 return max(values)-min(values)<tolerance

def processes(cgroup):
 rows=[]
 for raw in (Path(cgroup)/'cgroup.procs').read_text().split():
  p=Path('/proc')/raw
  try:
   status=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
   rows.append({'pid':int(raw),'name':status.get('Name','').strip(),'ppid':status.get('PPid','').strip(),
    'rss':status.get('VmRSS','').strip(),'anon':status.get('RssAnon','').strip(),
    'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')[:2048]})
  except (FileNotFoundError,ProcessLookupError):pass
 return rows

def wait(cg,ab,proc,receipt_path,*,window=120,tolerance=64*1024**2,timeout=900,interval=5):
 rows=[];start=time.monotonic();path=Path(receipt_path)
 receipt={'status':'WAITING','window_seconds':window,'minimum_wait_seconds':180,'tolerance_bytes':tolerance,
  'timeout_seconds':timeout,'interval_seconds':interval,'samples':rows,'semantics':'Absolute range of anonymous memory, not net growth. Read-only before unchanged reclaim/host gates.'}
 def save():path.write_text(json.dumps(receipt,indent=2)+'\n')
 while True:
  if proc.poll() is not None:raise RuntimeError('server exited during startup settling')
  sample=ab.snapshot(cg);ab.check_memory(sample,phase='LOADING')
  stat=dict(line.split() for line in sample['memory.stat'].splitlines());now=time.monotonic()
  rows.append({'utc':sample['host']['utc'],'monotonic':now,'memory_current':int(sample['memory.current']),
   'anon':int(stat['anon']),'shmem':int(stat['shmem']),'file':int(stat['file']),'processes':processes(cg)})
  save()
  if stable_window(rows,now,start=start,window=window,tolerance=tolerance):
   receipt.update(status='SETTLED',elapsed_seconds=now-start);save();return receipt
  if now-start>timeout:
   receipt.update(status='FAILED',elapsed_seconds=now-start);save();raise RuntimeError('startup anonymous-memory range did not settle')
  time.sleep(interval)

def recording_snapshot(snapshot,path):
 """Persist every sample before the existing guard evaluates it, including failure."""
 path=Path(path)
 assert not path.exists()
 def record(cg):
  sample=snapshot(cg)
  with path.open('a') as f:f.write(json.dumps({'snapshot':sample,'processes':processes(cg)},separators=(',',':'))+'\n')
  return sample
 return record
