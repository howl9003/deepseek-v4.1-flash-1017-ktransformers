"""Strict serialized-score gate for r19 bit-neutral candidates. No tolerance/NLL waiver."""
from pathlib import Path
import argparse,hashlib,json,math,struct

def compare(reference,candidate):
    result={'status':'EXACT','compared_scores':0,'first_difference':None}
    def walk(a,b,path):
        if isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b):
                return {'index':path,'reason':'shape','reference_length':len(a),'candidate_length':len(b) if isinstance(b,list) else None}
            for i,(x,y) in enumerate(zip(a,b)):
                diff=walk(x,y,path+[i])
                if diff:return diff
            return None
        # Compare the exact JSON numeric values as binary64; do not round them
        # back to float32 or allow a decimal tolerance that could conceal drift.
        if isinstance(a,bool) or isinstance(b,bool) or not isinstance(a,(int,float)) or not isinstance(b,(int,float)):
            return {'index':path,'reason':'invalid_score_type'}
        if not math.isfinite(a) or not math.isfinite(b):return {'index':path,'reason':'nonfinite'}
        aa,bb=struct.pack('>d',float(a)),struct.pack('>d',float(b))
        result['compared_scores']+=1
        if aa!=bb:return {'index':path,'reason':'score_bits','reference':a,'candidate':b,'reference_bits':aa.hex(),'candidate_bits':bb.hex()}
        return None
    result['first_difference']=walk(reference,candidate,[])
    if result['first_difference']:result['status']='FAIL_EXACT_SCORES'
    if result['compared_scores']==0 and result['first_difference'] is None:
        result.update(status='FAIL_EXACT_SCORES',first_difference={'index':[],'reason':'empty'})
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True,type=Path);ap.add_argument('--reference-sha256',required=True)
    ap.add_argument('--candidate',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(a.reference)==a.reference_sha256,'Accepted reference hash differs'
    assert not a.out.exists(),'Never overwrite a failed or successful gate receipt'
    r=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()))
    r.update(reference=str(a.reference),reference_sha256=sha(a.reference),candidate=str(a.candidate),candidate_sha256=sha(a.candidate),tolerance=0)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(r['status']);raise SystemExit(0 if r['status']=='EXACT' else 1)
