"""Read-only full routed-weight integrity baseline, never original-source proof.

Single bounded pass over existing MXFP4 experts; no dequantized weight copy.
An independently reviewed manifest hash is required by the actual reader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from gguf import GGUFReader


def identity(stat):
    return [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    started=time.monotonic()
    fd=os.open(a.source,os.O_RDONLY)
    original=identity(os.fstat(fd))
    if original[2]!=507954225856:
        raise ValueError('target file size differs from retained audited inventory')
    gg=GGUFReader(a.source)
    tensors={t.name:t for t in gg.tensors}
    header_bytes=min(int(t.data_offset) for t in gg.tensors)
    if header_bytes!=5833888 or len(tensors)!=1046:
        raise ValueError('target GGUF inventory differs')
    header=os.pread(fd,header_bytes,0)
    records=[];total=0;hist=np.zeros(256,dtype=np.uint64)
    # Geometry and extent validation before the potentially long pass.
    selected=[]
    for layer in range(40):
        for projection,n,k in [('gate',2304,5120),('up',2304,5120),('down',5120,2304)]:
            name=f'blk.{layer}.ffn_{projection}_exps.weight'
            t=tensors[name]
            if t.tensor_type.name!='MXFP4' or [int(x) for x in reversed(t.shape)] != [384,n,k]:
                raise ValueError('non-native tensor or geometry: '+name)
            size=n*k//32*17;offset=int(t.data_offset)
            if offset<header_bytes or offset+384*size>original[2]:
                raise ValueError('invalid tensor extent')
            selected.append((layer,name,n,k,size,offset))
    extents=sorted((offset,offset+384*size) for _,_,_,_,size,offset in selected)
    if any(a1>b0 for (_,a1),(b0,_) in zip(extents,extents[1:])):
        raise ValueError('overlapping expert stacks')
    for index,(layer,name,n,k,size,offset) in enumerate(selected):
        for expert in range(384):
            raw=os.pread(fd,size,offset+expert*size)
            if len(raw)!=size:
                raise ValueError('short source read')
            scales=np.frombuffer(raw,dtype=np.uint8)[::17]
            hist+=np.bincount(scales,minlength=256).astype(np.uint64)
            if hist[255]:
                raise ValueError('source contains reserved NaN scale code')
            records.append(dict(tensor=name,expert=expert,shape=[n,k],offset=offset+expert*size,
                                bytes=size,sha256=hashlib.sha256(raw).hexdigest()))
            total+=size
        if identity(os.fstat(fd))!=original or identity(a.source.stat())!=original:
            raise ValueError('source changed during inventory')
        if index%3==2:
            print(json.dumps(dict(layer=layer,bytes=total,seconds=time.monotonic()-started)),flush=True)
    manifest=dict(schema='kt-native-mxfp4-ranges-v1',file_size=original[2],header_bytes=header_bytes,
        header_sha256=hashlib.sha256(header).hexdigest(),ranges=records,
        provenance='Integrity baseline of existing local GGUF; full original-HF byte identity is not independently proven.',
        original_model_revision='1bae0e1088993b9b82de08f59de5be5a34e42170')
    payload=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != "9246e760c554a746662f3537cb2aadb626a90d827d2edbdda2e31438c7f30d12":
        raise ValueError("Range inventory differs from the qualified 1017 GGUF")
    (a.output/'expert-ranges.json').write_bytes(payload)
    os.close(fd)
    print(f"PASS {len(records)} native expert ranges, SHA-256 {actual}", flush=True)


if __name__ == "__main__":
    main()
