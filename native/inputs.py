"""Generate the pinned SiLU table, compile the two SM120 libraries, and qualify a rebuilt image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def kernels(out):
    import numpy as np
    import torch
    from torch.nn.functional import silu
    out.mkdir(exist_ok=True,parents=True)
    bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    np.save(out/"q6-silu-gpu.npy",silu(bits.cuda().float()).cpu().view(torch.int32).numpy())
    contract=json.loads((HERE.parent/"inputs/contract.json").read_bytes())
    if sha(out/"q6-silu-gpu.npy")!=contract["silu_sha256"]:
        raise SystemExit("CUDA SiLU table differs")
    for name in ("v3gpu_r3i","v3dense"):
        subprocess.run([str(Path(os.environ.get("CUDA_HOME","/usr/local/cuda"))/"bin/nvcc"),"-O3","-std=c++17",
            "-gencode","arch=compute_120a,code=sm_120a","-fmad=false","--shared","-Xcompiler",
            "-fPIC,-O2,-fno-fast-math,-ffp-contract=off","-o",str(out/("lib"+name+".so")),str(HERE/(name+".cu"))],check=True)


def prove_image(out, directory, native):
    """Verify every reconstructed expert against its independently pinned source range."""
    import img_format as fmt
    geometry=fmt.Geometry()
    source=out/"expert-ranges.json"
    pins=json.loads((HERE.parent/"profile.json").read_bytes())["environment"]
    if sha(source)!=pins["KT_MXFP4_GGUF_MANIFEST_SHA256"]:
        raise ValueError("Expert range inventory differs")
    entries={(r["tensor"],r["expert"]):r for r in json.loads(source.read_bytes())["ranges"]}
    proof=json.loads((HERE.parent/"inputs/arena_format.json").read_bytes())
    build=json.loads((native/"ARENA_FORMAT.json").read_bytes())
    if build.get("format",build.get("arena_format")) != proof["arena_format"]:
        raise ValueError("Native arena layout differs")
    manifest,pin=fmt.load_manifest(directory)
    if manifest["geometry"]!=dict(geometry.as_dict(),layers=list(range(40))):
        raise ValueError("Image geometry differs")
    import mmap
    for layer in range(40):
        rec=manifest["layers"][str(layer)]
        if rec["file"]!=f"L{layer:02d}.img" or rec["bytes"]!=geometry.layer_bytes:
            raise ValueError(f"Image file differs for layer {layer}")
        with (directory/rec["file"]).open("rb") as file, mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as image:
            if len(image)!=rec["bytes"]:
                raise ValueError(f"Image length differs for layer {layer}")
            for p,proj in enumerate(("gate","up","down")):
                chunks={}
                for section,off,size in (("w",geometry.w_off,geometry.w_chunk),("s",geometry.s_off,geometry.s_chunk)):
                    group=[]
                    for b in range(geometry.nblk):
                        chunk=image[off(p,b):off(p,b)+size]
                        if hashlib.sha256(chunk).hexdigest()!=rec[section][p][b]:
                            raise ValueError(f"Chunk digest differs at {layer}/{proj}/{section}/{b}")
                        group.append(chunk)
                    chunks[section]=group
                for expert in range(geometry.E):
                    weights=fmt.expert_from_chunks(geometry,p,False,chunks["w"],expert)
                    scales=fmt.expert_from_chunks(geometry,p,True,chunks["s"],expert)
                    raw=fmt.arena_to_gguf(weights,scales)
                    if hashlib.sha256(raw).hexdigest()!=entries[f"blk.{layer}.ffn_{proj}_exps.weight",expert]["sha256"]:
                        raise ValueError(f"Source mismatch at layer {layer}, {proj}, expert {expert}")
        print(f"Verified layer {layer}",flush=True)
    proof.update(image_manifest_sha256=pin,module_sha256=build["module_sha256"])
    write(out/"arena_format.json",proof)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    sub.add_parser("kernels").add_argument("inputs",type=Path)
    im=sub.add_parser("image-proof")
    for key in ("inputs","image","native_build"):
        im.add_argument(key,type=Path)
    ns=p.parse_args()
    if ns.mode=="kernels":
        kernels(ns.inputs)
    else:
        prove_image(ns.inputs,ns.image,ns.native_build)


if __name__=="__main__":
    main()
