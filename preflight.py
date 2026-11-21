"""Check release file integrity, source tree pins, and the two serving configurations."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import launch

TREES={"sglang":"f62b8d11d5d10a30b53b7f836283ee4fd85aa4f0","kt":"36f72b46dccf6ce3036f2c6162625d3adf147bf9"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sglang",type=Path)
    p.add_argument("kt",type=Path)
    p.add_argument("--config",type=Path,help="also check installed binary, image and per-device inputs")
    args=p.parse_args()
    for repo, pin in TREES.items():
        path=getattr(args,repo)
        tree=subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD^{tree}"],text=True).strip()
        if tree!=pin:
            raise SystemExit(f"{repo}: source tree differs")
        subprocess.run(["git","-C",str(path),"diff","--quiet","HEAD","--"],check=True)
    sums=(HERE/"SHA256SUMS").read_text().splitlines()
    for line in sums:
        h,rel=line.split("  ",1)
        data=(HERE/rel).read_bytes().replace(b"\r\n",b"\n")
        if hashlib.sha256(data).hexdigest()!=h:
            raise SystemExit(f"Recipe file differs: {rel}")
        if rel.endswith(".py"):
            ast.parse(data,filename=rel)
    cfg=json.loads((args.config or HERE/"config.example.json").read_bytes())
    for table in ("A","B"):
        root,argv,env=launch.profile(cfg,table)
        spec=[k for k in env if k.startswith("SGLANG_V3_SPEC") and env[k] not in ("0","")]
        if any(a.startswith("--speculative") for a in argv) or spec:
            raise SystemExit("The profile requests a speculative worker")
        if env.get("SGLANG_C4_ROUTE_R120_ORDER") != "1":
            raise SystemExit(f"Wrong router reduction for {table}")
        if env.get("KT_TP_PART_BLOCKS") != (None if table=="A" else "10,10,10,10,6,6,10,10"):
            raise SystemExit(f"Wrong NUMA split for {table}")
        if args.config:
            launch.qualify(cfg,root,env)
    print(f"PASS both source trees, {len(sums)} recipe files and both plain launch profiles")


if __name__=="__main__":
    main()
