"""Bind the installed contract to the native build and two GPU qualification results."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("inputs",type=Path)
    p.add_argument("devices",nargs=2)
    args=p.parse_args()
    inputs=args.inputs
    here=Path(__file__).resolve().parents[1]
    contract=json.loads((here/"inputs/contract.json").read_bytes())
    kt=json.loads((args.root/"native-build/BUILD_RESULT.json").read_bytes())
    if kt["status"]!="KT_PB_BUILD_PASS" or kt["v3_suspect_instructions"]:
        raise SystemExit("Native build or disassembly check failed")
    so, = (args.root/"kt-pkg/kt_kernel").glob("*kt_kernel_ext*.so")
    if sha(so)!=kt["module_sha256"]:
        raise SystemExit("Installed module differs from the build")
    lib=sha(inputs/"libv3gpu_r3i.so")
    if len(set(args.devices))!=2:
        raise SystemExit("Two distinct component reviews are required")
    for uuid in args.devices:
        review=json.loads((inputs/(uuid+".json")).read_bytes())
        if not (review["gpu_uuid"]==uuid and review["kind"]=="ok" and review["total_mismatch"]==0 and
                review["checks"] and all(v["compared"]>0 and v["mismatch"]==0 for v in review["checks"].values())
                and review["lib_sha256"]==lib):
            raise SystemExit(f"GPU component qualification failed for {uuid}")
    if sha(inputs/"libv3dense.so")!=contract["libv3dense_sha256"]:
        raise SystemExit("Dense library differs from the measured binary. Supply the qualified binary.")
    contract.update(kt_module_sha256=kt["module_sha256"],libv3gpu_sha256=lib)
    dest=inputs/"contract.json"
    tmp=dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
    tmp.replace(dest)


if __name__=="__main__":
    main()
