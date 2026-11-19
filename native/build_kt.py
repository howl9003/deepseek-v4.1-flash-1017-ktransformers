"""Build the release overlay on the public recipe's patched kTransformers tree."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kt", type=Path)
    p.add_argument("root", type=Path)
    p.add_argument("wheel", type=Path, help="the 0.7.0 reference wheel required by the native builder")
    p.add_argument("input", type=Path)
    p.add_argument("--cuda",type=Path,default=Path("/usr/local/cuda"))
    args=p.parse_args()
    root, kt = args.root.resolve(), args.kt.resolve()
    tree = root/"tree"
    cpu=Path(__file__).resolve().parent/"cpu"
    if (root/"kt-pkg").exists():
        p.error("kt-pkg already exists")
    spec=importlib.util.spec_from_file_location("overlay_build", kt/"build_bs.py")
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.TREE, mod.SRC = tree, tree/"third_party/ktransformers"
    mod.VENV_PY, mod.SHIPPED = tree/".venv/bin/python", args.wheel.resolve()
    mod.CPU, mod.SILU_NPY, mod.CUDA = cpu,args.input.resolve()/"q6-silu-gpu.npy",args.cuda.resolve()
    out=root/"native-build"
    out.mkdir(exist_ok=False)
    sys.argv=[str(kt/"build_bs.py"),str(out)]
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    mod.main()
    report=json.loads((out/"BUILD_RESULT.json").read_bytes())
    if report["status"]!="KT_PB_BUILD_PASS" or report["v3_suspect_instructions"]:
        raise SystemExit("Native disassembly or build check failed")
    # Keep the recipe's Python host adapter with the rebuilt AVX2 extension.
    src=tree/".venv/lib/python3.12/site-packages/kt_kernel"
    pkg=out/"install/kt_kernel"
    shutil.copytree(src,pkg,ignore=shutil.ignore_patterns("*.so","*.pyc","__pycache__"))
    shutil.copyfile(report["module"],pkg/Path(report["module"]).name)
    amx=pkg/"utils/amx.py"
    # The base wheel uses a safetensors loader. Install the qualified GGUF bridge
    # before adding the strict direct-image path and its fallback.
    storage = '            from .recipe_gguf_storage import create_mxfp4_loader'
    current = amx.read_text(encoding="utf-8")
    if storage not in current:
        stock = '        elif method == "MXFP4":\n            return MXFP4SafeTensorLoader(weight_path)'
        if current.count(stock) != 1:
            raise ValueError("The MXFP4 loader selector differs from the 1017 base")
        upgraded = '        elif method == "MXFP4":\n' + storage + '\n            return create_mxfp4_loader(weight_path, MXFP4SafeTensorLoader)'
        amx.write_text(current.replace(stock, upgraded), encoding="utf-8", newline="\n")
    for helper in (cpu.parent/"host").glob("recipe_gguf_*.py"):
        shutil.copyfile(helper, pkg/"utils"/helper.name)
    subprocess.run([sys.executable,"-I","-B",str(cpu.parent/"select_contract.py"),str(amx)],check=True)
    subprocess.run([sys.executable,"-I","-B",str(kt/"python/utils/patch_amx_image.py"),str(amx),str(kt/"python/utils/kt_image.py")],check=True)
    (out/"install").rename(root/"kt-pkg")


if __name__=="__main__":
    main()
