"""Render the plain serving profile, build its extensions or launch the server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from string import Template

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prebuild


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def profile(config, table):
    locations = dict(config["locations"], RECIPE=str(HERE))
    root = Path(locations["ROOT"])
    locations.update(TREE=str(root/"tree"), STATE=str(root/("state-"+table)))
    data = json.loads((HERE/"profile.json").read_bytes())
    args = [Template(s).substitute(locations) for s in data["argv"]]
    env = {k:Template(s).substitute(locations) for k,s in data["environment"].items()}
    if table == "B":
        env["KT_TP_PART_BLOCKS"] = "10,10,10,10,6,6,10,10"
    return root, args, env


def qualify(config, root, env):
    inputs = Path(config["locations"]["INPUT"])
    contract_path = inputs / "contract.json"
    contract = json.loads(contract_path.read_bytes())
    for variable, fact in (("V3DENSE_LIB","libv3dense_sha256"),("KT_V3H_LIB","libv3gpu_sha256")):
        pin = digest(Path(env[variable]))
        if pin != contract[fact]:
            raise ValueError(f"{variable} differs from its qualified contract")
        env[variable+"_SHA256"] = pin
    proof = inputs / "arena_format.json"
    env["KT_IMAGE_FORMAT_FILE"] = str(proof)
    env["KT_IMAGE_FORMAT_SHA256"] = digest(proof)
    proof_data = json.loads(proof.read_bytes())
    image_sha = digest(Path(env["KT_IMAGE_DIR"])/"manifest.json")
    if proof_data["image_manifest_sha256"] != image_sha:
        raise ValueError("Image and format proof differ")
    env["KT_IMAGE_MANIFEST_SHA256"] = image_sha
    devices = env["CUDA_VISIBLE_DEVICES"].split(",")
    if len(set(devices)) != 2 or len(devices) != 2:
        raise ValueError("Configure the two distinct GPU UUIDs in PCI bus order")
    for device in devices:
        path = inputs/(device+".json")
        report = json.loads(path.read_bytes())
        if not (report["gpu_uuid"] == device and report["kind"] == "ok" and
                report["total_mismatch"] == 0 and report["lib_sha256"] == env["KT_V3H_LIB_SHA256"]):
            raise ValueError(f"Component review differs for {device}")
        key = "KT_V3H_REVIEW_GPU_"+device.removeprefix("GPU-").replace("-","").upper()
        env[key], env[key+"_SHA256"] = str(path), digest(path)
    for key in ("KT_GGUF_REPACK_AVX2_LIBRARY","KT_V3H_SILU","KT_V3H_BL","KT_MXFP4_GGUF_MANIFEST"):
        pin = "KT_GGUF_REPACK_AVX2_SHA256" if key.endswith("_LIBRARY") else key+"_SHA256"
        if digest(Path(env[key])) != env[pin]:
            raise ValueError(f"Wrong input for {key}")
    modules = list((root/"kt-pkg/kt_kernel").glob("*kt_kernel_ext*.so"))
    if len(modules) != 1 or digest(modules[0]) != contract["kt_module_sha256"]:
        raise ValueError("Native package differs from the qualified module")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","prebuild","serve","verify"))
    parser.add_argument("config",type=Path)
    parser.add_argument("--table",choices=("A","B"),default="A")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_bytes())
    root, argv, env = profile(cfg,args.table)
    if args.action=="plan":
        print(json.dumps({"argv":argv,"environment":env},indent=2))
        return
    qualify(cfg,root,env)
    receipt = root/"cache"/prebuild.MARKER
    os.environ["C4_BUILD_LOCK"] = str(root/".build.lock")
    prebuild.BENCH_LOCK = os.environ["C4_BUILD_LOCK"]
    if args.action=="prebuild":
        for p in (Path(env["TORCH_EXTENSIONS_DIR"]),Path(env["TMPDIR"])):
            p.mkdir(exist_ok=True,parents=True)
        path=root/"prebuild-launch.json"
        if receipt.exists():
            raise SystemExit("An extension receipt already exists for this installation")
        path.write_text(json.dumps({"argv":argv,"environment":env},indent=2)+"\n",encoding="utf-8")
        raise SystemExit(prebuild.run(str(root),str(path),str(root/"prebuild-output")))
    record = json.loads(receipt.read_bytes())
    cache = prebuild.marked_cache(str(root),record)
    if not cache or not prebuild.ready(str(root),str(root/"prebuild-launch.json")):
        raise SystemExit("The prebuild receipt, builder or extension plan differs")
    env["TORCH_EXTENSIONS_DIR"]=cache
    # Pin the actual qualified freeze binary for this installation.
    fz, = [v for v in record["exts"].values() if v["key"]=="fz"]
    name = next(k for k,v in record["exts"].items() if v["key"]=="fz")
    env["SGLANG_V3_FZ_EXT_SHA256"]=fz["state"][name+".so"][0]
    result_path=root/("launch-"+args.table+".json")
    if args.action=="verify":
        raise SystemExit(prebuild.verify(str(root),str(result_path)))
    for key in ("KT_V3H_ORIGIN_DIR","KT_LEND_CACHE_DIR","SGLANG_V3_A1_CAPTURE","SGLANG_V3_KVT_STATS_DIR","SGLANG_V3_KVT_L3_DIR","TMPDIR"):
        Path(env[key]).mkdir(exist_ok=True,parents=True)
    Path(env["SGLANG_V3_HEADROOM_FILE"]).parent.mkdir(exist_ok=True,parents=True)
    result_path.write_text(json.dumps({"argv":argv,"environment":env},indent=2)+"\n",encoding="utf-8")
    os.execve(argv[0],argv,prebuild.server_env(os.environ,env))


if __name__=="__main__":
    main()
