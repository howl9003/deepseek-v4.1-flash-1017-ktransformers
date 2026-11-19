"""Path B, option A: rebuild 05yuki main's kt-kernel AVX2 variant from his patched source, with the contract v3 (spec r3)
backend added, in a COPY (new namespace; his tree and wheel are only read).

  python -B build_pb.py <out_dir> [<sif>] (run through guard_pbc.py: node 3, no-swap service, bench lock, FEA idle)
  With <sif> (9 Nov, owner: "use GCC 13.3"): the pip wheel step runs inside that apptainer image (nvidia/cuda
  13.1.2-devel-ubuntu24.04 + his CI apt packages: Ubuntu 24.04 GCC 13.3, CUDA 13.1) with his prebuilt-pack.sh PATH order
  ($CUDA_HOME/bin : .venv/bin : system); the tree and venv are mounted into the container.

Recipe recovered from his repo (tools/prebuilt-pack.sh, .github/workflows/prebuilt.yml, prebuilt/manifest.json):
  `pip wheel --no-deps --no-build-isolation third_party/ktransformers/kt-kernel` with CPUINFER_BUILD_ALL_VARIANTS=1,
  CPUINFER_USE_CUDA=1, CPUINFER_CUDA_ARCHS="89;90;100;120", on Ubuntu 24.04 (GCC 13.3, CUDA 13.1, Python 3.12, glibc 2.39).
  setup.py's avx2 variant = CPUINFER_CPU_INSTRUCT=AVX2, CPUINFER_ENABLE_AVX512=OFF, CPUINFER_ENABLE_AMX=OFF,
  CPUINFER_ENABLE_ONEDNN_VNNI=OFF. Here: that variant alone, with the test server's compiler (recorded) and CUDA 13.4. The shipped
  module's .comment says GCC 13.3 (Ubuntu); the test server has GCC 12/15/16, so faithfulness is checked by output, not assumed
  (faith_test.py: his MXFP4 AVX2 backend from this module vs the shipped wheel, bit for bit, on real experts).
Adds: operators/avx2/contract_v3h-moe.hpp, operators/avx2/contract_v3/{v3_rm.h,v3_rm.cpp,v3_common.inc,v3_cpu.h,
v3_silu_table.cpp}; CMake option KT_ENABLE_CONTRACT_V3 (default ON in this copy; the v3 sources compile with
-fno-fast-math -ffp-contract=off); bindings AVX2ContractV3_MOE (forward_partials[_task], set_bl, ...); amx.py
KT_MXFP4_BACKEND=contract_v3. Everything else in kt-kernel is his source, unchanged.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

B = Path(os.environ.get("C4_ROOT", ".")).resolve()
TREE = B / "tree"
SRC = TREE / "third_party/ktransformers"
VENV_PY = TREE / ".venv/bin/python"
SHIPPED = Path(os.environ.get("WHEEL", TREE / "prebuilt/kt_kernel-0.7.0-cp312-cp312-linux_x86_64.whl"))
SHIPPED_SHA = "5b5141c81f12f76f18dedceb67fe85a6d31ee19c1690a64fc9a4c32c27107164"
CPU = Path(os.environ.get("RECIPE", ".")) / "native/cpu"
HERE = Path(__file__).resolve().parent
NS = HERE.parent
CUDA = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.4"))
SILU_NPY = Path(os.environ.get("C4_INPUT", B / "inputs")) / "q6-silu-gpu.npy"
SILU_SHA = "6498ac2e55adcee9e37e6e95c8ad620dcda33a76b1ea52e3f620be079ed48dec"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_once(path, old, new):
    s = path.read_text()
    if s.count(old) != 1:
        raise SystemExit(f"patch anchor not unique in {path}: {old[:80]!r} ({s.count(old)})")
    path.write_text(s.replace(old, new))


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
    N = Path(sys.argv[1])
    SIF = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    report = dict(status="RUNNING", steps=[], source=str(SRC), shipped_wheel=str(SHIPPED))

    def save():
        (N / "BUILD_RESULT.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(cmd, label, env, cwd=None):
        t0 = time.monotonic()
        with (N / (label + ".log")).open("w") as log:
            r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, timeout=4800)
        report["steps"].append(dict(label=label, exit=r.returncode, seconds=round(time.monotonic() - t0, 1), cmd=[str(c) for c in cmd]))
        save()
        if r.returncode:
            raise SystemExit(f"{label} failed, see {label}.log")

    try:
        assert sha(SHIPPED) == SHIPPED_SHA
        report["his_tree_head"] = subprocess.check_output(["git", "-C", str(TREE), "rev-parse", "HEAD"], text=True).strip()
        report["kt_source_head"] = subprocess.check_output(["git", "-C", str(SRC), "rev-parse", "HEAD"], text=True).strip()
        report["kt_source_status_lines"] = len(subprocess.check_output(["git", "-C", str(SRC), "status", "--porcelain"], text=True).splitlines())
        shutil.copytree(SRC, N / "ktransformers", symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__", "build"))
        kt = N / "ktransformers/kt-kernel"
        patched = [kt / "CMakeLists.txt", kt / "ext_bindings.cpp", kt / "python/utils/amx.py"]
        report["before"] = {str(p.relative_to(kt)): sha(p) for p in patched}

        d = kt / "operators/avx2/contract_v3"
        d.mkdir()
        for f in ("v3_rm.h", "v3_rm.cpp", "v3_common.inc", "v3_cpu.h"):
            shutil.copy2(CPU / f, d / f)
        assert sha(SILU_NPY) == SILU_SHA
        bits = np.load(SILU_NPY).astype(np.uint32)
        assert bits.shape == (65536,)
        rows = [", ".join(f"0x{int(v):08x}u" for v in bits[i:i + 8]) for i in range(0, 65536, 8)]
        (d / "v3_silu_table.cpp").write_text(
            "// Generated by the CPU builder from q6-silu-gpu.npy (torch F.silu on the RTX 5070 Ti, indexed by\n"
            f"// BF16 bits); source sha256 {SILU_SHA}.\n#include <stdint.h>\n"
            f'extern "C" const char kt_contract_v3_silu_sha256[] = "{SILU_SHA}";\n'
            'extern "C" const uint32_t kt_contract_v3_silu_bits[65536] = {\n' + ",\n".join(rows) + "\n};\n")
        shutil.copy2(HERE / "contract_v3h-moe.hpp", kt / "operators/avx2/contract_v3h-moe.hpp")

        cm = kt / "CMakeLists.txt"
        replace_once(cm, 'option(CPUINFER_ENABLE_LTO ',
                     'option(KT_ENABLE_CONTRACT_V3 "Build the contract v3 (spec r3, tiers C/D) AVX2 MoE backend; runtime opt-in KT_MXFP4_BACKEND=contract_v3" ON)\n'
                     'option(CPUINFER_ENABLE_LTO ')
        cm.write_text(cm.read_text() + (
            '\n# path B: contract v3 (spec r3) AVX2 MoE backend\n'
            'if(KT_ENABLE_CONTRACT_V3)\n'
            '    set(KT_V3_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/operators/avx2/contract_v3/v3_rm.cpp\n'
            '                      ${CMAKE_CURRENT_SOURCE_DIR}/operators/avx2/contract_v3/v3_silu_table.cpp)\n'
            '    target_sources(${PROJECT_NAME} PRIVATE ${KT_V3_SOURCES})\n'
            '    # the v3 arithmetic must not see -ffast-math: these flags come last\n'
            '    set_source_files_properties(${KT_V3_SOURCES} PROPERTIES\n'
            '        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-reciprocal-math;-fno-unsafe-math-optimizations;-mavx2;-mfma")\n'
            '    target_compile_definitions(${PROJECT_NAME} PRIVATE KT_ENABLE_CONTRACT_V3=1)\n'
            '    message(STATUS "contract v3 MoE backend: enabled (runtime opt-in KT_MXFP4_BACKEND=contract_v3)")\n'
            'endif()\n'))

        eb = kt / "ext_bindings.cpp"
        replace_once(eb, '#include "operators/avx2/mxfp4-moe.hpp"\n',
                     '#include "operators/avx2/mxfp4-moe.hpp"\n#ifdef KT_ENABLE_CONTRACT_V3\n#include "operators/avx2/contract_v3h-moe.hpp"\n#endif\n')
        replace_once(eb, '  // kt-moe-partials-out.py\n  if constexpr (requires { &MoeClass::set_partial_outputs; }) {',
                     '#ifdef KT_ENABLE_CONTRACT_V3\n'
                     '  // path B contract v3 (spec r3): raw int64 partial [tokens, hidden] as a CPUInfer task, plus the per-layer grid\n'
                     '  if constexpr (requires { &MoeClass::forward_partials_binding; }) {\n'
                     '    struct ContractV3PartialBindings {\n'
                     '      struct Args { CPUInfer* cpuinfer; MoeClass* moe; intptr_t qlen; int k; intptr_t ids; intptr_t weights; intptr_t input; intptr_t output; };\n'
                     '      static void inner(void* p) {\n'
                     '        Args* a = (Args*)p;\n'
                     '        a->cpuinfer->enqueue(&MoeClass::forward_partials_binding, a->moe, a->qlen, a->k, a->ids, a->weights, a->input, a->output);\n'
                     '      }\n'
                     '      static std::pair<intptr_t, intptr_t> task(std::shared_ptr<MoeClass> moe, intptr_t qlen, int k, intptr_t ids, intptr_t weights,\n'
                     '                                                intptr_t input, intptr_t output) {\n'
                     '        Args* a = new Args{nullptr, moe.get(), qlen, k, ids, weights, input, output};\n'
                     '        return std::make_pair((intptr_t)&inner, (intptr_t)a);\n'
                     '      }\n'
                     '    };\n'
                     '    moe_cls.def("forward_partials_task", &ContractV3PartialBindings::task)\n'
                     '        .def("forward_partials", &MoeClass::forward_partials_binding)\n'
                     '        .def("set_bl", &MoeClass::set_bl)\n'
                     '        .def("get_bl", &MoeClass::get_bl)\n'
                     '        .def("contract_v3_tier", &MoeClass::contract_v3_tier)\n'
                     '        .def("contract_v3_spec", &MoeClass::contract_v3_spec)\n'
                     '        .def("contract_v3_silu_sha256", &MoeClass::contract_v3_silu_sha256);\n'
                     '  }\n'
                     '#endif\n'
                     '  // kt-moe-partials-out.py\n  if constexpr (requires { &MoeClass::set_partial_outputs; }) {')
        replace_once(eb, '  bind_moe_module<AVX2_MXFP4_MOE_TP<avx2::GemmKernelAVX2MXFP4>>(moe_module, "AVX2MXFP4_MOE");\n',
                     '  bind_moe_module<AVX2_MXFP4_MOE_TP<avx2::GemmKernelAVX2MXFP4>>(moe_module, "AVX2MXFP4_MOE");\n'
                     '#ifdef KT_ENABLE_CONTRACT_V3\n'
                     '  bind_moe_module<AVX2_CONTRACT_V3H_TP<avx2::GemmKernelAVX2MXFP4>>(moe_module, "AVX2ContractV3_MOE");\n'
                     '#endif\n')
        amx = kt / "python/utils/amx.py"
        replace_once(amx, 'AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)\n',
                     'AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)\n'
                     'AVX2ContractV3_MOE = getattr(_moe_mod, "AVX2ContractV3_MOE", None)   # path B contract v3 (spec r3) builds only\n')
        replace_once(amx, '    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()\n\n    if forced == "amx":\n',
                     '    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()\n\n'
                     '    if forced == "contract_v3":\n'
                     '        if AVX2ContractV3_MOE is None:\n'
                     '            raise RuntimeError("KT_MXFP4_BACKEND=contract_v3 requested, but AVX2ContractV3_MOE is not compiled in.")\n'
                     '        return AVX2ContractV3_MOE\n'
                     '    if forced == "amx":\n')
        report["after"] = {str(p.relative_to(kt)): sha(p) for p in patched}
        report["added"] = {str(p.relative_to(kt)): sha(p) for p in sorted(d.iterdir())}
        report["added"]["operators/avx2/contract_v3h-moe.hpp"] = sha(kt / "operators/avx2/contract_v3h-moe.hpp")
        if SIF is None:
            report["compiler"] = subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]
        else:
            report["sif"] = str(SIF)
            report["sif_sha256"] = sha(Path(SIF) / ".kt-image.json") if Path(SIF).is_dir() else sha(SIF)
            # r2 (build without fakeroot): the pulled image + an extracted prefix (mk_sif.sh r2), bound at /opt/ktpre
            PREFIX = Path(SIF).parent / "prefix"
            if not (PREFIX / "CC_CXX").is_file():
                raise SystemExit("no toolchain prefix next to the image: " + str(PREFIX))
            CC, CXX = (PREFIX / "CC_CXX").read_text().split()
            report["prefix"] = str(PREFIX)
            report["cc_cxx"] = [CC, CXX]
            cx = ["/usr/bin/apptainer", "exec", "--cleanenv", "--no-home", "--bind", f"{PREFIX}:/opt/ktpre:ro", SIF]
            report["compiler"] = subprocess.check_output(cx + [CC, "--version"], text=True).splitlines()[0]
            report["compiler_cxx"] = subprocess.check_output(cx + [CXX, "--version"], text=True).splitlines()[0]
            report["nvcc"] = subprocess.check_output(cx + ["/usr/local/cuda-13.1/bin/nvcc", "--version"], text=True).splitlines()[-1]
            if "13.3.0" not in report["compiler"]:
                raise SystemExit("container gcc is not 13.3: " + report["compiler"])
        save()

        venv_bin = VENV_PY.parent
        env = dict(PATH=f"{venv_bin}:{CUDA}/bin:/usr/bin:/bin", HOME=str(Path.home()),
                   CUDA_VISIBLE_DEVICES="", CUDA_HOME=str(CUDA), MAX_JOBS="4", CPUINFER_PARALLEL="4",
                   CPUINFER_CPU_INSTRUCT="AVX2", CPUINFER_ENABLE_AVX512="OFF", CPUINFER_ENABLE_AMX="OFF",
                   CPUINFER_ENABLE_ONEDNN_VNNI="OFF", CPUINFER_USE_CUDA="1", CPUINFER_CUDA_ARCHS="89;90;100;120",
                   CPUINFER_BUILD_TYPE="Release", TMPDIR=str(N / "tmp"), PIP_NO_CACHE_DIR="1", PYTHONDONTWRITEBYTECODE="1")
        (N / "tmp").mkdir(exist_ok=True)
        report["build_env"] = {k: v for k, v in env.items() if k.startswith(("CPUINFER", "CUDA", "MAX_JOBS"))}
        cmd = [VENV_PY, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-v", "-w", N / "wheel", kt]
        if SIF is not None:
            cuda_c = "/usr/local/cuda-13.1"
            pre = "/opt/ktpre/usr"
            lib = f"{pre}/lib/x86_64-linux-gnu"
            env.update(CUDA_HOME=cuda_c, PATH=f"{cuda_c}/bin:{venv_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                       CC=CC, CXX=CXX, CUDAHOSTCXX=CXX, CPATH=f"{pre}/include:{pre}/include/x86_64-linux-gnu", LIBRARY_PATH=lib, LD_LIBRARY_PATH=lib,
                       PKG_CONFIG_PATH=f"{lib}/pkgconfig:{pre}/share/pkgconfig", CMAKE_PREFIX_PATH=pre)
            report["build_env"] = {k: v for k, v in env.items() if k.startswith(("CPUINFER", "CUDA", "MAX_JOBS", "PATH", "CC", "CXX",
                                                                                 "CPATH", "LIBRARY", "LD_", "PKG", "CMAKE"))}
            host = {"PATH": "/usr/bin:/bin", "HOME": env["HOME"]}
            host.update({"APPTAINERENV_" + k: v for k, v in env.items()})
            uv = Path(os.path.realpath(VENV_PY)).parents[2]
            cmd = ["/usr/bin/apptainer", "exec", "--cleanenv", "--no-home", "--bind", str(N.resolve()),
                   "--bind", f"{TREE.resolve()}:{TREE.resolve()}:ro", "--bind", f"{uv}:{uv}:ro",
                   "--bind", f"{PREFIX}:/opt/ktpre:ro",
                   "--pwd", str(N), SIF] + cmd
            env = host
        run(cmd, "build", env)
        whl, = list((N / "wheel").glob("kt_kernel-*.whl"))
        with zipfile.ZipFile(whl) as z:
            names = [n for n in z.namelist() if n.endswith(".so") and "kt_kernel" in n]
            report["wheel_so"] = names
            z.extractall(N / "pkg")
        sos = sorted((N / "pkg/kt_kernel").glob("*kt_kernel_ext*.so"))
        report["artifacts"] = {p.name: sha(p) for p in sos}
        # the strict-FP evidence for the v3 entry points (no reciprocal approximations)
        so = sos[0]
        dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", str(so)], capture_output=True, text=True).stdout
        cur, bad = None, {}
        for ln in dis.splitlines():
            if ln.endswith(">:"):
                cur = ln
                continue
            if cur and ("v3_" in cur or "contract_v3" in cur) and any(t in ln for t in ("rcpps", "rcpss", "rsqrtps", "rsqrtss")):
                bad.setdefault(cur, []).append(ln.strip())
        report["v3_suspect_instructions"] = bad
        report.update(status="KT_PB_BUILD_PASS", module=str(so), module_sha256=sha(so))
        save()
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
