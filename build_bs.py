"""pathb-image-r3 (16 Nov; branch pathb-image-r3 off pathb-image-r2 7a83c4d): r2 with its preload race removed. r2 made
each layer's module on the Python main thread while an earlier layer's image task ran on the CPUInfer worker; both call
NumaJobDistributor::do_numa_job (the module's constructor; the task's read, code tables and collapse), which keeps one
compute_func and per-node flags with no lock. r3: (h) contract_v3h-moe.hpp ImageBuild: the module's constructor, the
part-rows check, image_prepare and load_weights_image in ONE CPUInfer task (bound as AVX2ContractV3_MOE.image_build(...)
and the record AVX2ContractV3_ImageBuild: task, done, status, rows, seconds, tid, take), so every do_numa_job of a
preloaded load runs on the worker in queue order; (i) python/utils/kt_image.py queues that task at wrapper creation and
makes or drops a module on the main thread only once every queued build has run (_kt_image_drain); a module without
image_build gets no preload. The load's statements are r2's, ARENA_WRITERS untouched: the arena format string is r1's.

pathb-image-r2 (16 Nov, concept-r1 STARTUP_CUTS_r1.md section 3; branch pathb-image-r2 off pathb-image-r1 b9eb35d): the
pathb-image-r1 build below with (e) a second amx.py hook, at the GGUF loader's creation in NativeMoEWrapper.__init__
(IMAGE_INIT_HOOK): when the image can serve the layer the wrapper makes no GGUF loader (load_weights makes it on a
fallback, as his code already does after a release), and the layer's image task is queued at once (KT_IMAGE_PRELOAD,
on unless 0; the map is the one sglang published before building the model). load_weights adopts the module once that
layer's task is done; (f) contract_v3h-moe.hpp: an atomic done flag raised when load_weights_image() returns, and the
binding image_done(); (g) KT_IMAGE_DIGEST=1 prints staged-imgid's per-part arena digests after every image load. No
arena writer changes (ARENA_WRITERS untouched: the arena format string is r1's), no kernel or arithmetic change.

pathb-image-r1 (15 Nov, concept-r1 IMAGE_LOAD_DESIGN_r1.md; branch pathb-image-r1 off pathb-stagger-r2 e2eef18 = arm T3S's
module 23194614, with pathb-nodescale-r1 0264d2b merged): the two builds below together, plus (c) the arena format
pin: this script hashes the sources that write the expert arena bytes as they stand in the tree after every edit
(ARENA_WRITERS: mxfp4-moe.hpp, kt_image.hpp, his moe_base.hpp, his moe-tp.hpp), and compiles the string
"pathb-arena-v1;blk=32;scale=e8m0;w=<first 16 hex of that hash>" into the module as the static binding
AVX2ContractV3_MOE.kt_arena_format(); amx.py's image hook loads a layer only when the image's proven arena format
(its manifest field, or the sidecar arena_format.json pinned by KT_IMAGE_FORMAT_SHA256) is that string and the
layer's part rows are among the proven ones; (d) after the wheel step, the package's utils/amx.py must carry the
image hook and the appended python/utils/kt_image.py (else the build fails). KT_IMAGE_DIR unset: no image code runs
and the arenas are built by the same statements as the stagger-r2 module's (KT_TP_PART_BLOCKS unset: his split,
kt_part_row0(i) = i * I / P). No arithmetic, kernel or thread change.

pathb-stagger-r2 (13 Nov, concept-r1 STAGGER_DESIGN_r1.md section 9.2; branch pathb-stagger-r2 off pathb-cpu-r1 c64e1ca):
the pathb-cpu-r1 build below, unchanged in its steps, with KT_MEMOP_SYNC_PER_TASK=1 added (off unless set) in
cpu_backend/cpuinfer.h, task_queue.h, task_queue.cpp and the new cpu_backend/memop_per_task.h (copied next to
memop_detect.h): a stream-memop sync is served only after its own submit (KT_MEMOP_PAIRED's rule, implied) and raises its
done flag once that submit's task has finished, without blocking the poller, so two decode lanes stop waiting for each
other's CPU task. Unset, the module behaves as the pathb-cpu-r1 one (the TaskQueue counters stay off, the pollers are
unchanged). No arithmetic, binding or kernel changes.

pathb-nodescale-r1 (14 Nov, concept-r1 NODE_SCALING_r1.md section 3; branch pathb-nodescale-r1 off pathb-cpu-r1): the
build below plus (a) parts of different sizes: KT_TP_PART_BLOCKS="b0,b1,..." gives NUMA part i b_i blocks of 32
intermediate columns, by anchored edits of his operators/moe-tp.hpp (TP_MOE_Common's constructor; the tree's file must be
263d40c1, the one the edit was made from, and the three anchors unique, or the build stops) and the loader at prefix
offsets in mxfp4-moe.hpp (equal parts load exactly as before);
only the contract v3 backend loads uneven parts; (b) the block-ordered expert image: operators/avx2/kt_image.hpp (new),
load_weights_image() in the contract v3 class, and in amx.py the hook that loads a layer from the image when
KT_IMAGE_DIR and KT_IMAGE_MANIFEST_SHA256 are set (python/utils/kt_image.py appended to amx.py). Bindings added:
load_weights_image_task, image_prepare, image_status, image_stats, contract_v3_part_rows. Unset, every path is the
pathb-cpu-r1 module's.

pathb-cpu-r1 (13 Nov, concept-r1 ZEN3_OPTIMISATION_r1.md section 3.2; branch pathb-cpu-r1 off pathb-pairfix with
pathb-preread and pathb-kernel-rw merged): every source of the three lines below, plus the CPU switches of this branch, each
off unless set: KT_V3_CCD (per-CCD static split and per-CCD pre-read), KT_V3_RW_RED=2 (the row-wise kernel's microcode-free
reduction), KT_V3_THP_COLLAPSE (MADV_COLLAPSE of the written arena extents at load), KT_V3_SUM_PART (the part sum's
subpool), KT_MEMOP_FLAGS_NODE (the memop flags' node), and in his worker_pool.cpp, by one anchored edit applied only if
its anchors are unique (else the file is left as it is and BUILD_RESULT.json says so), KT_NUMA_WAIT_PAUSE (a PAUSE in
do_numa_job's wait; a live knob of his kt_knobs.h). Bindings added: contract_v3_set_ccd, contract_v3_ccd,
contract_v3_set_rw, contract_v3_rw, contract_v3_thp, contract_v3_sum_part. A compile error stops the build; the job then
prints the compiler's error lines from build.log.

concept-r1 poller pairing fix (SEAM_FUSION_r1.md section 11, 13 Nov): the detector build below, with the same four files
carrying KT_MEMOP_PAIRED=1 too (cpuinfer.h, memop_detect.h; task_queue.* unchanged): a stream-memop sync is served only
after its own submit; a sync met first is left raised for the next pass. Unset, the module behaves as the detector one.
This script's steps are unchanged; only the copied sources differ.

concept-r1 poller detector (SEAM_FUSION_r1.md section 9, 13 Nov): the byte-scales build below, unchanged, plus his three
task-queue files (cpu_backend/cpuinfer.h, task_queue.cpp, task_queue.h) replaced by this directory's after checking
that the tree's are the ones they were made from, and cpu_backend/memop_detect.h added. KT_MEMOP_DETECT=1 counts the
stream-memop poller's out-of-order syncs into a JSON-lines file; unset, the module behaves as the byte-scales one (the
poller and TaskQueue run unchanged). No arithmetic, binding or kernel changes.

concept-r1 byte scales (BYTE_SCALES_r1.md, 12 Nov): the pinned CPU module build with
this directory's contract_v3h-moe.hpp in place of the pinned one and this directory's operators/avx2/mxfp4-moe.hpp in
place of his (after checking that the tree's file is the one it was made from). Same tree, same toolchain image, same
flags. KT_ARENA_SCALE_BYTES=1 makes the scale arenas hold E8M0 bytes; unset, the module behaves as the pinned one. The
module it writes is a NEW module in its own out dir; the pinned one (freeze/queue/lib.sh MOD / MODPIN) is never touched.
Bindings added: contract_v3_arena_scale_bytes, contract_v3_codes_fnv.

concept-r1 pre-read (PREREAD_r1.md, 13 Nov; branch pathb-preread off pathb-cpusmooth): as below, plus cpu_backend/
task_queue.h replaced (after the same sha check) for the idle hook, and five test bindings: contract_v3_set_preread,
contract_v3_preread, contract_v3_preread_log, contract_v3_preread_hint, contract_v3_tq_last_qwait_ns. KT_PREREAD unset:
the hook is never installed.

concept-r1 CPU smoothing (RAM_BANDWIDTH_r1.md, 12 Nov): the pinned CPU module build with
this directory's contract_v3h-moe.hpp in place of the pinned one. Same tree, same toolchain image, same flags. The module
it writes is a NEW module in its own out dir; the pinned one (freeze/queue/lib.sh MOD / MODPIN) is never touched.
Bindings added: contract_v3_codes_placement, contract_v3_set_clean, contract_v3_clean. cpu_backend/task_queue.cpp and
cpu_backend/cpuinfer.h are replaced by this directory's (two pin knobs, off unless set), after checking that the tree's
files are the ones they were made from.

Path B, option A: rebuild 05yuki main's kt-kernel AVX2 variant from his patched source, with the contract v3 (spec r3)
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


# pathb-image-r1: the arena format. These files, as they stand in the build tree after every edit, decide the bytes of
# the expert arenas (allocation and stride: his moe_base.hpp; the part split: his moe-tp.hpp with the KT_TP_PART_BLOCKS
# edit; the GGUF-path writer: mxfp4-moe.hpp; the image-path writer: kt_image.hpp). contract_v3h-moe.hpp reads the arenas
# (code tables, MADV_COLLAPSE, pre-read) and writes none of their bytes, so a kernel-only change keeps the format.
ARENA_FORMAT_ID = "pathb-arena-v1"   # bump by hand only if the image schema (pathb-expert-image-v1) changes
ARENA_WRITERS = ("operators/avx2/mxfp4-moe.hpp", "operators/avx2/kt_image.hpp", "operators/avx2/moe_base.hpp",
                 "operators/moe-tp.hpp")


def arena_format(kt):
    """(format string, {file: sha256}, writer sha256) of the kt-kernel tree at kt."""
    files = {rel: sha(kt / rel) for rel in ARENA_WRITERS}
    w = hashlib.sha256("".join(f"{rel}\0{files[rel]}\n" for rel in ARENA_WRITERS).encode()).hexdigest()
    return f"{ARENA_FORMAT_ID};blk=32;scale=e8m0;w={w[:16]}", files, w


# pathb-image-r1: the image hook at the top of NativeMoEWrapper.load_weights (pathb-nodescale's text, unchanged); the
# package's utils/amx.py must carry it once and end with python/utils/kt_image.py (python/utils/patch_amx_image.py
# puts the same two things into a served kt-pkg's amx.py)
IMAGE_ANCHOR = '            return self._load_weights_mxfp4_cache(physical_to_logical_map_cpu)\n'
IMAGE_HOOK = (IMAGE_ANCHOR +
              '        # pathb-nodescale (concept-r1 NODE_SCALING_r1.md section 3): the block-ordered expert image\n'
              '        if self.method == "MXFP4" and os.environ.get("KT_IMAGE_DIR"):\n'
              '            if _kt_image_load_layer(self, physical_to_logical_map_cpu):\n'
              '                return\n')
# pathb-image-r2: the second hook, at the first wrapper's GGUF loader creation in NativeMoEWrapper.__init__
IMAGE_INIT_ANCHOR = ('        if NativeMoEWrapper._native_loader_instance is None:\n'
                     '            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)\n'
                     '        self.loader = NativeMoEWrapper._native_loader_instance\n')
IMAGE_INIT_HOOK = ('        # pathb-image-r2 (concept-r1 STARTUP_CUTS_r1.md section 3): no GGUF loader while the image serves the layer\n'
                   '        # (load_weights makes one on a fallback); the layer\'s image task is queued here (KT_IMAGE_PRELOAD)\n'
                   '        if method == "MXFP4" and os.environ.get("KT_IMAGE_DIR") and _kt_image_init(self):\n'
                   '            self.loader = None\n'
                   '        else:\n'
                   '            if NativeMoEWrapper._native_loader_instance is None:\n'
                   '                NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)\n'
                   '            self.loader = NativeMoEWrapper._native_loader_instance\n')


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

        # concept-r1 byte scales: his MXFP4 backend file with the KT_ARENA_SCALE_BYTES switch
        report["replaced"] = {}
        for rel, base in (("operators/avx2/mxfp4-moe.hpp", "354d926e9465b9ec27d84b2a148d94aa92a56c42b3df6620f5f19f22e39178db"),):
            if sha(kt / rel) != base:
                raise SystemExit(f"{rel} in the tree is not the file the change was made from")
            shutil.copy2(HERE / rel, kt / rel)
            report["replaced"][rel] = {"from": base, "to": sha(kt / rel)}

        # concept-r1 poller detector, CPU smoothing and pre-read: his three task-queue files with KT_MEMOP_DETECT and
        # KT_MEMOP_PAIRED, the pin knobs (KT_TASKQUEUE_CPU, KT_MEMOP_POLLER_CPU) and the idle hook (all off unless set),
        # and the detector's header
        for rel, base in (("cpu_backend/task_queue.h", "ddf0967d003159399c967ef885b20261aa690dfd695ef5c10ceaccc06d76f319"),
                          ("cpu_backend/task_queue.cpp", "9080c8ad286235f02ba9b7d4c048f4dcc4e5bf524ebf720785a22d1086f34a0f"),
                          ("cpu_backend/cpuinfer.h", "565c84e1466de281e90c1bad61ce7eafb97b257dd55fc24d0f4899c7ac162719")):
            if sha(kt / rel) != base:
                raise SystemExit(f"{rel} in the tree is not the file the change was made from")
            shutil.copy2(HERE / rel, kt / rel)
            report["replaced"][rel] = {"from": base, "to": sha(kt / rel)}
        # concept-r1 stagger r2: the per-task sync's header (KT_MEMOP_SYNC_PER_TASK, off unless set) next to the detector's
        for rel in ("cpu_backend/memop_detect.h", "cpu_backend/memop_per_task.h"):
            if (kt / rel).exists():
                raise SystemExit(f"{rel} already exists in the tree")
            shutil.copy2(HERE / rel, kt / rel)
            report["replaced"][rel] = {"from": None, "to": sha(kt / rel)}

        d = kt / "operators/avx2/contract_v3"
        d.mkdir()
        for f in ("v3_rm.h", "v3_rm.cpp", "v3_common.inc", "v3_cpu.h"):
            shutil.copy2(CPU / f, d / f)
        # row-wise kernels (concept-r1/CPU_KERNEL_COMPUTE_r1.md): used when KT_V3_RW=1, else every call is the v3_rm one
        for f in ("v3_rw.h", "v3_rw.cpp"):
            shutil.copy2(HERE / "contract_v3" / f, d / f)
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

        # pathb-nodescale: the image reader (new file) and KT_TP_PART_BLOCKS in his moe-tp.hpp (mandatory edits)
        if (kt / "operators/avx2/kt_image.hpp").exists():
            raise SystemExit("operators/avx2/kt_image.hpp already exists in the tree")
        shutil.copy2(HERE / "operators/avx2/kt_image.hpp", kt / "operators/avx2/kt_image.hpp")
        report["replaced"]["operators/avx2/kt_image.hpp"] = {"from": None, "to": sha(kt / "operators/avx2/kt_image.hpp")}
        mt = kt / "operators/moe-tp.hpp"
        mt_before = sha(mt)
        mt_src = mt.read_text()
        mt_a0 = "template <MOE_TP_PART T, typename Concrete = T>\nclass TP_MOE_Common : public MoE_Interface {\n"
        mt_b0 = (
            "// pathb-nodescale (concept-r1 NODE_SCALING_r1.md section 3, 14 Nov): KT_TP_PART_BLOCKS=\"b0,b1,...\" gives NUMA part i\n"
            "// b_i blocks of the part type's KT_PART_GRANULE intermediate columns, in part order. Unset or empty: the split below,\n"
            "// unchanged. Set: one positive count a part, summing to the layer's blocks, and a part type that declares\n"
            "// KT_PART_GRANULE; anything else refuses (std::runtime_error) when the layer is made.\n"
            "inline std::vector<int> kt_part_blocks_table(int tp_count) {\n"
            "  std::vector<int> out;\n"
            "  const char* env = std::getenv(\"KT_TP_PART_BLOCKS\");\n"
            "  if (!env || !*env) return out;\n"
            "  const std::string s(env);\n"
            "  size_t i = 0;\n"
            "  while (true) {\n"
            "    const size_t j = s.find(',', i);\n"
            "    const std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);\n"
            "    if (tok.empty() || tok.size() > 6 || tok.find_first_not_of(\"0123456789\") != std::string::npos || std::stoi(tok) <= 0)\n"
            "      throw std::runtime_error(\"KT_TP_PART_BLOCKS: \\\"\" + s + \"\\\" is not a comma-separated list of positive integers\");\n"
            "    out.push_back(std::stoi(tok));\n"
            "    if (j == std::string::npos) break;\n"
            "    i = j + 1;\n"
            "  }\n"
            "  if ((int)out.size() != tp_count)\n"
            "    throw std::runtime_error(\"KT_TP_PART_BLOCKS: \" + std::to_string(out.size()) + \" entries for \" + std::to_string(tp_count) +\n"
            "                             \" NUMA parts\");\n"
            "  return out;\n"
            "}\n\n" + mt_a0)
        mt_a1 = ("    tp_count = config.pool->config.subpool_count;\n"
                 "    if (config.intermediate_size % tp_count != 0) {\n")
        mt_b1 = ("    tp_count = config.pool->config.subpool_count;\n"
                 "    const std::vector<int> kt_part_tbl = kt_part_blocks_table(tp_count);   // pathb-nodescale KT_TP_PART_BLOCKS\n"
                 "    if (kt_part_tbl.empty() && config.intermediate_size % tp_count != 0) {\n")
        mt_a2 = "    if constexpr (is_llamafile || has_tp_block) {\n"
        mt_b2 = ("    if (!kt_part_tbl.empty()) {   // pathb-nodescale: the part sizes from KT_TP_PART_BLOCKS\n"
                 "      if constexpr (requires { Concrete::KT_PART_GRANULE; }) {\n"
                 "        constexpr int gran = Concrete::KT_PART_GRANULE;\n"
                 "        long blocks = 0;\n"
                 "        for (int b : kt_part_tbl) blocks += b;\n"
                 "        if (blocks * gran != (long)config.intermediate_size)\n"
                 "          throw std::runtime_error(\"KT_TP_PART_BLOCKS: \" + std::to_string(blocks) + \" blocks of \" + std::to_string(gran) +\n"
                 "                                   \" columns are not the layer's intermediate size \" + std::to_string(config.intermediate_size));\n"
                 "        std::string kt_rows;\n"
                 "        for (int i = 0; i < tp_count; i++) {\n"
                 "          tps.push_back(nullptr);\n"
                 "          GeneralMOEConfig tp_config = config;\n"
                 "          tp_config.intermediate_size = kt_part_tbl[i] * gran;\n"
                 "          tp_configs.push_back(tp_config);\n"
                 "          kt_rows += (i ? \",\" : \"\") + std::to_string(tp_config.intermediate_size);\n"
                 "        }\n"
                 # The constructor shares unbuffered stdout with the image loader. Keep the newline in this printf.
                 "        printf(\"[kt-parts] layer %d: KT_TP_PART_BLOCKS: %d parts of %s rows (granule %d)\\n\", config.layer_idx, tp_count,\n"
                 "               kt_rows.c_str(), gran);\n"
                 "      } else {\n"
                 "        throw std::runtime_error(\"KT_TP_PART_BLOCKS is set, but this backend does not take parts of different sizes\");\n"
                 "      }\n"
                 "    } else if constexpr (is_llamafile || has_tp_block) {\n")
        counts = [mt_src.count(a) for a in (mt_a0, mt_a1, mt_a2)]
        report["moe_tp"] = {"before": mt_before, "made_from": "263d40c173e60ad69afd2406723081a381d53c703af400eaef94311c7ee8ff40",
                            "anchor_counts": counts}
        if mt_before != report["moe_tp"]["made_from"]:   # fail closed: the edit is only known good on that file
            save()
            raise SystemExit("operators/moe-tp.hpp in the tree is not the file the KT_TP_PART_BLOCKS edit was made from: "
                             f"sha256 {mt_before}, want {report['moe_tp']['made_from']}")
        if counts != [1, 1, 1]:
            save()
            raise SystemExit(f"operators/moe-tp.hpp: KT_TP_PART_BLOCKS anchors not unique {counts}")
        mt.write_text(mt_src.replace(mt_a0, mt_b0).replace(mt_a1, mt_b1).replace(mt_a2, mt_b2))
        report["moe_tp"].update(after=sha(mt), same_base=mt_before == report["moe_tp"]["made_from"])

        # pathb-cpu-r1: KT_NUMA_WAIT_PAUSE in his worker_pool.cpp (do_numa_job's wait), by two anchored edits applied only
        # when both anchors are unique; otherwise the file stays as it is and the knob does nothing (recorded here)
        wp = kt / "cpu_backend/worker_pool.cpp"
        wp_src = wp.read_text()
        wp_a1 = ('static int kt_spin_pauses() {\n  // kt-kernel-live-knobs.py: settable while running\n'
                 '  static std::atomic<int>& n = kt_knob("KT_SPIN_PAUSES", 64);\n  return n.load(std::memory_order_relaxed);\n}\n')
        wp_b1 = wp_a1 + ('\n// pathb-cpu-r1 (concept-r1 ZEN3_OPTIMISATION_r1.md section 2.11): KT_NUMA_WAIT_PAUSE > 0 puts a PAUSE in\n'
                         '// do_numa_job\'s wait for the distributors (a live knob, read once a job); 0 (the default): the wait as before\n'
                         'static int kt_numa_wait_pause() {\n'
                         '  static std::atomic<int>& n = kt_knob("KT_NUMA_WAIT_PAUSE", 0);\n'
                         '  return n.load(std::memory_order_relaxed);\n}\n')
        wp_a2 = ('  for (int i = 0; i < numa_count; i++) {\n'
                 '    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {\n'
                 '    }\n  }\n}\n#endif\n')
        wp_b2 = ('  const bool kt_wait_pause = kt_numa_wait_pause() > 0;   // pathb-cpu-r1 KT_NUMA_WAIT_PAUSE\n'
                 '  for (int i = 0; i < numa_count; i++) {\n'
                 '    while (status[i]->load(std::memory_order_acquire) == ThreadStatus::WORKING) {\n'
                 '      if (kt_wait_pause) _mm_pause();\n'
                 '    }\n  }\n}\n#endif\n')
        report["worker_pool"] = {"before": sha(wp), "expected_before": "e2840080910437e1ad98e0ca3466749a1a7c5aadbf8a8f148674814581d1f704",
                                 "anchor_counts": [wp_src.count(wp_a1), wp_src.count(wp_a2)]}
        if wp_src.count(wp_a1) == 1 and wp_src.count(wp_a2) == 1:
            wp.write_text(wp_src.replace(wp_a1, wp_b1).replace(wp_a2, wp_b2))
            report["worker_pool"].update(patch="applied", after=sha(wp))
        else:
            report["worker_pool"].update(patch="skipped: anchors not unique; KT_NUMA_WAIT_PAUSE has no effect in this module")

        cm = kt / "CMakeLists.txt"
        replace_once(cm, 'option(CPUINFER_ENABLE_LTO ',
                     'option(KT_ENABLE_CONTRACT_V3 "Build the contract v3 (spec r3, tiers C/D) AVX2 MoE backend; runtime opt-in KT_MXFP4_BACKEND=contract_v3" ON)\n'
                     'option(CPUINFER_ENABLE_LTO ')
        cm.write_text(cm.read_text() + (
            '\n# path B: contract v3 (spec r3) AVX2 MoE backend\n'
            'if(KT_ENABLE_CONTRACT_V3)\n'
            '    set(KT_V3_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/operators/avx2/contract_v3/v3_rm.cpp\n'
            '                      ${CMAKE_CURRENT_SOURCE_DIR}/operators/avx2/contract_v3/v3_rw.cpp\n'
            '                      ${CMAKE_CURRENT_SOURCE_DIR}/operators/avx2/contract_v3/v3_silu_table.cpp)\n'
            '    target_sources(${PROJECT_NAME} PRIVATE ${KT_V3_SOURCES})\n'
            '    # the v3 arithmetic must not see -ffast-math: these flags come last\n'
            '    set_source_files_properties(${KT_V3_SOURCES} PROPERTIES\n'
            '        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-fno-reciprocal-math;-fno-unsafe-math-optimizations;-mavx2;-mfma")\n'
            '    target_compile_definitions(${PROJECT_NAME} PRIVATE KT_ENABLE_CONTRACT_V3=1)\n'
            '    message(STATUS "contract v3 MoE backend: enabled (runtime opt-in KT_MXFP4_BACKEND=contract_v3)")\n'
            'endif()\n'))

        # pathb-image-r1: the arena format of this tree (every writer edit is done by now), compiled in below
        fmt, fmt_files, fmt_w = arena_format(kt)
        report["arena_format"] = {"format": fmt, "writer_sha256": fmt_w, "files": fmt_files}
        if not all(c.isalnum() or c in ";=-" for c in fmt):
            raise SystemExit("arena format string has characters that cannot go in a C++ literal: " + fmt)
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
                     '      // pathb-nodescale: the layer from the block-ordered expert image, as a CPUInfer task (map as load_weights_task)\n'
                     '      struct ImgArgs { CPUInfer* cpuinfer; MoeClass* moe; };\n'
                     '      static void img_inner(void* p) {\n'
                     '        ImgArgs* a = (ImgArgs*)p;\n'
                     '        a->cpuinfer->enqueue(&MoeClass::load_weights_image, a->moe);\n'
                     '      }\n'
                     '      static std::pair<intptr_t, intptr_t> img_task(std::shared_ptr<MoeClass> moe, const uintptr_t map) {\n'
                     '        ImgArgs* a = new ImgArgs{nullptr, moe.get()};\n'
                     '        if (map) moe->config.physical_to_logical_map = reinterpret_cast<void*>(map);\n'
                     '        return std::make_pair((intptr_t)&img_inner, (intptr_t)a);\n'
                     '      }\n'
                     "      // pathb-image-r3: the layer's module made, prepared and read from the image in ONE CPUInfer task\n"
                     '      // (MoeClass::ImageBuild, contract_v3h-moe.hpp), so every do_numa_job of a preloaded load runs on the worker;\n'
                     '      // the queued task holds the record (a shared_ptr), so a dropped Python handle cannot free it under the task\n'
                     '      using ImgBuild = typename MoeClass::ImageBuild;\n'
                     '      struct BuildArgs { CPUInfer* cpuinfer; std::shared_ptr<ImgBuild>* b; };\n'
                     '      static void build_inner(void* p) {\n'
                     '        BuildArgs* a = (BuildArgs*)p;\n'
                     '        std::shared_ptr<ImgBuild> b = std::move(*a->b);\n'
                     '        CPUInfer* ci = a->cpuinfer;\n'
                     '        delete a->b;\n'
                     '        delete a;\n'
                     '        ci->task_queue_->enqueue([b]() { b->run(); });\n'
                     '      }\n'
                     '      static std::pair<intptr_t, intptr_t> build_task(std::shared_ptr<ImgBuild> b) {\n'
                     '        if (b->submitted_.exchange(1)) throw std::runtime_error("image build: its task was made already");\n'
                     '        return std::make_pair((intptr_t)&build_inner, (intptr_t) new BuildArgs{nullptr, new std::shared_ptr<ImgBuild>(b)});\n'
                     '      }\n'
                     '      static std::shared_ptr<ImgBuild> build(const GeneralMOEConfig& cfg, const std::string& path, uint64_t nbytes,\n'
                     '                                             const std::vector<int64_t>& geometry, const std::string& digests,\n'
                     '                                             const std::vector<std::vector<int>>& rows, bool any_rows, uintptr_t map) {\n'
                     '        return std::make_shared<ImgBuild>(cfg, path, nbytes, geometry, digests, rows, any_rows, map);\n'
                     '      }\n'
                     '    };\n'
                     '    moe_cls.def("forward_partials_task", &ContractV3PartialBindings::task)\n'
                     '        .def("forward_partials", &MoeClass::forward_partials_binding)\n'
                     '        .def("set_bl", &MoeClass::set_bl)\n'
                     '        .def("get_bl", &MoeClass::get_bl)\n'
                     '        .def("contract_v3_tier", &MoeClass::contract_v3_tier)\n'
                     '        .def("contract_v3_spec", &MoeClass::contract_v3_spec)\n'
                     '        .def("contract_v3_silu_sha256", &MoeClass::contract_v3_silu_sha256)\n'
                     '        .def("contract_v3_arena_scale_bytes", &MoeClass::contract_v3_arena_scale_bytes)\n'
                     '        .def("contract_v3_codes_fnv", &MoeClass::contract_v3_codes_fnv)\n'
                     '        .def("contract_v3_codes_placement", &MoeClass::contract_v3_codes_placement)\n'
                     '        .def("contract_v3_set_clean", &MoeClass::contract_v3_set_clean)\n'
                     '        .def("contract_v3_clean", &MoeClass::contract_v3_clean)\n'
                     '        .def("contract_v3_set_preread", &MoeClass::contract_v3_set_preread)\n'
                     '        .def("contract_v3_preread", &MoeClass::contract_v3_preread)\n'
                     '        .def("contract_v3_preread_log", &MoeClass::contract_v3_preread_log)\n'
                     '        .def("contract_v3_preread_hint", &MoeClass::contract_v3_preread_hint)\n'
                     '        .def("contract_v3_tq_last_qwait_ns", &MoeClass::contract_v3_tq_last_qwait_ns)\n'
                     '        .def("contract_v3_set_ccd", &MoeClass::contract_v3_set_ccd)\n'
                     '        .def("contract_v3_ccd", &MoeClass::contract_v3_ccd)\n'
                     '        .def("contract_v3_set_rw", &MoeClass::contract_v3_set_rw)\n'
                     '        .def("contract_v3_rw", &MoeClass::contract_v3_rw)\n'
                     '        .def("contract_v3_thp", &MoeClass::contract_v3_thp)\n'
                     '        .def("contract_v3_sum_part", &MoeClass::contract_v3_sum_part)\n'
                     '        .def("load_weights_image_task", &ContractV3PartialBindings::img_task)\n'
                     '        .def("image_prepare", &MoeClass::image_prepare)\n'
                     '        .def("image_status", &MoeClass::image_status)\n'
                     '        .def("image_stats", &MoeClass::image_stats)\n'
                     '        .def("image_done", &MoeClass::image_done)   // pathb-image-r2\n'
                     '        .def("contract_v3_part_rows", &MoeClass::contract_v3_part_rows)\n'
                     '        .def_static("image_build", &ContractV3PartialBindings::build)   // pathb-image-r3\n'
                     '        // pathb-image-r1: the arena format this module writes (build_bs.py ARENA_WRITERS)\n'
                     '        .def_static("kt_arena_format", []() { return std::string("' + fmt + '"); });\n'
                     '    // pathb-image-r3: the build record (one per preloaded layer)\n'
                     '    using ImgBuildT = typename ContractV3PartialBindings::ImgBuild;\n'
                     '    py::class_<ImgBuildT, std::shared_ptr<ImgBuildT>>(moe_module, "AVX2ContractV3_ImageBuild")\n'
                     '        .def("task", &ContractV3PartialBindings::build_task)\n'
                     '        .def("done", &ImgBuildT::done)\n'
                     '        .def("status", &ImgBuildT::status)\n'
                     '        .def("rows", &ImgBuildT::rows)\n'
                     '        .def("seconds", &ImgBuildT::seconds)\n'
                     '        .def("tid", &ImgBuildT::tid)\n'
                     '        .def("take", &ImgBuildT::take);\n'
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
        # pathb-nodescale: the image hook at the top of NativeMoEWrapper.load_weights, and its functions appended
        replace_once(amx, IMAGE_ANCHOR, IMAGE_HOOK)
        replace_once(amx, IMAGE_INIT_ANCHOR, IMAGE_INIT_HOOK)   # pathb-image-r2
        amx.write_text(amx.read_text() + (HERE / "python/utils/kt_image.py").read_text())
        report["after"] = {str(p.relative_to(kt)): sha(p) for p in patched}
        report["added"] = {str(p.relative_to(kt)): sha(p) for p in sorted(d.iterdir())}
        report["added"]["operators/avx2/contract_v3h-moe.hpp"] = sha(kt / "operators/avx2/contract_v3h-moe.hpp")
        report["cpu_r1"] = {"dir": str(HERE), "pinned_header_sha256": sha(CPU / "contract_v3h-moe.hpp"),
                               "commit": (HERE / "SOURCE_COMMIT").read_text().strip() if (HERE / "SOURCE_COMMIT").is_file() else None,
                               "preread": True, "bytescales": True, "pairfix": True, "stagger_r2_per_task": True,
                               "nodescale": True, "image_r1": True, "image_r2": True, "image_r3": True,
                               "kt_image_py_sha256": sha(HERE / "python/utils/kt_image.py")}
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
        # pathb-image-r1: the image hook ships in the package (its utils/amx.py), or the build fails
        pamx = N / "pkg/kt_kernel/utils/amx.py"
        if not pamx.is_file():
            raise SystemExit("the wheel has no kt_kernel/utils/amx.py")
        ptxt = pamx.read_text()
        if ptxt.count(IMAGE_HOOK) != 1 or ptxt.count(IMAGE_INIT_HOOK) != 1 or \
                not ptxt.endswith((HERE / "python/utils/kt_image.py").read_text()):
            raise SystemExit("the wheel's kt_kernel/utils/amx.py does not carry the two image hooks and kt_image.py")
        report["pkg_amx_py_sha256"] = sha(pamx)
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
        # pathb-image-r1: the format record beside the module (the imgid job and the preflight check read it)
        (N / "ARENA_FORMAT.json").write_text(json.dumps(dict(report["arena_format"], module_sha256=report["module_sha256"],
                                                             pkg_amx_py_sha256=report["pkg_amx_py_sha256"]), indent=1) + "\n")
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
