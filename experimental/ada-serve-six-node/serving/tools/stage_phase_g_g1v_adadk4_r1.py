"""CPU-only stage of the Ada serving gate (Claude, 31 Oct 2026): the tree r6 gate on the arm 0 Route 3 path, prefill path A on, staggered spill, round worker.

CPU-only stage of the host-reload gate on tree r6 (Claude, 30 Oct 2026): the r6 identity stage with the accepted A3 serving arguments
and cache policy, so the frozen feature client runs on the final tree with prefill path A on.

CPU-only stage of the tree r6 identity load (Claude, 30 Oct 2026): selectors as G1p left them, tree r6 (TP2 draft and the
all-reduce repair, both off) and the G0 rebuild of the KT native library with the four C1 Python files. All at once to save a load; any
difference from arm 2 is then split apart.

CPU-only stage of Phase G G1p (g1pon) (Claude, 30 Oct 2026): the arm-2 performance stage on tree r5 with the prefill
path A selector on, the wrapper's pinned sources in packages/patha_prefill and the component pin.

CPU-only stage of the Phase G G1v arm 2 PERFORMANCE load (Claude, 30 Oct 2026): the arm-2 numerical stage with
the routing-dump instrumentation removed, as C1's performance stage did, and the repeated-performance client.
r3: integrated tree r3 (whole single-route attention call row-blocked; r2 round 1 hit a 4.27 GiB OOM in fixed wo_a).
r5: integrated tree r4 (fixed wo_a's [4,m,g,n] buffer row-blocked on the single route; r3 round 1 still fragmented) and 4K rows.

CPU-only stage of Phase G G1v arm 2 (Claude, 30 Oct 2026): the arm-0 r3 stage with the integrated G1 tree
(G0 + Codex's G1a r2 path A dense + G1e r1 single route, integrate_g1_r1.py), G0's dual-target path A binaries and the
indexer prefill identity receipt copied into the trial, Route 3 and the fixed split-K off, and SGLANG_FP8_PATHA_DENSE,
SGLANG_SINGLE_ROUTE on. Everything else is the accepted C1-on profile. Arm-0 docstring follows.

CPU-only stage of Phase G G1v arm 0: the G0 tree with the accepted C1-on profile unchanged.

Claude (Opus 5.5), 30 Oct 2026, taking over from Codex. Derived from stage_r19_C1_static_identity_r2.py.
Copies the accepted C1-on trial, then swaps in exactly the G0 pieces: the repaired SGLang tree, G0's
kt_kernel package, G0's rebuild of the C1 expert binary, the env-r3 interpreter and the new recipe's
engram adapter (one method rename that tracks SGLang's engram.py). Every other byte, flag and argument
stays as accepted. No model, no CUDA, no change to C1 evidence or the G0 trees.

r2 (same day): r1 stopped in the pre-weights profile because G0's rebuilt kt_kernel package lacks three
fea01-local Python changes that C1 serves and that neither the recipe nor the fork package carries: the
recipe GGUF expert reader (utils/recipe_gguf_reader.py, recipe_gguf_storage.py and amx.py's loader branch),
the KT_HANDOFF_PACK packer and the max_len guard in experts_base.py. KTransformers' pin did not move in the
port, so r2 keeps C1's whole kt_kernel package (Python and native) and tests the SGLang port alone. The G0
KT rebuild is a separate packaging item for the 5070 Ti build. r1's namespace is kept as evidence.

r3: r2 loaded and settled, then stopped in the unchanged server-info parser because SGLang 5d8e90a0 adds
one metadata key (hicache_object_layout, hierarchical cache disabled). The controller now uses
server_info_profile_g0_r1.py, which accepts only that key in its disabled form. r2 is kept as evidence.
"""
from pathlib import Path
import fcntl
import os
import shutil
import subprocess
import sys

B = Path('/scratch/aster/bench')
P = B / 'six-node-performance-r1'
R = B / 'fp32-subtotals-model-r1'
sys.path[:0] = [str(P / 'tools'), str(R), str(R / 'workflow')]
from trial_common import read, save, sha, inventory, require
from r19_E1_profile_r1 import POLICY_KEYS, C1_KEYS
import module_origins_fast_r1 as origins

F = B / 'feature-trial-B-r19-c1-static-on-identity-six-r2'
A3 = B / 'feature-trial-B-r19-swa-numerics-six-r1'
T = B / 'ada-gate-disk-20261102-r5'
F_PREPARED = 'R19_C1_STATIC_ON_IDENTITY_PREPARED_r2.json'
PREPARED = 'PHASE_G_G1V_ADADK4_PREPARED_r1.json'
G0 = B / 'phase-g-g0-repair-20261030-r1'
G0_SRC = B / 'phase-g-g1-integrated-20261030-r6/src/sglang'   # arm 2: the integrated G1 tree
INTEGRATED = B / 'phase-g-g1-integrated-20261030-r6/INTEGRATION_RESULT.json'
INTEGRATED_INVENTORY = 'b71f046342cecba077034ddbd2dea2f6ff6ef2d191c78a6cc1a12c42052a32b7'
KT_SO = 'kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so'
G1P_PIN = B / 'phase-g-g1p-component-20261030-r1/G1P_NATIVE_PIN.json'
PATHA_SO = dict(decode=('patha_decode_g0_dual_s2', 'decode-gemv/patha_decode_g0_dual_s2.so', '9a79de77b588b7f3d84576240ba4d10cb9d8260c8616aae770d845697576e026'),
                prefill=('patha_prefill_g0_dual', 'prefill-imma/patha_prefill_g0_dual.so', '85ea9dee9876cb62524c348149427f8524be1c95487e3b5f267b09cbafe34e6c'))
INDEXER_RECEIPT = B / 'phase-g-g1a-gpu-component-20261030-r1/out/INDEXER_PREFILL_IDENTITY.json'
INDEXER_RECEIPT_SHA = '6f6c4059648d9039'
G0_NATIVE = B / 'phase-g-g0-native-20261030-r1'
ENV = B / 'phase-g-g0-environment-20261030-r3/venv'
NEW_ADAPTER = B / 'phase-g-g0-sources-20261030-r1/src/recipe/engram-adapter'
OLD_PYTHON = '/scratch/aster/bench/05yuki-main-00fd12a-candidate/.venv/bin/python'
OLD_INCLUDE = '/scratch/aster/bench/05yuki-main-00fd12a-candidate/.venv/lib/python3.12/site-packages/nvidia/cu13/include'
NEW_INCLUDE = str(ENV / 'lib/python3.12/site-packages/nvidia/cu13/include')
EXPERT = 'patha_expert_static_sm89_e0_s3_b8_t0.so'
PINS = dict(
    g0_source_stack='03a2a10fd63a61ae765b3f15365be167181f826900ac975aff0f4069d0920f4f',
    kt_ext='fd5072c3832a15abec609cbc238cfb127754131a257c0b7fcaca2900098e6751',
    experts_base='f24ef260d9cd3be88837a1798166f42056b09e9c59cf07841b1a0a680b0b3d02',
    expert_so='4d44ef332bfd9aba8cdb963225fd55ec9c0b2b5281f42adcc93fed92a3e08f61',
    expert_py='6b215f8a669d6f85abbdd17991e316e4455991c69d141d424d1d2ad6ca83ed75',
    route3_so='51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649',
    swa='f2efb9b5cac3a794efedd784f951e6eabff74480f14b09e8c1a91a19d9d6c5a5',
    host_reload='eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4')
ADAPTER_FROM_NEW = ('engram_backend.py', 'row_store.cpp', 'test_adapter_cpu.py', 'verify_shard_manifest.py')
TOOLS = ('stage_phase_g_g1v_adadk4_r1.py', 'controller_phase_g_g1v_adadk4_r1.py', 'launch_phase_g_g1v_adadk4_r1.py',
         'profile_phase_g_g1v_adadk4_r1.py', 'client_ada_adadk4_r1.py', 'client_fast_r1.py', 'compare_r19_exact_scores_r1.py',
         'server_info_profile_g0_r3.py', 'r19_E1_profile_r1.py', 'client_r19_B1b_logical_r2.py', 'client_r19_bitneutral_identity_r1.py')

cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
require(int((cg / 'memory.max').read_text()) == 8 * 2**30, 'stager memory cap differs')
require((cg / 'memory.swap.max').read_text().strip() == '0', 'stager swap differs')
require(os.sched_getaffinity(0) == {23} and os.environ['CUDA_VISIBLE_DEVICES'] == '', 'stager placement differs')

with (B / '.bench.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU busy')
    require(not T.exists(), 'arm 0 namespace exists; never overwrite evidence')

    # The accepted C1-on trial, as its own controller left it.
    p = read(F / F_PREPARED)
    require(p['status'] == 'PREPARED_NO_RUNTIME' and p['trial'] == str(F), 'C1 manifest differs')
    require(sha(F / 'LAUNCH.json') == p['launch_sha256'], 'C1 launch changed')
    for name, wanted in p['pinned_files'].items():
        require(sha(F / name) == wanted, 'C1 pinned file changed: ' + name)
    for root, key in [('recipe/third_party/sglang', 'source_after'), ('packages', 'native_after'),
                      ('recipe/engram-adapter', 'adapter_after'), ('client-tools', 'client_sha256')]:
        require(inventory(F / root) == p[key], 'C1 tree changed: ' + root)
    run = read(F / 'RUN-r19-c1-static-on-identity-r2.json')
    require(run['status'] == 'R19_BITNEUTRAL_IDENTITY_PASS_TIMING_NOT_RUN' and run['cleanup']['empty'], 'C1 run not accepted')
    require(not run.get('cleanup_errors') and not run.get('post_cleanup_errors'), 'C1 cleanup errors')

    # The G0 pieces.
    require(sha(G0 / 'SOURCE_STACK.json') == PINS['g0_source_stack'], 'G0 source stack differs')
    require(read(G0 / 'REPAIR_RESULT.json')['status'] == 'G0_REPAIRED_SOURCE_IMPORTS_PASS', 'G0 repair not passed')
    g0_inventory = inventory(G0_SRC)
    integration = read(INTEGRATED)
    require(integration['status'] == 'G1_INTEGRATED_SOURCE_CPU_ONLY' and integration['inventory_sha256'] == INTEGRATED_INVENTORY, 'integration receipt differs')
    require(len(g0_inventory) == integration['files'], 'integrated tree size differs')
    require(sha(INDEXER_RECEIPT).startswith(INDEXER_RECEIPT_SHA) and read(INDEXER_RECEIPT)['status'] == 'PASS', 'indexer receipt differs')
    for module, rel, digest in PATHA_SO.values():
        require(sha(G0_NATIVE / 'build' / rel) == digest, 'path A binary differs: ' + rel)
    require(g0_inventory['python/sglang/srt/mem_cache/unified_cache/components/swa.py'] == PINS['swa'], 'G0 swa differs')
    require(sha(G0_NATIVE / 'build/expert-static-r3' / EXPERT) == PINS['expert_so'], 'G0 expert binary differs')
    require(Path(OLD_INCLUDE).is_dir() and Path(NEW_INCLUDE).is_dir(), 'include roots missing')
    for name in ADAPTER_FROM_NEW:
        old = (F / 'recipe/engram-adapter' / name).read_bytes().replace(b'\r\n', b'\n')
        new = (NEW_ADAPTER / name).read_bytes()
        if name == 'engram_backend.py':
            require(old.replace(b'self._empty(indices)', b'self.new_rows(indices)') == new, 'adapter change is not the rename')
        else:
            require(old == new, 'adapter differs beyond line endings: ' + name)
    canonical = (NEW_ADAPTER / 'sitecustomize.py').read_bytes()
    accepted_site = (F / 'recipe/engram-adapter/sitecustomize.py').read_bytes()
    require(accepted_site.replace(b'\r\n', b'\n') == origins.sitecustomize_bytes(canonical, sha(NEW_ADAPTER / 'sitecustomize.py')),
            'accepted sitecustomize is not canonical plus the observer')

    # Stage.
    T.mkdir()
    shutil.copytree(G0_SRC, T / 'recipe/third_party/sglang', symlinks=True)
    source_after = inventory(T / 'recipe/third_party/sglang')
    require(source_after == g0_inventory, 'G0 copy differs')
    # Candidate overlay: dynamic rounds on a worker thread behind KT_KV_HOT_DYN_WORKER.
    H = T / 'recipe/third_party/sglang/python/sglang/srt/layers/moe/kt_kv_hot_lend.py'
    require(sha(H) == '09be733debc39f7597f83740a6d485963482f46ff44b01b6fea829dd3a9b2a55', 'hot lend differs')
    shutil.copyfile(P / 'ada-spill-r1/layer-quota-r1/kt_kv_hot_lend.py', H)
    require(sha(H) == 'c96c434afd1d48e6b6aafcd84ec834df9f4d8dd0afcee4e30b6a3064457ee762', 'round worker differs')
    # Serving overlay: a streaming session's virtual node never reaches the L3 prefetch. Only the disk tier reaches this line.
    SC = T / 'recipe/third_party/sglang/python/sglang/srt/managers/scheduler.py'
    require(sha(SC) == '1c2483041a7c2b626af6556353e0e8b4bd09bbd1af34a70e23060c7371b88a15', 'scheduler differs')
    shutil.copyfile(P / 'ada-spill-r1/serve-disk-r1/scheduler.py', SC)
    require(sha(SC) == 'd393f2967462ac9f36d4ad26e2a6a76610211e30cab5ed7c9825e6af19cb1cd1', 'scheduler overlay differs')
    # Flush option for the disk tier: keep_storage, and SGLANG_FLUSH_CLEARS_STORAGE for gate chains.
    HS = T / 'recipe/third_party/sglang/python/sglang/srt/entrypoints/http_server.py'
    require(sha(HS) == 'd5d7ae6c5e8f403f9a46617377ab96977d9d4f1f263c0b90fc9fd2769eef061e', 'http server differs')
    shutil.copyfile(P / 'ada-spill-r1/serve-disk-r1/http_server.py', HS)
    require(sha(HS) == '7537f80b7c94cb2f279be110d9198360fdbb9d6e6d5ecc1e1d798dee986027fb', 'http server overlay differs')
    # Candidate overlay: the fused route glue behind KT_CONTRACT_V2_GLUE_FUSED (default off in the files, on in this profile).
    G = T / 'recipe/third_party/sglang/python/sglang/srt/layers/moe/contract_v2'
    require(sha(G / 'adapter.py') == '8037ce074a7b3531ee04d641762b29113d051c83eea1853a98bab6795d82f3af', 'adapter differs')
    shutil.copyfile(P / 'ada-spill-r1/glue-fused-r1/adapter.py', G / 'adapter.py')
    shutil.copyfile(P / 'ada-spill-r1/glue-fused-r1/glue_fused.py', G / 'glue_fused.py')
    require(sha(G / 'adapter.py') == '01beed62fdf94920144c42b939546b4b4612c6c44ba70ec1f13f9045e23687c8' and sha(G / 'glue_fused.py') == '20f13ac58d4e0cf09a5a384981bb3f3047e645e3556fb1f9f0dcbef05b599d48', 'glue overlay differs')
    # Candidate overlay: the wo_a CUDA kernel behind SGLANG_DSV41_FIXED_WO_A_CUDA (default off in the file, on in this profile).
    W = T / 'recipe/third_party/sglang/python/sglang/kernels/ops/gemm/fixed_wo_a_fp8.py'
    require(sha(W) == '75e5aaecfae546c68e1cf1278a8ab91cbb602af02b51e62ece8aa81b49c51458', 'fixed wo_a differs')
    shutil.copyfile(P / 'ada-spill-r1/wo-a-cuda-r1/fixed_wo_a_fp8.py', W)
    require(sha(W) == 'd8d6ac4d13941c34ca729e847a8346d9b995273dfa187c3d07bb89f2829ab778', 'wo_a overlay differs')
    # Candidate overlay: live top-k at the served capacity, KT_IDX_LIVE_TOPK_MAX_L in this profile, live -inf dropped among blocks as torch does.
    LT = T / 'recipe/third_party/sglang/python/sglang/kernels/ops/attention/dsv4/live_topk.py'
    require(sha(LT) == '9ca99b15b14d5b7e0176a509ee75cf7a3119f2d737b16f429acadc9a806c510a', 'live top-k differs')
    shutil.copyfile(P / 'ada-spill-r1/topk-live-r1/live_topk.py', LT)
    require(sha(LT) == 'd6d616217b735d07d59a4e63f2dd595b8f26607e025af38708047721d8c64236', 'live top-k overlay differs')
    # Candidate overlay: the KV lend reaches every pool tensor laid out by the allocator's pages (05yuki 7fef854e), KT_PREFILL_KV_LEND_ALL_TENSORS in this profile.
    KL = T / 'recipe/third_party/sglang/python/sglang/srt/layers/moe/kt_kv_prefill_lend.py'
    require(sha(KL) == '7c8389adae34f2778b1ce36bf8cd7b1093108675099c68b7086a57d5ba68c688', 'KV lend differs')
    shutil.copyfile(P / 'ada-spill-r1/kv-lend-all-r1/kt_kv_prefill_lend.py', KL)
    require(sha(KL) == '281ce3141d6075aab77a2ff6d2a98520902418f32ae1bf3aacd9e634c4f33d64', 'KV lend overlay differs')
    # Candidate overlay: the unsplit gather as three staged kernels, the same bits, SGLANG_SM89_ATTN_STAGED in this profile.
    SD = T / 'recipe/third_party/sglang/python/sglang/kernels/ops/attention/sm89_dual_scope.py'
    require(sha(SD) == 'ad723c6495a06af27caf0e024e77d0d24dbc3c1f693a6aeba1bc35dc93f4fd13', 'SM89 attention differs')
    shutil.copyfile(P / 'ada-spill-r1/gather-r1/sm89_dual_scope.py', SD)
    require(sha(SD) == '75d14867db6194c3e46280cfb247ca752de1855ce60df6752baa3c086eb56ad6', 'SM89 attention overlay differs')
    shutil.copyfile(P / 'ada-spill-r1/gather-r1/gather_staged.served.py', SD.with_name('sm89_gather_staged.py'))
    require(sha(SD.with_name('sm89_gather_staged.py')) == 'f7b9cb17970c081edbf3842c4a7d70f936746eff8fffb6dfbc5444fa80689fe1', 'staged gather differs')
    source_after = inventory(T / 'recipe/third_party/sglang')
    adapter = T / 'recipe/engram-adapter'
    adapter.mkdir()
    for name in ADAPTER_FROM_NEW:
        shutil.copyfile(NEW_ADAPTER / name, adapter / name)
    for name in ('librow_store.so', 'module_origins.py'):
        shutil.copyfile(F / 'recipe/engram-adapter' / name, adapter / name)
    # Candidate overlay: a call's Engram misses in one io_submit, polled, behind DSV41_ENGRAM_AIO in this profile.
    shutil.copyfile(P / 'ada-spill-r1/engram-aio-r1/row_store.cpp', adapter / 'row_store.cpp')
    shutil.copyfile(P / 'ada-spill-r1/engram-aio-r1/librow_store.so', adapter / 'librow_store.so')
    require(sha(adapter / 'librow_store.so') == '45850492306cca3830a3832927f4f7b2a7ee63e91f27d1b72fdff74d64b8f237', 'row store differs')
    (adapter / 'sitecustomize.py').write_bytes(origins.sitecustomize_bytes(canonical, sha(NEW_ADAPTER / 'sitecustomize.py')))
    require(sha(adapter / 'module_origins.py') == sha(R / 'module_origins_fast_r1.py'), 'observer copy differs')
    shutil.copytree(F / 'packages', T / 'packages', ignore=lambda d, names: [n for n in names if Path(d) == F / 'packages' and n == EXPERT])
    shutil.copyfile(G0_NATIVE / 'build/expert-static-r3' / EXPERT, T / 'packages' / EXPERT)
    shutil.copyfile(B / 'ada-native-r16-20261101-r1/kt-build' / Path(KT_SO).name, T / 'packages' / KT_SO)
    shutil.copyfile(P / 'ada-spill-r1/wo-a-cuda-r1/build/wo_a_parts_b4s3.so', T / 'packages/wo_a_parts_b4s3.so')
    shutil.copyfile(P / 'ada-spill-r1/wo-a-cuda-r1/wo_a_parts.py', T / 'packages/wo_a_parts.py')
    require(sha(T / 'packages/wo_a_parts_b4s3.so') == 'e42dc74f1695994c3e0b698813288a15e8d5af5ffef6af0cd3b279c63ecbb5de', 'wo_a binary differs')
    (T / 'packages/patha').mkdir()
    manifest = {}
    for key, (module, rel, digest) in PATHA_SO.items():
        target = T / 'packages/patha' / Path(rel).name
        shutil.copyfile(G0_NATIVE / 'build' / rel, target)
        require(sha(target) == digest, 'path A copy differs')
        manifest[key] = dict(module=module, path=str(target), sha256=digest)
    shutil.copyfile(INDEXER_RECEIPT, T / 'INDEXER_PREFILL_IDENTITY.json')
    manifest['indexer_prefill_identity'] = dict(path=str(T / 'INDEXER_PREFILL_IDENTITY.json'), sha256=sha(T / 'INDEXER_PREFILL_IDENTITY.json'))
    save(T / 'PATHA_NATIVE_MANIFEST.json', manifest)
    wrapper = T / 'packages/patha_prefill'
    wrapper.mkdir()
    pin = read(G1P_PIN)
    require(pin['status'] == 'G1P_QUALIFIED_NATIVE_PIN' and pin['binary_sha256'] == PATHA_SO['prefill'][2], 'G1p pin differs')
    for name, wanted in pin['source_sha256'].items():
        source = T / 'recipe/third_party/sglang/python/sglang/srt/layers/quantization/patha/vendor/prefill' / name
        require(sha(source) == wanted, 'G1p wrapper source differs: ' + name)
        shutil.copyfile(source, wrapper / name)
    shutil.copyfile(G1P_PIN, T / 'G1P_NATIVE_PIN.json')
    # Candidate overlay: Engram wkv through the Route 3 kernel, (25600, 6144) in its shapes behind SGLANG_ROUTE3_ENGRAM in this profile. The binary is unchanged.
    require(sha(T / 'packages/route3_decode.py') == 'b9f9f91efa50512f3e5a6dc3a4d5b688412e3e15220c7ac51ca3287471818789', 'Route 3 wrapper differs')
    shutil.copyfile(P / 'engram-route3-r1/evidence/route3_decode.py', T / 'packages/route3_decode.py')
    require(sha(T / 'packages/route3_decode.py') == '40ed2446db9d802bb5ab621962985ceab799af7ff5a9d2dfe64d1ac5f464768b', 'Route 3 wrapper overlay differs')
    native_after = inventory(T / 'packages')
    require(native_after[EXPERT] == PINS['expert_so'] and native_after['patha_expert_static.py'] == PINS['expert_py'], 'expert pins')
    require(native_after['route3_phase2_l2p0_r1.so'] == PINS['route3_so'], 'Route 3 binary differs')
    require(native_after['kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so'] == PINS['kt_ext'], 'KT copy differs')
    require(native_after['kt_kernel/experts_base.py'] == PINS['experts_base'], 'KT experts_base differs')
    kept = {k: v for k, v in p['native_after'].items() if k not in (EXPERT, KT_SO, 'route3_decode.py')}
    require({k: native_after[k] for k in kept} == kept, 'preserved packages differ')
    for name in ('client-tools', 'static-inputs', '.cache'):
        shutil.copytree(F / name, T / name, symlinks=True)
    require(inventory(T / 'client-tools') == p['client_sha256'], 'client tools differ')
    require(sha(T / 'client-tools/kv_checks.py') == PINS['host_reload'], 'host reload gate differs')
    for name in ('INPUTS.json', 'PREPARED.json', 'B1B_PAIR_CONFIG.json', 'CAMPAIGN_IDENTITY_BASELINE.json'):
        shutil.copy2(F / name, T / name)
    for name in ('hints', 'origins', 'route3-binary-origins', 'routing-dump', 'c1-binary-origins', 'patha-prefill-origins'):
        (T / name).mkdir(exist_ok=True)
    (T / '.cache/pycache').mkdir(exist_ok=True)
    require(not (T / 'static-inputs/combine-order.json').exists(), 'unexpected combine-order input')

    def relocate(value):
        return value.replace(str(F), str(T)) if isinstance(value, str) else value

    origin = read(F / 'ORIGINS_CONFIG.json')
    for name in ('source_root', 'receipt_dir', 'pycache_prefix'):
        origin[name] = relocate(origin[name])
    origin['trial_id'] = T.name
    origin['source_sha256'] = {k: v for k, v in source_after.items() if k.startswith('python/sglang/')}
    origin['native_modules'] = {
        'kt_kernel.kt_kernel_ext': dict(path=str(T / 'packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so'), sha256=PINS['kt_ext']),
        'kt_kernel.experts_base': dict(path=str(T / 'packages/kt_kernel/experts_base.py'), sha256=PINS['experts_base'])}
    origin['namespace_packages'] = origins.derive_namespace_packages(T / 'recipe/third_party/sglang', source_after)
    origin['namespace_source_inventory_sha256'] = origins.inventory_digest(source_after)
    save(T / 'ORIGINS_CONFIG.json', origin)
    save(T / 'ORIGINS_TOPOLOGY_SMOKE.json', origins.topology_smoke(origin))
    origins.load_config(T / 'ORIGINS_CONFIG.json', sha(T / 'ORIGINS_CONFIG.json'))

    plan = read(F / 'LAUNCH.json')
    old_env = plan['environment']
    env = {key: relocate(value) for key, value in old_env.items()}
    require(env['CPLUS_INCLUDE_PATH'] == OLD_INCLUDE, 'unexpected include binding')
    env['CPLUS_INCLUDE_PATH'] = NEW_INCLUDE
    require(old_env['SGLANG_FP8_ROUTE3_CUDA'] == '1' and old_env['SGLANG_FP8_FIXED_SPLIT_K'] == '4', 'C1-on dense profile differs')
    for k in ('SGLANG_FP8_FIXED_CUDA_CORE', 'SGLANG_FP8_FIXED_DECODE_GEOMETRY', 'SGLANG_FP8_FIXED_DECODE_TRANSPOSE'):
        require(env.get(k, '0') == '0', 'path A conflict: ' + k)
    require(old_env['KT_SM89_FUSED_MERGE'] == '1', 'C1 fused merge differs')
    require(old_env['KT_IDX_LIVE_TOPK'] == '1' and 'KT_IDX_LIVE_TOPK_MAX_L' not in old_env, 'C1 live top-k differs')
    env.update(SGLANG_PATHA_NATIVE_MANIFEST=str(T / 'PATHA_NATIVE_MANIFEST.json'), KT_EXPERT_ARENA_SPILL_STAGGER='1', KT_KV_HOT_DYN_HALF_LIFE='64', KT_KV_HOT_DYN_EVERY='4', KT_KV_HOT_DYN_HYST='1.5', KT_CONTRACT_V2_ROUTE_REF='1', KT_CONTRACT_V2_GLUE_FUSED='1', SGLANG_DSV41_FIXED_WO_A_CUDA='1', KT_IDX_LIVE_TOPK_MAX_L='1048580', KT_PREFILL_KV_LEND_ALL_TENSORS='1', SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR='/scratch/aster/bench/ada-kv-disk-gate-20261102-r5', SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE='32G', SGLANG_FLUSH_CLEARS_STORAGE='1', SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE='60G', SGLANG_ROUTE3_ENGRAM='1', SGLANG_SM89_ATTN_STAGED='1', KT_KV_HOT_LAYER_QUOTA=str(P / 'ada-spill-r1/layer-quota-r1/LAYER_QUOTA.json'), DSV41_ENGRAM_AIO='1', KT_EXPERT_ARENA_REBALANCE_NODE0='3', KT_KV_HOT_DYN_WORKER='1')
    require(env.pop('KT_ROUTING_DUMP') == str(T / 'routing-dump/counts'), 'routing dump binding differs')
    require('KT_GPU_STREAM_PATHA' not in old_env, 'prefill selector already bound')
    env.update(KT_GPU_STREAM_PATHA='1', KT_GPU_STREAM_PATHA_DIR=str(wrapper), KT_GPU_STREAM_PATHA_DOWN_PLANES='3', KT_GPU_STREAM_PATHA_MAX_ROWS='0',
               KT_GPU_STREAM_PATHA_FLAGS_LOG='1', KT_GPU_STREAM_PATHA_NATIVE_PIN=str(T / 'G1P_NATIVE_PIN.json'),
               KT_GPU_STREAM_PATHA_NATIVE_PIN_SHA256=sha(T / 'G1P_NATIVE_PIN.json'), KT_GPU_STREAM_PATHA_ORIGIN_DIR=str(T / 'patha-prefill-origins'),
               PATHA_EXT_NAME='patha_prefill_g0_dual')
    env['KT_PREFILL_PLAN_COST'] = '2600,491840,280,820'
    env['FEATURE_ORIGINS_CONFIG_SHA256'] = sha(T / 'ORIGINS_CONFIG.json')
    argv = [relocate(value) for value in plan['argv']]
    require(argv[0] == OLD_PYTHON, 'unexpected interpreter binding')
    argv[0] = str(ENV / 'bin/python')

    # The accepted serving profile: A3's arguments and cache policy. Placement stays C1's, as E1's build kept it.
    serving = read(A3 / 'LAUNCH.json')
    require(read(A3 / 'R19_A3_NUMERICAL_REVIEW_r1.json')['status'] == 'A3_ALL_GATES_PASS_HELD', 'A3 not accepted')
    norm = lambda d, root: {k: (v.replace(str(root), 'TRIAL') if isinstance(v, str) else v) for k, v in d.items()}
    a, b = norm(serving['environment'], A3), norm(old_env, F)
    extra = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - set(POLICY_KEYS + C1_KEYS) - {'FEATURE_ORIGINS_CONFIG_SHA256', 'KT_ROUTING_DUMP'}
    require(not extra, 'A3 environment differs beyond the policy: ' + str(sorted(extra)))
    for key in POLICY_KEYS:
        env[key] = serving['environment'][key]
    require(env['KT_PREFILL_SWA_LEND'] == '0' and env['SGLANG_CACHE_TTL'] == '1' and env['KT_STREAMING_SESSION_ADMISSION'] == '1', 'serving policy differs')
    # Serving keeps SWA mapped, about 3.5 GB less window room than the performance profile, which the plan's cost model does not see.
    require(env['KT_PREFILL_PLAN_MARGIN_MB'] == '128', 'plan margin differs')
    env['KT_PREFILL_PLAN_MARGIN_MB'] = '4096'
    only = ['--enable-streaming-session', '--prefill-max-requests', '4', '--max-queued-requests', '4', '--enable-hierarchical-cache',
            '--enable-cache-report', '--hicache-ratio', '4.0', '--hicache-write-policy', 'write_through']
    wanted = [value.replace(str(A3), str(T)) for value in serving['argv']]
    require(wanted[-len(only):] == only, 'A3 serving arguments differ')
    expected = [value for value in argv if value != '--disable-radix-cache']
    expected[expected.index('--max-running-requests') + 1] = '4'
    require(wanted[1:-len(only)] == expected[1:], 'A3 arguments differ beyond the serving set')
    argv = [argv[0], *wanted[1:], '--hicache-storage-backend', 'file']
    for key, value in env.items():
        require(not (isinstance(value, str) and (str(F) in value or '05yuki-main-00fd12a-candidate/.venv' in value)), 'stale binding: ' + key)
    require(env['SGLANG_FP8_PATHA_EXPERT_STATIC'] == '1', 'C1-on profile differs')
    require(env['KT_CONTRACT_V2_COMBINE_ORDER'] == 'logical' and env['KT_HOT_DYNAMIC'] == '0' and env['KT_CPU_THREADS'] == '42', 'r19 profile differs')
    changed_env = {k: dict(before=old_env.get(k), after=env[k]) for k in env if env[k] != old_env.get(k)}
    plan.update(argv=argv, environment=env, purpose='Ada serving gate: arm 0 plus prefill path A on the staggered spill with the round worker and the accepted A3 cache policy; frozen feature client and host-reload gate')
    save(T / 'LAUNCH.json', plan)
    profile = read(A3 / 'PROFILE.json')
    require(profile['kind'] == 'combined-prefix' and profile['exclusive_caller_owned_server'] is True, 'A3 profile differs')
    profile.update(environment=env, purpose=plan['purpose'], source_sha256=source_after)
    save(T / 'PROFILE.json', profile)
    refs = read(F / 'R19_IDENTITY_REFERENCES.json')
    for item in refs['files']:
        require(sha(item['path']) == item['sha256'], 'reference changed: ' + item['path'])
    refs['logical_baseline'] = dict(path=str(T / 'CAMPAIGN_IDENTITY_BASELINE.json'), sha256=sha(T / 'CAMPAIGN_IDENTITY_BASELINE.json'))
    require(refs['logical_baseline']['sha256'] == read(F / 'R19_IDENTITY_REFERENCES.json')['logical_baseline']['sha256'], 'baseline differs')
    save(T / 'R19_IDENTITY_REFERENCES.json', refs)
    cache = read(F / 'CACHE_PREPARED.json')
    require(cache['prepared_sha256'] == sha(T / 'PREPARED.json'), 'cache preparation mismatch')
    for rel, wanted in cache['files'].items():
        require(sha(T / '.cache/sglang' / rel) == wanted, 'cache file differs: ' + rel)
    cache.update(workspace=str(T / '.cache/sglang'), reuse=dict(source=str(F / 'CACHE_PREPARED.json'), source_sha256=sha(F / 'CACHE_PREPARED.json'), operation='byte-verified copy; no execution'))
    save(T / 'CACHE_PREPARED.json', cache)
    save(T / 'CACHE_RUN.json', dict(status='CACHE_PREPARED', cleanup=dict(empty=True), operation='byte-verified copy'))

    prepared = dict(p)
    prepared.update(trial=str(T), launch_sha256=sha(T / 'LAUNCH.json'), source_after=source_after,
                    native_after=native_after, adapter_after=inventory(adapter), client_sha256=inventory(T / 'client-tools'))
    prepared['pinned_files'] = {name: sha(T / name) for name in p['pinned_files']}
    prepared['tool_sha256'] = dict(p['tool_sha256'])
    for name in TOOLS:
        prepared['tool_sha256']['../six-node-performance-r1/tools/' + name] = sha(P / 'tools' / name)
    prepared['shared_input_sha256'] = dict(p['shared_input_sha256'])
    for path in (A3 / 'LAUNCH.json', A3 / 'PROFILE.json', A3 / 'R19_A3_NUMERICAL_REVIEW_r1.json', G1P_PIN, G0 / 'SOURCE_STACK.json', G0 / 'REPAIR_RESULT.json', F / F_PREPARED, F / 'LAUNCH.json',
                 F / 'RUN-r19-c1-static-on-identity-r2.json', F / 'PROFILE_RESOLVED.json', ENV / 'bin/python',
                 *[NEW_ADAPTER / n for n in (*ADAPTER_FROM_NEW, 'sitecustomize.py')]):
        prepared['shared_input_sha256'][str(path)] = sha(path)
    prepared['r19_C1'] = dict(p['r19_C1'], binary_sha256=PINS['expert_so'])
    prepared['phase_g_arm0'] = dict(
        parent=str(F), parent_manifest_sha256=sha(F / F_PREPARED), g0_source_root=str(G0_SRC),
        r1_failure=str(B / 'phase-g-g1v-arm0-identity-20261030-r1'),
        r2_failure=str(B / 'phase-g-g1v-arm0-identity-20261030-r2'),
        g0_source_stack_sha256=PINS['g0_source_stack'], g0_native_root=str(G0_NATIVE), python=str(ENV / 'bin/python'),
        replaced=dict(sglang_tree='G0 repaired tree (copied byte-for-byte)', kt_kernel='C1 Python files with the G0 native rebuild',
                      expert_binary=dict(before=p['r19_C1']['binary_sha256'], after=PINS['expert_so']),
                      engram_adapter='new recipe adapter: _empty -> new_rows rename only', interpreter=dict(before=OLD_PYTHON, after=argv[0])),
        environment_delta=changed_env, selectors_on=['KT_GPU_STREAM_PATHA', 'KT_EXPERT_ARENA_SPILL_STAGGER', 'KT_KV_HOT_DYN_WORKER'], selectors_off=['SGLANG_FP8_PATHA_DENSE', 'SGLANG_SINGLE_ROUTE'],
        gate='Ada: frozen feature client, host-reload gate and replays on the staggered spill with the round worker; numerics are bit-exact to C1 on the numerical profile',
        integrated=dict(receipt=str(INTEGRATED), inventory_sha256=INTEGRATED_INVENTORY), patha_manifest=manifest,
        g1_component_receipts=dict(g1a=str(B / 'phase-g-g1a-gpu-component-20261030-r1/out/RESULT.json'), g1e=str(B / 'phase-g-g1e-gpu-component-20261030-r1/out-r3/RESULT.json')))
    prepared['r19_revision'] = dict(phase='Phase G G1v arm 2', parent=str(F), performance_acceptance=False)
    save(T / PREPARED, prepared)
    import controller_phase_g_g1v_adadk4_r1 as controller
    controller.verify_prepared(T)
    controller.check_cache(T)
    print('PHASE_G_G1V_ARM0_PREPARED_NO_MODEL', sha(T / PREPARED), flush=True)
