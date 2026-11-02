# Ada baseline profile (Claude, 31 Oct 2026): the arm-2 G1p profile back on the C1 Route 3 path.
# Phase G G1v arm 2 profile (Claude, 30 Oct 2026): the arm-0 profile with Route 3 off, path A dense and the single
# route on, and the C1 server-argument comparison recorded instead of required. Arm-0 header follows.
# Phase G G1v arm 0 profile (Claude, 30 Oct 2026): profile_r19_C1_static_on_identity_r2.py with G0's swa.py pin
# (f2efb9b5..., the A3 fix now upstream), the arm-0 manifest, and a stop before weights if any accepted C1
# resolved server argument changes under the G0 tree.
"""Resolve this exact launch before weight loading, within the owned model scope."""
import json,os,sys
from pathlib import Path
T=Path(sys.argv[1]); plan=json.loads((T/'LAUNCH.json').read_text())
from sglang.srt.server_args import prepare_server_args
args=prepare_server_args(plan['argv'][3:]); args.resolve_once(); args.check_server_args(); resolved=args.resolved_dict()
assert resolved['enable_decoder_swa_bounded_replay'] is False
assert os.environ['SGLANG_FP8_FIXED_SPLIT_K']=='4'
assert os.environ['SGLANG_FP8_FIXED_WORKSPACE_MB']=='512'
assert os.environ['SGLANG_FP8_FIXED_DECODE_TRANSPOSE']=='0'
assert os.environ['SGLANG_SM89_ATTN_SPLITK']=='0'
assert resolved['kt_num_gpu_experts']==1
assert resolved['kt_cpuinfer']==42 and resolved['kt_threadpool_count']==6
assert resolved['chunked_prefill_size']==36864
assert resolved['cuda_graph_config']['decode']['backend']=='full'
assert resolved['cuda_graph_config']['prefill']['backend']=='disabled'
assert os.environ['SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS']=='0'
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail
from types import SimpleNamespace
assert envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.get()==0
assert DecoderReplayTail.maybe_create(SimpleNamespace(tail_writes_source_kv=True)) is None
assert os.environ['KT_GPU_STREAM_PREFILL']=='512' and os.environ['KT_KV_HOT_LEND']=='1'
assert resolved['disable_radix_cache'] and resolved['max_running_requests']==1
assert not resolved['enable_hierarchical_cache']
assert all(os.environ[k]=='1' for k in ('KT_CONTRACT_V2','KT_CONTRACT_V2_FAST_ADAPTER','KT_CONTRACT_V2_STATIC_GPU','KT_CONTRACT_V2_STREAM_GRAPH','KT_CONTRACT_V2_LAZY_ROUTE_SCRATCH','KT_SHARED_AFTER_SUBMIT','KT_HANDOFF_PACK','KT_STREAM_MEMOPS','KT_MEMOP_POLL_HINT'))
assert all(os.environ[k]=='0' for k in ('KT_MOE_FP32_INTERMEDIATES','KT_MOE_FP32_SUBTOTALS','KT_MOE_FP32_GRAPH','KT_MOE_FP32_HOT','KT_MOE_FP32_STREAM'))
assert os.environ['KT_MXFP4_BACKEND']=='contract_v2' and os.environ['KT_CONTRACT_V2_GPU_MODE']=='int8'
from sglang.srt.layers.moe.contract_v2 import moe, adapter, static_routes_candidate, stream_graph_candidate, arena_stream_candidate
assert moe._STATIC_GPU and moe._FAST
assert os.environ['KT_CONTRACT_V2_DECODE_SPLITK']=='8' and moe._DECODE_SPLITS==8
from sglang.srt.layers.moe.contract_v2 import decode_splitk_candidate

assert os.environ['KT_NUMA_NODES']=='0,1,2,4,5,6'
assert os.environ['KT_EXPERT_ARENA_SPILL_NODES']=='3,7'
assert os.environ['KT_EXPERT_ARENA_REBALANCE_NODE0']=='3'
assert os.sched_getaffinity(0)<=set(range(24))|set(range(32,56))
import kt_kernel.kt_kernel_ext as native
import flash_mla
import sglang.srt.models.deepseek_v4
assert 'FEATURE_REPLAY_OBSERVER_DIR' not in os.environ
assert 'model_observer_r4' not in sys.modules
assert os.environ['SGLANG_DSV41_FIXED_PROJECTIONS']==os.environ['SGLANG_MXFP4_FIXED_GEMM']==os.environ['SGLANG_DSV41_FIXED_WO_A']=='1'
from kt_kernel.utils.amx import NativeMoEWrapper
loader=NativeMoEWrapper._create_loader('MXFP4',resolved['kt_weight_path'])
packed,scales=loader.read_projection_arrays('blk.0.ffn_gate_exps.weight',0)
assert packed.dtype.name==scales.dtype.name=='uint8'
loader.close_all_handles()
from transformers import AutoTokenizer
from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe
from sglang.srt.sampling.watermarking.core import redact_watermark_secrets
tokenizer=AutoTokenizer.from_pretrained(resolved['tokenizer_path'],trust_remote_code=True,local_files_only=True)
renderer=DeepSeekRecipe.from_hf_tokenizer('dsv41',tokenizer)
ids=renderer.encode(dict(messages=[dict(role='user',content='Reply with OK.')],thinking=False))
assert len(ids)>0
(T/'RENDERER_PREFLIGHT.json').write_text(json.dumps(dict(status='PASS',input_ids=ids,package='deepseek-recipe==0.2.0'))+'\n')
(T/'PROFILE_RESOLVED_RAW.json').write_text(json.dumps(resolved,indent=2,default=str)+'\n')
(T/'PROFILE_RESOLVED.json').write_text(json.dumps(redact_watermark_secrets(resolved),indent=2,default=str)+'\n')
print('FULL_MODEL_IMPORT_AND_PROFILE_PASS',flush=True)

assert os.environ['SGLANG_FP8_ROUTE3_CUDA']=='1'
assert all(os.environ[k]=='1' for k in ('KT_CONTRACT_V2_GPU_DP4A','KT_CONTRACT_V2_QUANT_AVX2','SGLANG_DSV41_FIXED_LM_HEAD','KT_PREFILL_LEND_EXPERT_SCALES','KT_PREFILL_SWA_CLEAR_RELEASE_HOT'))
assert os.environ['KT_PREFILL_SWA_LEND']=='1'
assert moe._DP4A and adapter.expert_routes is moe.expert_routes
from sglang.kernels.ops.gemm import fixed_fp8_cuda_core,fixed_lm_head
from sglang.srt.layers.moe.contract_v2 import dp4a_projection_r1,dp4a_routes_candidate
assert fixed_lm_head.CHUNK_ROWS==16
assert not any('OBSERVER' in k for k in os.environ if k.startswith('FEATURE_'))

assert hasattr(fixed_lm_head, 'padded_bf16_head')

assert os.environ['SGLANG_FP8_FIXED_CUDA_CORE']=='0'
assert os.environ['SGLANG_FP8_FIXED_DECODE_GEOMETRY']=='0'
assert os.environ['ROUTE3_L2_POLICY']=='0'
from sglang.srt.layers.quantization.patha import policy as patha_policy
patha_policy.validate_policy()
assert not patha_policy.enabled()
patha_manifest=json.loads(Path(os.environ['SGLANG_PATHA_NATIVE_MANIFEST']).read_text())
import hashlib
for key in ('decode','prefill'):
    assert Path(patha_manifest[key]['path']).is_relative_to(T/'packages/patha')
    assert hashlib.sha256(Path(patha_manifest[key]['path']).read_bytes()).hexdigest()==patha_manifest[key]['sha256']
receipt=patha_manifest['indexer_prefill_identity']
assert hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
from sglang.kernels.ops.attention.dsv4.single_route import SINGLE_ROUTE
assert not SINGLE_ROUTE
import route3_decode
extension=route3_decode._load()
assert Path(extension.__file__).resolve()==T/'packages/route3_phase2_l2p0_r1.so'
print('ADA_ROUTE3_PREFLIGHT_PASS',flush=True)
assert 'KT_ROUTING_DUMP' not in os.environ and os.environ.get('KT_WORKER_TIMESTAMPS','0')=='0'

assert resolved['enable_eplb'] is False and os.environ['KT_HOT_DYNAMIC']=='0'
assert os.environ['KT_CONTRACT_V2_COMBINE_ORDER']=='logical'
from sglang.srt.layers.moe.contract_v2 import placement_order
assert placement_order._MODE=='logical'
print('R19_FIXED_PLACEMENT_ORDER_PROFILE_PASS',flush=True)

import hashlib
assert hashlib.sha256((T/'recipe/third_party/sglang/python/sglang/srt/mem_cache/unified_cache/components/swa.py').read_bytes()).hexdigest()=='f2efb9b5cac3a794efedd784f951e6eabff74480f14b09e8c1a91a19d9d6c5a5'
print('R19_FAST_ORDER_AND_SWA_SOURCE_PASS',flush=True)

assert os.environ['SGLANG_FP8_PATHA_EXPERT_STATIC']=='1'
assert moe._PATHA_STATIC is True
assert all(os.environ[k]==v for k,v in dict(PATHA_EXPERT_LUT='0',PATHA_STAGES='3',PATHA_EXPERT_BPC='8',PATHA_EXPERT_TC='0').items())
c1_preparation=json.loads((T/'PHASE_G_G1V_ADAYPERF_PREPARED_r1.json').read_text())
for rel,wanted in c1_preparation['native_after'].items():
    assert hashlib.sha256((T/'packages'/rel).read_bytes()).hexdigest()==wanted
assert hasattr(native.moe.AVX2ContractV2_MOE,'forward_routes_task') and native.contract_v2_truncation_events()==0
print('R19_C1_STATIC_ON_PROFILE_PASS',flush=True)

assert not (T/'static-inputs/combine-order.json').exists()

from sglang.srt.model_executor.model_runner import ModelRunner
assert ModelRunner.init_cuda_graphs.__globals__["os"] is os
print("C1_MODEL_RUNNER_OS_IMPORT_PREFLIGHT_PASS",flush=True)

# Phase G G1v arm 0 additions (Claude, 30 Oct 2026).
assert os.environ.get('SGLANG_FP8_PATHA_DENSE', '0') == '0' and os.environ.get('SGLANG_SINGLE_ROUTE', '0') == '0', 'Ada selectors'
assert os.environ['KT_GPU_STREAM_PATHA'] == '1' and os.environ['KT_GPU_STREAM_PATHA_FLAGS_LOG'] == '0' and os.environ['KT_GPU_STREAM_PATHA_DOWN_PLANES'] == '3'
from sglang.srt.layers.moe.contract_v2 import stream_graph_candidate, stream_patha
assert stream_graph_candidate.PATHA is True
for name, wanted in stream_patha.PINNED_SOURCES.items():
    assert hashlib.sha256((Path(os.environ['KT_GPU_STREAM_PATHA_DIR']) / name).read_bytes()).hexdigest() == wanted
assert hashlib.sha256(Path(os.environ['KT_GPU_STREAM_PATHA_NATIVE_PIN']).read_bytes()).hexdigest() == os.environ['KT_GPU_STREAM_PATHA_NATIVE_PIN_SHA256']
assert os.environ['KT_PREFILL_PLAN_COST'] == '2600,491840,280,820'
print('G1P_SELECTOR_PROFILE_PASS', flush=True)
assert Path(sys.executable).resolve() == Path('/scratch/aster/bench/phase-g-g0-environment-20261030-r3/venv/bin/python').resolve(), 'interpreter differs'
assert Path(sglang.srt.models.deepseek_v4.__file__).resolve().is_relative_to(T / 'recipe/third_party/sglang/python'), 'SGLang not from the staged integrated copy'
assert Path(native.__file__).resolve() == T / 'packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so', 'KT not from the staged G0 package'
F = Path('/scratch/aster/bench/feature-trial-B-r19-c1-static-on-identity-six-r2')
accepted = json.loads((F / 'PROFILE_RESOLVED.json').read_text())
current = json.loads((T / 'PROFILE_RESOLVED.json').read_text())
def norm(value):
    text = json.dumps(value, sort_keys=True).replace(str(F), str(T))
    return json.loads(text.replace('/scratch/aster/bench/05yuki-main-00fd12a-candidate/.venv/bin/python', str(Path(sys.executable))))
changed = {k: dict(accepted=accepted[k], current=current[k]) for k in sorted(set(accepted) & set(current)) if norm(accepted[k]) != norm(current[k])}
diff = dict(changed=changed, added={k: current[k] for k in sorted(set(current) - set(accepted))}, removed={k: accepted[k] for k in sorted(set(accepted) - set(current))})
(T / 'PROFILE_DIFF_VS_C1.json').write_text(json.dumps(diff, indent=2, default=str) + '\n')
assert not changed and not diff['removed'], 'resolved server arguments differ from accepted C1; classify before loading weights'
print('ADA_PROFILE_PASS changed=' + ','.join(changed) + ' removed=' + ','.join(diff['removed']) + ' added=' + str(len(diff['added'])), flush=True)
