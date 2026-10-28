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

assert all(os.environ[k]=='1' for k in ('SGLANG_FP8_ROUTE3_CUDA','KT_CONTRACT_V2_GPU_DP4A','KT_CONTRACT_V2_QUANT_AVX2','SGLANG_DSV41_FIXED_LM_HEAD','KT_EXPERT_ARENA_REBALANCE_NODE0','KT_PREFILL_LEND_EXPERT_SCALES','KT_PREFILL_SWA_CLEAR_RELEASE_HOT'))
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
import route3_decode
extension=route3_decode._load()
assert Path(extension.__file__).resolve()==T/'packages/route3_phase2_l2p0_r1.so'
print('ROUTE3_DEFAULT_BINARY_PERFORMANCE_PREFLIGHT_PASS',flush=True)
