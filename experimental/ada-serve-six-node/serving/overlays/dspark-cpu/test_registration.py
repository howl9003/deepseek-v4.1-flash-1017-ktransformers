"""Target and draft KT wrappers registered in one process, CPU only (Claude, 3 Nov 2026).

Runs the real SGLang code of the tree on PYTHONPATH (the served copy or the overlay): create_kt_config_from_server_args,
draft_model_build_scope, KTEPWrapperMethod.__init__ / create_weights / process_weights_after_loading, kt_kv_hot_lend
(make_shadow, cpu_holds_all, _Layer, split for the draft), FusedMoE.weight_loader and contract_v2.adapter.apply's
dispatch. Stand-ins, because a CPU process cannot run them: kt_kernel's KTMoEWrapper (a recorder; the real one is
exercised in test_draft_forward.py), the GPU MoE method (records shapes, tiny tensors), the expert-location metadata
(seeded [40, 384] permutations), torch.cuda.synchronize (no-op) and kt_kv_hot_lend.split for target layers (the real
one maps CUDA slots; the stand-in appends the same _Layer the real one would). Config bags: the C2 launch's KT values.

argv: out.json, scenario (target | target+draft). The selector comes from the environment (KT_DSPARK_CPU_DRAFT)."""
import dataclasses
import hashlib
import json
import os
import sys
import types

import torch

OUT, SCENARIO = sys.argv[1], sys.argv[2]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available()

from sglang.srt import runtime_context as rc  # noqa: E402


def bag(path, fields, subs=None):
    b = rc._ConfigBag(path)
    for k, v in fields.items():
        b._set(k, v)
    for k, s in (subs or {}).items():
        b._set_sub(k, s)
    return b


TARGET_GGUF = '/scratch/aster/models/gguf/DeepSeek-V4.1-Flash-1017-MXFP4.gguf'
moe = bag('exec.moe', dict(kt_weight_path=TARGET_GGUF, kt_num_gpu_experts=1, kt_cpuinfer=42, kt_threadpool_count=6,
                           kt_method='MXFP4', kt_max_deferred_experts_per_token=0, enable_w4a4_mxfp4_megamoe=False))
rc._CONTEXT._config_bags = dict(
    exec=bag('exec', {}, dict(moe=moe, comm=bag('exec.comm', dict(boundary_reduction=None)))),
    schedule=bag('schedule', dict(chunked_prefill_size=36864, max_running_requests=4)),
    spec=bag('spec', dict(speculative_boundary_reduction=None, speculative_enable_w4a4_mxfp4_megamoe=None)))
rc.get_parallel()._stamp['tp_rank'] = 0

import sglang.srt.arg_groups.overrides as overrides  # noqa: E402
from sglang.srt.layers.moe import kt_ep_wrapper as w  # noqa: E402
from sglang.srt.layers.moe import kt_kv_hot_lend as hot  # noqa: E402
from sglang.srt.layers.moe import contract_v2  # noqa: E402
from sglang.srt.layers.moe.utils import draft_model_build_scope  # noqa: E402
import sglang.srt.eplb.expert_location_dispatch as eld  # noqa: E402
from kt_kernel.utils import amx  # noqa: E402

model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(num_hidden_layers=40))
overrides.model_config_of = w.model_config_of = lambda server_args: model_config
torch.cuda.synchronize = lambda *a, **k: None
# the C2 hot-lend switches (KT_KV_HOT_LEND=1 also needs the CUDA VMM library at load; forced here)
hot.ENABLED, hot.EXTRA_AUTO, hot.EXTRA, hot.DYNAMIC, hot.DYN_RES = True, True, 0, True, True
assert contract_v2.ENABLED, 'run with KT_CONTRACT_V2=1'
gen = torch.Generator().manual_seed(20261103)
P2L = torch.stack([torch.randperm(384, generator=gen) for _ in range(40)])
eld.get_global_expert_location_metadata = lambda: types.SimpleNamespace(physical_to_logical_map_cpu=P2L)

TAG = {}


def tag(obj):
    return TAG.get(id(obj), repr(type(obj).__name__))


class FakeKT:
    made = []

    def __init__(self, **kw):
        self.kw = kw
        self.layer_idx = kw['layer_idx']
        self.own_loader_at_init = None if not hasattr(amx, '_OWN_LOADER') else amx._OWN_LOADER.get()
        self.class_loader_at_init = amx.NativeMoEWrapper._native_loader_instance
        self.loaded = None
        FakeKT.made.append(self)

    def load_weights(self, m):
        self.loaded = m.clone()


w.KTMoEWrapper = FakeKT


class FakeGPU:
    def __init__(self):
        self.made = []

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra):
        self.made.append(num_experts)
        for name in ('w13_weight', 'w2_weight'):
            layer.register_parameter(name, torch.nn.Parameter(torch.zeros(num_experts, 2, dtype=torch.int8), requires_grad=False))

    def create_moe_runner(self, layer, cfg):
        pass

    def get_moe_runner_backend(self, layer):
        return None

    def process_weights_after_loading(self, layer):
        pass


class FakeLayer(torch.nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k, self.intermediate_size_per_partition, self.moe_tp_size, self.moe_ep_size = top_k, 2304, 1, 1
        self.moe_runner_config = types.SimpleNamespace(swiglu_limit=10.0, gemm1_alpha=None, gemm1_clamp_limit=None)


real_split = hot.split
split_calls = []


def split(wrapper, layer):
    split_calls.append(tag(wrapper))
    sh = wrapper.__dict__.get('_kt_kv_hot_shadow')
    if sh is None:
        return real_split(wrapper, layer)   # must return at once: nothing to lend
    wrapper.__dict__.pop('_kt_kv_hot_shadow')
    L = hot._Layer(wrapper, len(hot.LAYERS))
    L.shadow, L.method = sh[0], sh[1]
    hot.LAYERS.append(L)
    wrapper.__dict__['_kt_kv_hot'] = L


hot.split = split


def build(layer_id, top_k, num_experts, name):
    cfg = w.create_kt_config_from_server_args(None, layer_id)
    m = w.KTEPWrapperMethod(FakeGPU(), cfg)
    TAG[id(m)] = name
    layer = FakeLayer(top_k)
    m.create_weights(layer, num_experts=num_experts, hidden_size=5120, intermediate_size_per_partition=2304,
                     params_dtype=torch.bfloat16, weight_loader=None)
    return m, layer


targets = [build(L, 6, 384, f'target{L}') for L in range(40)]
for m, layer in targets:
    m.process_weights_after_loading(layer)
drafts = []
if SCENARIO == 'target+draft':
    with draft_model_build_scope():
        drafts = [build(s, 3, 128, f'draft{s}') for s in range(3)]
    for m, layer in drafts:
        m.process_weights_after_loading(layer)
    # the real split on each draft wrapper: nothing to lend, nothing appended
    for m, layer in drafts:
        real_split(m, layer)


def cfg_dict(c):
    d = dataclasses.asdict(c)
    d['type'] = type(c).__name__
    return d


def kt_record(r):
    kw = dict(r.kw)
    mask = kw.pop('gpu_experts_mask')
    kw['gpu_mask_true'] = int(mask.sum())
    kw['gpu_mask_len'] = int(mask.numel())
    out = dict(kw=kw, own_loader_at_init=None if r.own_loader_at_init is None else repr(r.own_loader_at_init.func.__name__) + str(r.own_loader_at_init.args),
               class_loader_at_init=r.class_loader_at_init is not None, weight_base_key=getattr(r, 'weight_base_key', None))
    if r.loaded is not None:
        out['load_map_len'] = int(r.loaded.numel())
        out['load_map_identity'] = bool(torch.equal(r.loaded, torch.arange(r.loaded.numel())))
        out['load_map_sha'] = hashlib.sha256(r.loaded.to(torch.int64).numpy().tobytes()).hexdigest()[:16]
    return out


def method_state(m):
    return dict(cfg=cfg_dict(m.kt_config), num_gpu_experts=m.num_gpu_experts, global_num_experts=m.global_num_experts,
                gpu_rows=m.gpu_method.made, shadow=('_kt_kv_hot_shadow' in m.__dict__), hot_layer=(m.__dict__['_kt_kv_hot'].idx if '_kt_kv_hot' in m.__dict__ else None),
                cpu_holds_all=hot.cpu_holds_all(m), wrapper=kt_record(m.wrapper) if m.wrapper is not None else None)


state = dict(
    tree=os.path.dirname(w.__file__).split('/sglang/')[0],
    selector=os.environ.get('KT_DSPARK_CPU_DRAFT', '0'),
    scenario=SCENARIO,
    KT_LAYERS={str(k): tag(v) for k, v in sorted(w.KT_LAYERS.items())},
    KT_DRAFT_LAYERS={str(k): tag(v) for k, v in sorted(getattr(w, 'KT_DRAFT_LAYERS', {}).items())},
    hot_LAYERS=[dict(idx=L.idx, layer_idx=L.layer_idx, wrapper=tag(L.wrapper), n=L.n, x=L.x, b=L.b, e=L.e, held=L.held[:4], res_held=L.res_held)
                for L in hot.LAYERS],
    hot_STATE_keys=sorted(hot.STATE),
    split_calls=split_calls,
    class_loader=amx.NativeMoEWrapper._native_loader_instance is not None,
    kt_constructions=[(tag_of, kt_record(r)) for tag_of, r in
                      [(next((TAG[id(m)] for m, _ in targets + drafts if m.wrapper is r), '?'), r) for r in FakeKT.made]],
    targets={TAG[id(m)]: method_state(m) for m, _ in targets},
    drafts={TAG[id(m)]: method_state(m) for m, _ in drafts},
)

# FusedMoE.weight_loader on a draft layer: the checkpoint's routed experts 0, 1, 64, 127 of stage 0
if drafts:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    m, layer = drafts[0]
    stub = types.SimpleNamespace(quant_config=None, uses_expert_location=False, _expert_storage_rank=0, _num_local_routed=128,
                                 _num_global_routed=128, _has_fused_shared=False, quant_method=m, impl=[])
    stub._map_global_expert_id_to_local_expert_id = types.MethodType(FusedMoE._map_global_expert_id_to_local_expert_id, stub)
    param = layer.w13_weight

    def impl(param, loaded_weight, weight_name, shard_id, expert_id):
        stub.impl.append(expert_id)
        param.data[expert_id].copy_(loaded_weight)   # what _weight_loader_impl indexes

    stub._weight_loader_impl = impl
    res = {}
    for e in (0, 1, 64, 127):
        try:
            FusedMoE.weight_loader(stub, param, torch.ones(2, dtype=torch.int8), 'experts.w13_weight', 'w1', e)
            res[e] = 'loaded to GPU row' if e in stub.impl else 'skipped'
        except IndexError as exc:
            res[e] = 'IndexError: ' + str(exc)[:60]
    state['draft_weight_loader'] = dict(gpu_rows=int(param.shape[0]), result={str(k): v for k, v in res.items()})

    # contract_v2.adapter.apply: which path a draft and a target layer take (stopped at the first CUDA need)
    from sglang.srt.layers.moe.contract_v2 import adapter
    try:
        from sglang.srt.layers.moe import kt_dspark_draft
        kt_dspark_draft.apply = lambda wr, ly, dp: f'kt_dspark_draft.apply(layer {wr.kt_config.layer_idx})'
    except ImportError:
        pass
    disp = types.SimpleNamespace(hidden_states=torch.zeros(4, 8, dtype=torch.bfloat16),
                                 topk_output=types.SimpleNamespace(topk_ids=torch.zeros(4, 3, dtype=torch.int64), topk_weights=torch.ones(4, 3)))
    paths = {}
    for name, (mm, ly) in (('draft0', drafts[0]), ('target0', targets[0])):
        try:
            paths[name] = str(adapter.apply(mm, ly, disp))
        except Exception as exc:
            paths[name] = 'target path, stopped at ' + type(exc).__name__ + ': ' + str(exc)[:70]
    state['adapter_paths'] = paths

    # the draft's own pinned handoff buffers against kt-kernel's row-keyed ones (selector on)
    try:
        if not getattr(w, '_DSPARK_CPU_DRAFT', False):
            raise ImportError
        from sglang.srt.layers.moe import kt_dspark_draft
        from kt_kernel.experts_base import KExpertsCPUBuffer
        before = dict(KExpertsCPUBuffer.capture_buffers)
        hx, hid, hw, hb = kt_dspark_draft._host_buffers(torch.zeros(4, 5120, dtype=torch.bfloat16), 3)
        state['draft_host_buffers'] = dict(ids_shape=list(hid[0].shape), x_shape=list(hx[0].shape), depth=len(hid),
                                           kt_buffers_untouched=KExpertsCPUBuffer.capture_buffers == before)
    except ImportError:
        pass

with open(OUT, 'w') as f:
    json.dump(state, f, indent=1, default=str)
print('ok', SCENARIO, 'KT_LAYERS', len(state['KT_LAYERS']), 'draft', state['KT_DRAFT_LAYERS'], 'hot', len(state['hot_LAYERS']))
