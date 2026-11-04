"""Overlay: the DSpark draft's routed experts in the KT CPU pool, selector KT_DSPARK_CPU_DRAFT, default off (Claude, 3 Nov 2026).
argv: source sglang package, destination sglang package, source kt_kernel package, destination kt_kernel package,
kt_dspark_draft.py. Reads the sources only; writes the patched files (and the new module) under the destinations,
which hold copies of the sources. Selector off, every patched path runs the served code unchanged."""
import shutil
import sys
from pathlib import Path

SRC, DST, KSRC, KDST, MODULE = (Path(a) for a in sys.argv[1:6])


def rd(root, rel):
    return (root / rel).read_text(encoding='utf-8')


def wr(root, rel, s):
    if rel.endswith('.py'):
        compile(s, rel, 'exec')
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(s.encode('utf-8'))


def sub(s, old, new, n=1):
    assert s.count(old) == n, (s.count(old), old[:90])
    return s.replace(old, new)


GATE = '_DSPARK_CPU_DRAFT and _draft_layer(self)'

# B1, B1b: draft scope, layer ids 40-42, own registry; out of the target's KT_LAYERS and hot lend
rel = 'srt/layers/moe/kt_ep_wrapper.py'
s = rd(SRC, rel)
s = sub(s, 'KT_LAYERS = {}\n',
        'KT_LAYERS = {}\n'
        '# KT_DSPARK_CPU_DRAFT=1 (default off): the DSpark draft\'s MoE layers get KT layer ids of their own\n'
        '# (KT_DSPARK_DRAFT_LAYER_BASE + stage, 40-42) in this registry, never in KT_LAYERS (kt_dspark_draft.py).\n'
        '_DSPARK_CPU_DRAFT = os.environ.get("KT_DSPARK_CPU_DRAFT", "0") == "1"\n'
        'KT_DRAFT_LAYERS = {}\n'
        '\n\n'
        'def _draft_layer(method) -> bool:\n'
        '    return getattr(method.kt_config, "draft_stage", None) is not None\n')
s = sub(s, '    if get_exec().moe.kt_weight_path is None:\n        return None\n\n    num_layers = getattr(',
        '    if get_exec().moe.kt_weight_path is None:\n        return None\n'
        '    if _DSPARK_CPU_DRAFT:\n'
        '        # a MoE layer built inside draft_model_build_scope: the draft\'s own KT config\n'
        '        from sglang.srt.layers.moe import kt_dspark_draft\n'
        '\n'
        '        draft = kt_dspark_draft.draft_kt_config(server_args, layer_idx)\n'
        '        if draft is not None:\n'
        '            return draft\n'
        '\n    num_layers = getattr(')
s = sub(s, '        # kt_kv_hot_lend: the next X experts\' weights on a shadow layer, for the\n        # KV pool\'s free pages\n',
        f'        if {GATE}:\n'
        '            # KT_DSPARK_CPU_DRAFT: every draft expert on the CPU, no shadow, own loader and registry\n'
        '            from sglang.srt.layers.moe import kt_dspark_draft\n'
        '\n'
        '            kt_dspark_draft.create_wrapper(self, layer, num_experts, num_experts_per_tok, hidden_size,\n'
        '                                           intermediate_size_full)\n'
        '            return\n'
        '        # kt_kv_hot_lend: the next X experts\' weights on a shadow layer, for the\n        # KV pool\'s free pages\n')
s = sub(s, '        # 2. Load CPU weights using KT wrapper\n        if self.tp_rank == 0 and self.wrapper is not None:\n',
        f'        if {GATE}:\n'
        '            # KT_DSPARK_CPU_DRAFT: identity map (drafts have no placement), no hot-lend split, no streamer\n'
        '            from sglang.srt.layers.moe import kt_dspark_draft\n'
        '\n'
        '            kt_dspark_draft.load(self, layer)\n'
        '            return\n'
        '        # 2. Load CPU weights using KT wrapper\n        if self.tp_rank == 0 and self.wrapper is not None:\n')
wr(DST, rel, s)

# The draft forward: the contract adapter's CPU-only case, own handoff buffers, logical combine order
rel = 'srt/layers/moe/contract_v2/adapter.py'
s = rd(SRC, rel)
s = sub(s, 'def apply(wrapper, layer, dispatch):\n'
           '    from sglang.srt.layers.moe import kt_kv_hot_lend, kt_stream_prefill, kt_ep_wrapper\n'
           '    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput\n',
        'def apply(wrapper, layer, dispatch):\n'
        '    from sglang.srt.layers.moe import kt_kv_hot_lend, kt_stream_prefill, kt_ep_wrapper\n'
        '    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput\n'
        '    if getattr(kt_ep_wrapper, "_DSPARK_CPU_DRAFT", False) and kt_ep_wrapper._draft_layer(wrapper):\n'
        '        # KT_DSPARK_CPU_DRAFT: a draft layer, every route on the CPU\n'
        '        from sglang.srt.layers.moe import kt_dspark_draft\n'
        '        return kt_dspark_draft.apply(wrapper, layer, dispatch)\n')
wr(DST, rel, s)

# The draft's GPU layer holds no routed expert: its weight loader skips them (the served loader would index past
# the layer's GPU rows: draft layers load without expert-location metadata, so no KT bounds check runs)
rel = 'srt/layers/moe/fused_moe_triton/layer.py'
s = rd(SRC, rel)
s = sub(s, '_TP2_EXACT = os.environ.get("SGLANG_PATHA_TP2_EXACT", "0") == "1"\n',
        '_TP2_EXACT = os.environ.get("SGLANG_PATHA_TP2_EXACT", "0") == "1"\n'
        '_KT_DSPARK_CPU_DRAFT = os.environ.get("KT_DSPARK_CPU_DRAFT", "0") == "1"\n')
s = sub(s, '        if global_expert_location_metadata is None:\n'
           '            if not getattr(param, "_sglang_require_global_experts", False):\n'
           '                expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)\n'
           '                if expert_id == -1:\n'
           '                    return\n',
        '        if global_expert_location_metadata is None:\n'
        '            if not getattr(param, "_sglang_require_global_experts", False):\n'
        '                expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)\n'
        '                if expert_id == -1:\n'
        '                    return\n'
        '                # KT_DSPARK_CPU_DRAFT: a draft KT layer keeps its routed experts on the CPU only\n'
        '                if (_KT_DSPARK_CPU_DRAFT and isinstance(self.quant_method, KTEPWrapperMethod)\n'
        '                        and getattr(self.quant_method.kt_config, "draft_stage", None) is not None\n'
        '                        and self.quant_method.num_gpu_experts <= expert_id < self._num_local_routed):\n'
        '                    return\n')
wr(DST, rel, s)

shutil.copyfile(MODULE, DST / 'srt/layers/moe/kt_dspark_draft.py')
compile((DST / 'srt/layers/moe/kt_dspark_draft.py').read_text(encoding='utf-8'), 'kt_dspark_draft.py', 'exec')

# B1c: a per-wrapper loader (kt_kernel); without a factory in the context, the class-level loader as served
rel = 'utils/amx.py'
s = rd(KSRC, rel)
s = sub(s, 'logger = logging.getLogger(__name__)\n',
        'logger = logging.getLogger(__name__)\n'
        '\n'
        'import contextvars\n'
        '\n'
        '# SGLang\'s KT_DSPARK_CPU_DRAFT (kt_dspark_draft.py) sets a loader factory around one wrapper\'s construction:\n'
        '# that wrapper then loads through its own loader. None (always, without the selector): the class-level loader.\n'
        '_OWN_LOADER = contextvars.ContextVar("kt_own_loader", default=None)\n')
s = sub(s, '        if NativeMoEWrapper._native_loader_instance is None:\n'
           '            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)\n'
           '        self.loader = NativeMoEWrapper._native_loader_instance\n',
        '        _own = _OWN_LOADER.get()\n'
        '        if _own is not None:\n'
        '            # its own loader, made at load_weights: the class-level one is not created, read or released\n'
        '            self._own_loader_factory = _own\n'
        '            self.loader = None\n'
        '        else:\n'
        '            if NativeMoEWrapper._native_loader_instance is None:\n'
        '                NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)\n'
        '            self.loader = NativeMoEWrapper._native_loader_instance\n')
s = sub(s, '    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):\n'
           '        import time\n'
           '\n'
           '        if NativeMoEWrapper._native_loader_instance is None:\n'
           '            t_recreate_start = time.time()\n'
           '            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(self.method, self.weight_path)\n',
        '    def load_weights(self, physical_to_logical_map_cpu: torch.Tensor):\n'
        '        import time\n'
        '\n'
        '        _own = self.__dict__.get("_own_loader_factory")\n'
        '        if _own is not None:\n'
        '            self.loader = _own()\n'
        '        elif NativeMoEWrapper._native_loader_instance is None:\n'
        '            t_recreate_start = time.time()\n'
        '            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(self.method, self.weight_path)\n')
s = sub(s, '        NativeMoEWrapper._release_loader(layer_idx=self.layer_idx)\n        t6 = time.time()\n',
        '        if _own is not None:\n'
        '            self.loader.close_all_handles()\n'
        '            self.loader = None\n'
        '        else:\n'
        '            NativeMoEWrapper._release_loader(layer_idx=self.layer_idx)\n'
        '        t6 = time.time()\n')
wr(KDST, rel, s)
print('patched')
