"""Overlay: KT_DSPARK_PLAIN_GATE, per-request plain decode steps for DSpark (Claude, 4 Nov 2026; r4 = r3 with the
confidence signal plus the draft-only probe signal, plain-gate-r4/; the worker hunks are r3's, unchanged: the probe
lives in the module and calls the worker's own proposer and window plan). Selector
KT_DSPARK_PLAIN_GATE=1, default off. Applied after apply_dspark_cpu.py, apply_dspark_flags.py and
apply_dspark_swa_match.py: the worker pin is the ada-identity-dsp-20261104-r3 tree, which has all three.

When a request's draft accepts poorly, the worker runs the target's one-token DECODE step (its own CUDA graph,
captured beside the verify graphs) instead of draft + 4-row verify, writes the draft KV of that position from the
captured hidden, and returns an accept length of 1. The logic is the new module dspark_plain_gate.py (next to this
script, pinned below); the worker gets six guarded hunks:

- a module constant read once at import (_PLAIN_GATE_ON) and a class default (_plain_gate = None);
- __init__: the controller, only with the selector on (it refuses a server it was not built for);
- init_cuda_graphs: the plain DECODE graph capture, before the draft's graphs;
- _forward_decode: the gate, after the idle return and before the draft;
- _forward_decode (r3): the confidence signal of a verify step, right after the target verify returns and before
  the grammar mask, accept or sampling read its logits (the signal only reads them);
- note_request_finished: drop the request's gate state.

With the selector unset the module is never imported and each hunk is a false test, so steps run as before.
The other pins are files the module reads but this overlay does not change; a changed one fails the overlay,
because the module depends on their internals (see PLAIN_GATE_r1.md). argv: sglang package, patched in place."""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_SRC = HERE / 'dspark_plain_gate.py'
MODULE_PIN = 'df7d0ee67d3c64013de04930ef9ad212ce9e3957b0e6b031158817e5013028ce'
MODULE_DST = 'srt/speculative/dspark_components/dspark_plain_gate.py'
WORKER = 'srt/speculative/dspark_components/dspark_worker_v2.py'
PINS = {
    WORKER: '8eef0b1af504a5fabcc4a2028b98ac5b00aeec803cb976ca53af7d98b5b9e3d6',
    # read by the module, unchanged here
    'srt/speculative/dspark_components/dspark_verify.py': 'f794385100777b5f30f96132b815cc74786f34d0b874210b959d42b117bad123',
    'srt/speculative/dspark_components/dspark_draft.py': 'f1886af3a0598dbf48242960c8b58cce86a2876074f376036d258b65eca09f7e',
    'srt/speculative/dspark_components/dspark_kv_inject.py': '16c303b1e2f6b4cd5e1766084a498abfe193cb8223711e184ae9e09e780e8a38',
    'srt/speculative/dspark_components/dspark_planner.py': 'f95e7cd3ceb430fcaaab77b342bf1123a09adcb35917ecc85500052ac2c8cfcc',
    'srt/speculative/dspark_components/dspark_deferred_commit.py': 'c5cd906defe806e04f6f5a67ae2867ba4681980efe6ac60d33019298c4d966a8',
    'srt/speculative/spec_tp_sync.py': '224fdb1aa782044fc7d253b72962c19ed105db39b772b19a3b2b71ebf799ce3f',
    'srt/speculative/dflash_info_v2.py': 'c66fd2a9629c5f0f36ad4eca34f2910dc8e8af509fd8c6b2fa3fd2722f18004f',
    'srt/model_executor/runner/decode_cuda_graph_runner.py': 'b81c6f19ad8f39269e163b2aa38a0515bcb6fec5f0e885a49c12973418e04fae',
    'srt/model_executor/model_runner.py': 'efb9fd670c913039806a178ac2b460819f95dd0a590ae0c3ad4eb171ba82a51c',
    'srt/model_executor/forward_batch_info.py': 'ec43cb98a6ad635a5044cd2de84b101ff72d600fb36d59be4302734999479879',
    'srt/model_executor/input_buffers.py': '893f105c50f93e141ce7cb457f4ae9c8864609a2c58697f5a9e59a343bd55aee',
    'srt/mem_cache/kv/alloc/iteration_ids.py': '12560c387d4a09221c5bfa537a20b3c765b56948f215f5d458f44af5fdaa3665',
    'srt/managers/schedule_batch.py': 'fc1acedc503d7dab13a3570793730320491ec8ddfbc8ac47a3f9e2aa0f26fe9d',
    'srt/managers/tp_worker.py':'c4e66a1fd07cdadd8ba7cf9e69fe8ca581b6e5f823135ea9652be03836e0429d',
    'srt/managers/utils.py': '7a12a5d0827e2de29c0f7cf36064edcb4fbcbb4fa60dd1ec75244fc2f14d9e2e',
    'srt/managers/scheduler_components/batch_result_processor.py': '58e1006d617c6269e0ca4f184431e9723eb98229e5d788b5b2b947dff22e1d1b',
    'srt/layers/attention/deepseek_v4_backend.py': '1c6ccdd48c1a69bf8fad8e67fd482972537e47ceafbf77411dc5bbcebadd6ff6',
    'srt/layers/engram.py': 'd440482dd9ae49d6ea7435f639c5722a08a2a4b4fb4f9ee86f3ed54b148327f2',
    'srt/layers/moe/contract_v2/patha_static_runtime.py': '218d3d8b0d69b141c0999f5a59fd8816fd80a5c1648b95cddf75b963dcfa9df4',
    # r3: the confidence signal reads the plain step's buffer after Sampler.forward (in-place softmax for a sampled
    # batch), LogitsProcessorOutput.next_token_logits and SamplingBatchInfo.temperatures / is_all_greedy
    'srt/layers/sampler.py': 'b9b0c965731104a5a7a1565bc5927d1ddb6dea278f19e3beb981382c468d63e6',
    'srt/layers/logits_processor.py': '51ce9204829bc3b3b2c97346acf342a366e21e06b6f2947bf8cf968dcb8999e7',
    'srt/sampling/sampling_batch_info.py': '5ddd691c5a3bd5e7f78fb4700012e00da75baecfa5c2dde2adf3766ca3fdfb90',
    # r4: the draft-only probe takes a full-width verify window, which must stay a pure plan: plan_uniform_extend
    # gathers reserved slot ids from req_to_token into new tensors (the gather kernel stores only its outputs)
    'srt/mem_cache/kv/alloc/runner_ids.py': '4b3c95edd38410a561bae37d5ffaed4814c80ffb65dbf65ce290875cbb729139',
    'kernels/ops/speculative/cache_locs.py': '05a0bf25a31b334b88275d803bf9da19b4ffd01f8521e94380e00d544bb0a586',
}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main(root):
    root = Path(root)
    text = {}
    for rel, pin in PINS.items():
        s = (root / rel).read_bytes().decode('utf-8')
        assert sha(s) == pin, f'{rel} differs: {sha(s)}'
        text[rel] = s
    assert not (root / MODULE_DST).exists(), f'{MODULE_DST} exists already'
    module = MODULE_SRC.read_bytes().decode('utf-8')
    assert sha(module) == MODULE_PIN, f'dspark_plain_gate.py differs: {sha(module)}'

    s = text[WORKER]

    def sub(old, new):
        nonlocal s
        assert s.count(old) == 1, (s.count(old), old[:80])
        s = s.replace(old, new)

    sub('_is_npu = is_npu()\n\n\n@runtime_checkable\n',
        '_is_npu = is_npu()\n'
        '\n'
        '# KT_DSPARK_PLAIN_GATE (dspark-cpu-r1/apply_dspark_plain_gate.py): plain decode steps for a request whose draft\n'
        '# accepts poorly. Read once; unset, the gate module is never imported and no step changes.\n'
        '_PLAIN_GATE_ON = os.environ.get("KT_DSPARK_PLAIN_GATE", "0") == "1"\n'
        '\n'
        '\n'
        '@runtime_checkable\n')
    sub('    _verify_inputs_ready = None\n\n    def weight_update_runners(',
        '    _verify_inputs_ready = None\n'
        '    # KT_DSPARK_PLAIN_GATE: the plain-step controller (dspark_plain_gate.PlainStepController), or None.\n'
        '    _plain_gate = None\n'
        '\n'
        '    def weight_update_runners(')
    sub('        if self._is_pd_prefill and not self._draft_is_moe:\n'
        '            self.draft_model.prune_to_ctx_kv_injection()\n',
        '        if _PLAIN_GATE_ON:\n'
        '            from sglang.srt.speculative.dspark_components.dspark_plain_gate import (\n'
        '                PlainStepController,\n'
        '            )\n'
        '\n'
        '            self._plain_gate = PlainStepController.from_env(worker=self)\n'
        '\n'
        '        if self._is_pd_prefill and not self._draft_is_moe:\n'
        '            self.draft_model.prune_to_ctx_kv_injection()\n')
    sub('    def init_cuda_graphs(self):\n'
        '        if not self._hosts_draft:\n'
        '            return\n'
        '        capture_decode_cuda_graph = self._decode_graph_allowed\n',
        '    def init_cuda_graphs(self):\n'
        '        if not self._hosts_draft:\n'
        '            return\n'
        '        if self._plain_gate is not None:\n'
        '            # KT_DSPARK_PLAIN_GATE: the target\'s plain DECODE graphs, beside its verify graphs and before the\n'
        '            # draft\'s graphs take their memory.\n'
        '            self._plain_gate.capture_plain_graph()\n'
        '        capture_decode_cuda_graph = self._decode_graph_allowed\n')
    sub('        batch.seq_lens.record_stream(\n'
        '            torch.get_device_module(self.device).current_stream()\n'
        '        )\n'
        '        bs = len(batch.seq_lens)\n',
        '        batch.seq_lens.record_stream(\n'
        '            torch.get_device_module(self.device).current_stream()\n'
        '        )\n'
        '        if self._plain_gate is not None and coordination_plan is None:\n'
        '            # KT_DSPARK_PLAIN_GATE: a request whose draft accepts poorly takes the target\'s plain decode step.\n'
        '            plain_result = self._plain_gate.maybe_plain_step(batch, on_publish)\n'
        '            if plain_result is not None:\n'
        '                return plain_result\n'
        '        bs = len(batch.seq_lens)\n')
    sub('        logits_output = target_verify.logits_output\n'
        '        can_run_cuda_graph = target_verify.can_run_cuda_graph\n',
        '        logits_output = target_verify.logits_output\n'
        '        can_run_cuda_graph = target_verify.can_run_cuda_graph\n'
        '        if self._plain_gate is not None:\n'
        '            # KT_DSPARK_PLAIN_GATE r3: the step\'s confidence signal, before anything reads or masks the logits.\n'
        '            self._plain_gate.note_verify_logits(batch, logits_output.next_token_logits)\n')
    sub('        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)\n',
        '        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)\n'
        '        if self._plain_gate is not None:\n'
        '            self._plain_gate.forget(rid)\n')

    compile(s, WORKER, 'exec')
    compile(module, MODULE_DST, 'exec')
    (root / WORKER).write_bytes(s.encode('utf-8'))
    (root / MODULE_DST).write_bytes(module.encode('utf-8'))
    print(WORKER, sha(s))
    print(MODULE_DST, sha(module))


if __name__ == '__main__':
    main(sys.argv[1])
