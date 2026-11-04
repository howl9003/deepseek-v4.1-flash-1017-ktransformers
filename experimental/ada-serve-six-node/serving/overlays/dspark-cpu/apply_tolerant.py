"""Native patch: KT_CONTRACT_V2_TOLERANT_FROM=<KT layer id>, default off (Claude, 3 Nov 2026).
argv: kt-kernel tree (operators/avx2/contract_v2-moe.hpp and ext_bindings.cpp are patched in place; run on a copy).

The draft's stage-2 down projection reaches the contract's truncation regime on ordinary inputs (test_draft_forward.py:
a term more than 20 bits below the row's top). kt-kernel then refuses the call, and its truncation counter is process
wide and never reset, so one draft call would stop every later target call too. With the variable set, layers at or above
the id (the DSpark draft's KT layers 40-42) keep the floor-per-part sum (deterministic for the six-part layout) and move
their events to a separate counter; every other layer, and every layer without the variable, is unchanged."""
import sys
from pathlib import Path

KT = Path(sys.argv[1])


def sub(s, old, new, n=1):
    assert s.count(old) == n, (s.count(old), old[:90])
    return s.replace(old, new)


p = KT / 'operators/avx2/contract_v2-moe.hpp'
s = p.read_text(encoding='utf-8')
s = sub(s, 'static std::atomic<long> g_truncation_events{0};\n',
        'static std::atomic<long> g_truncation_events{0};\n'
        'static std::atomic<long> g_tolerant_events{0};   // KT_CONTRACT_V2_TOLERANT_FROM layers\' events\n')
s = sub(s, 'inline long kt_contract_v2_truncation_events() { return kt_contract_v2::g_truncation_events.load(); }\n',
        'inline long kt_contract_v2_truncation_events() { return kt_contract_v2::g_truncation_events.load(); }\n'
        'inline long kt_contract_v2_tolerant_events() { return kt_contract_v2::g_tolerant_events.load(); }\n')
s = sub(s, '  // Exact merge of the parts\' partials for one (token, slot)',
        '  // KT_CONTRACT_V2_TOLERANT_FROM=<KT layer id>, default off (SGLang KT_DSPARK_CPU_DRAFT): a layer at or above the id\n'
        '  // (the DSpark draft\'s, 40-42) keeps the floor-per-part sum in the truncation regime instead of refusing. Its events\n'
        '  // move from the process-wide counter, which every other layer checks, to g_tolerant_events, so a draft event never\n'
        '  // stops the target. One MoE task runs at a time on the CPUInfer queue: the counter\'s change during this call is\n'
        '  // this call\'s. Below the id, and without the variable, nothing changes.\n'
        '  bool tolerant() const {\n'
        '    static const int from = [] { const char* v = std::getenv("KT_CONTRACT_V2_TOLERANT_FROM"); return v && *v ? std::atoi(v) : -1; }();\n'
        '    return from >= 0 && this->config.layer_idx >= from;\n'
        '  }\n'
        '  void settle_tolerant(long before) {\n'
        '    const long now = kt_contract_v2_truncation_events();\n'
        '    if (now > before) {\n'
        '      kt_contract_v2::g_truncation_events.fetch_sub(now - before, std::memory_order_relaxed);\n'
        '      kt_contract_v2::g_tolerant_events.fetch_add(now - before, std::memory_order_relaxed);\n'
        '    }\n'
        '  }\n'
        '\n'
        '  // Exact merge of the parts\' partials for one (token, slot)')
s = sub(s, '    const int qlen = *qlen_ptr;\n    run_parts(qlen, k, expert_ids, weights, input);\n',
        '    const int qlen = *qlen_ptr;\n    const long trunc_before = kt_contract_v2_truncation_events();\n    run_parts(qlen, k, expert_ids, weights, input);\n')
s = sub(s, '    run_parts(qlen, k, ids, weights, (const void*)input);\n',
        '    const long trunc_before = kt_contract_v2_truncation_events();\n    run_parts(qlen, k, ids, weights, (const void*)input);\n')
s = sub(s, '    run_merge(qlen, chunks, fn);\n    if (kt_contract_v2_truncation_events() != 0)\n',
        '    run_merge(qlen, chunks, fn);\n    if (tolerant()) settle_tolerant(trunc_before);\n    if (kt_contract_v2_truncation_events() != 0)\n', 2)
p.write_bytes(s.encode('utf-8'))

p = KT / 'ext_bindings.cpp'
s = p.read_text(encoding='utf-8')
s = sub(s, '  m.def("contract_v2_truncation_events", &kt_contract_v2_truncation_events);\n',
        '  m.def("contract_v2_truncation_events", &kt_contract_v2_truncation_events);\n'
        '  m.def("contract_v2_tolerant_events", &kt_contract_v2_tolerant_events);\n')
p.write_bytes(s.encode('utf-8'))
print('patched')
