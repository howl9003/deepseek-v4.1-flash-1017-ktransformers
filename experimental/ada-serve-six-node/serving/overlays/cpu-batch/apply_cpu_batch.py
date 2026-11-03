"""The other two CPU window fixes (Claude, 3 Nov 2026), on top of apply_shared_x.py, each behind its own selector, default off.

KT_CONTRACT_V2_FUSED_PHASES=1, decode (qlen <= 8): gate/up, activation, the second quantisation and down run as one pool job
instead of four. A task waits only for its own expert's previous phase, so a part's threads no longer stop at three barriers a layer.
The pool hands tasks out one at a time in increasing order from one counter, and every task waits only on earlier tasks,
so the earliest unfinished task can always run: no deadlock.

KT_CONTRACT_V2_MERGE_IN_PARTS=1: each part merges its share of the column chunks inside its own NUMA job, once every part has
finished, instead of in a second job after them.

Both are scheduling only: every task is the same call on the same split, the merge per column is unchanged, so the bits are the same.
argv: the shared-x header, patched in place."""
import hashlib
import sys
from pathlib import Path

SHARED_X = '7ab650542cf02fb2ba92a1fde0c678abfaeb4b65154f417fc69b770aaf3a5e6f'
p = Path(sys.argv[1])
s = p.read_bytes().decode('utf-8')
assert hashlib.sha256(s.encode('utf-8')).hexdigest() == SHARED_X, 'header is not the shared-x header'


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


sub("  static bool route_ref() {\n",
    "  // KT_CONTRACT_V2_FUSED_PHASES=1: at decode, the phases after the input quantisation run as one pool job (see forward_contract).\n"
    "  static bool fused_phases() {\n"
    "    static const bool on = [] { const char* value = std::getenv(\"KT_CONTRACT_V2_FUSED_PHASES\"); return value && value[0] == '1'; }();\n"
    "    return on;\n"
    "  }\n"
    "  static bool route_ref() {\n")

old_phases = s[s.index("    // Gate and up projections, finished to BF16\n"):s.index("    // Per (token, slot): the partial row and its base\n")]
assert old_phases.count('do_work_stealing_job') == 4
fused = '''    if (fused_phases() && qlen <= 8) {
      // One job for gate/up, activation, the second quantisation and down. A task waits only for its own expert's previous
      // phase. The pool deals tasks one at a time in increasing order and every wait is on an earlier task, so the earliest
      // unfinished task can always run. Each task is the same call on the same split as the phased path below.
      const int A = activated_expert;
      const int nu = nth_for(config_.intermediate_size, config_.hidden_size), nd = nth_for(config_.hidden_size, config_.intermediate_size);
      struct FTask { int t, e, r0, r1; };
      std::vector<FTask> q2;
      std::vector<int> q2_count(A, 0);
      for (int t = 0; t < A; t++) {
        const int e = m_expert_id_map_[t];
        for (int r0 = 0; r0 < m_local_num_[e]; r0 += 8) { q2.push_back({t, e, r0, std::min(m_local_num_[e], r0 + 8)}); q2_count[t]++; }
      }
      std::unique_ptr<std::atomic<int>[]> done(new std::atomic<int>[3 * A]);  // per expert slot: gate/up, activation, quantisation
      for (int i = 0; i < 3 * A; i++) done[i].store(0, std::memory_order_relaxed);
      const int n_gu = 2 * nu * A, n_act = nu * A, n_q2 = (int)q2.size(), n_down = nd * A;
      auto wait_for = [&done](int slot, int need) { while (done[slot].load(std::memory_order_acquire) < need) _mm_pause(); };
      mark(); mark(); mark();
      pool->do_work_stealing_job(n_gu + n_act + n_q2 + n_down, [](int) { T::config(); },
          [&](int id) {
            if (id < n_gu) {
              const int task_id = id / 2; const bool do_up = id % 2;
              const int t = task_id / nu, e = m_expert_id_map_[t], ith = task_id % nu;
              do_gate_up_gemm(do_up, e, ith, nu, 0);
              if (do_up) up_bc_[e]->to_mat(m_local_num_[e], m_local_up_output_ptr_[e], ith, nu);
              else gate_bc_[e]->to_mat(m_local_num_[e], m_local_gate_output_ptr_[e], ith, nu);
              done[3 * t].fetch_add(1, std::memory_order_acq_rel);
              return;
            }
            id -= n_gu;
            if (id < n_act) {
              const int t = id / nu;
              wait_for(3 * t, 2 * nu);
              apply_activation_contract(m_expert_id_map_[t], id % nu, nu);
              done[3 * t + 1].fetch_add(1, std::memory_order_acq_rel);
              return;
            }
            id -= n_act;
            if (id < n_q2) {
              const FTask& q = q2[id];
              wait_for(3 * q.t + 1, nu);
              down_ba_[q.e]->from_rows(q.r0, q.r1, m_local_gate_output_ptr_[q.e]);
              done[3 * q.t + 2].fetch_add(1, std::memory_order_acq_rel);
              return;
            }
            id -= n_q2;
            const int t = id / nd;
            wait_for(3 * t + 2, q2_count[t]);
            do_down_gemm(m_expert_id_map_[t], id % nd, nd, 0);
          }, nullptr);
      mark();
    } else {
'''
indented = ''.join(('  ' + line if line.strip() else line) for line in old_phases.splitlines(keepends=True))
sub(old_phases, fused + indented + "    }\n")

# Merge inside the parts' own job
sub("  void run_parts(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input) {\n",
    "  // KT_CONTRACT_V2_MERGE_IN_PARTS=1: each part merges its share of the column chunks inside its own NUMA job once every\n"
    "  // part has finished, instead of in a second job after them. Column chunks, per-route arithmetic and part order are unchanged.\n"
    "  static bool merge_in_parts() {\n"
    "    static const bool on = [] { const char* value = std::getenv(\"KT_CONTRACT_V2_MERGE_IN_PARTS\"); return value && value[0] == '1'; }();\n"
    "    return on;\n"
    "  }\n"
    "  template <class F>\n"
    "  void run_parts_and_merge(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, int chunks, F&& fn) {\n"
    "    if (!merge_in_parts()) { run_parts(qlen, k, expert_ids, weights, input); run_merge(qlen, chunks, fn); return; }\n"
    "    if (!this->weights_loaded) throw std::runtime_error(\"Not Loaded\");\n"
    "    const int parts = this->tp_count;\n"
    "    std::atomic<int> arrived{0};\n"
    "    this->config.pool->dispense_backend()->do_numa_job([&](int p) {\n"
    "      part(p)->forward_contract(qlen, k, expert_ids, weights, input);\n"
    "      arrived.fetch_add(1, std::memory_order_acq_rel);\n"
    "      while (arrived.load(std::memory_order_acquire) < parts) _mm_pause();\n"
    "      const int c0 = (chunks * p) / parts, c1 = (chunks * (p + 1)) / parts, w = c1 - c0;\n"
    "      if (w <= 0) return;\n"
    "      auto sub = [&](int i) { fn((i / w) * chunks + c0 + (i % w)); };\n"
    "      this->config.pool->get_subpool(p)->do_work_stealing_job(qlen * w, nullptr, sub, nullptr);\n"
    "    });\n"
    "  }\n"
    "  void run_parts(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input) {\n")

# forward(): define the merge first, then run parts and merge together
sub("    const int qlen = *qlen_ptr;\n    run_parts(qlen, k, expert_ids, weights, input);\n    auto& c = this->config;\n    const int chunk = 256;\n",
    "    const int qlen = *qlen_ptr;\n    auto& c = this->config;\n    const int chunk = 256;\n")
sub("    if (k > 64) throw std::runtime_error(\"contract v2: topk > 64\");\n    run_merge(qlen, chunks, fn);\n",
    "    if (k > 64) throw std::runtime_error(\"contract v2: topk > 64\");\n    run_parts_and_merge(qlen, k, expert_ids, weights, input, chunks, fn);\n")
# forward_routes_binding()
sub("    run_parts(qlen, k, ids, weights, (const void*)input);\n    const int chunk = 256, chunks = (c.hidden_size + chunk - 1) / chunk;\n",
    "    const int chunk = 256, chunks = (c.hidden_size + chunk - 1) / chunk;\n")
# the one run_merge left at the start of a line is forward_routes_binding's
sub("    run_merge(qlen, chunks, fn);\n", "    run_parts_and_merge(qlen, k, ids, weights, (const void*)input, chunks, fn);\n")
p.write_bytes(s.encode('utf-8'))
print(hashlib.sha256(s.encode('utf-8')).hexdigest())
