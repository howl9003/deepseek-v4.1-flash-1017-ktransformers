"""Shared input digits at one token (Claude, 2 Nov 2026), selector KT_CONTRACT_V2_SHARED_X, default off. At qlen 1 every route in a part reads the same row, so one quantisation serves them all by pointer. Same digits, so same bits. argv: header."""
import hashlib
import sys
from pathlib import Path

BASE = '1fdad87ce7d0be95'
p = Path(sys.argv[1])
s = p.read_bytes().decode('utf-8')
assert hashlib.sha256(s.encode('utf-8')).hexdigest().startswith(BASE), 'header differs'
for a, b in (
    ("    void from_mat(int m, const ggml_bf16_t* src, int /*ith*/, int /*nth*/) { from_rows(0, m, src); }\n  };\n",
     "    void from_mat(int m, const ggml_bf16_t* src, int /*ith*/, int /*nth*/) { from_rows(0, m, src); }\n"
     "    // Read another buffer's digits. Only for rows that are the same input.\n"
     "    void alias(const BufferA& o) { codes = o.codes; ea = o.ea; EA = o.EA; hi = o.hi; lo = o.lo; hi_f = o.hi_f; lo_f = o.lo_f; EAmin = o.EAmin; SA = o.SA; }\n"
     "  };\n"),
    ("  static bool route_ref() {\n",
     "  // KT_CONTRACT_V2_SHARED_X=1: at one token every route in a part reads the same input row, so one quantisation serves them all.\n"
     "  static bool shared_x() {\n"
     "    static const bool on = [] { const char* value = std::getenv(\"KT_CONTRACT_V2_SHARED_X\"); return value && value[0] == '1'; }();\n"
     "    return on;\n"
     "  }\n"
     "  static bool route_ref() {\n"),
    ("    plan_quant(4);\n    pool->do_work_stealing_job((int)qtasks.size(), nullptr,\n        [this, &qtasks](int t) { const QTask& q = qtasks[t]; gate_up_ba_[q.e]->from_rows(q.r0, q.r1, m_local_input_ptr_[q.e]); }, nullptr);\n",
     "    bool one_row = shared_x() && qlen == 1;\n"
     "    for (int t = 0; one_row && t < activated_expert; t++) one_row = m_local_num_[m_expert_id_map_[t]] == 1;\n"
     "    if (one_row) {\n"
     "      const int first = m_expert_id_map_[0];\n"
     "      gate_up_ba_[first]->from_rows(0, 1, m_local_input_ptr_[first]);\n"
     "      for (int t = 1; t < activated_expert; t++) gate_up_ba_[m_expert_id_map_[t]]->alias(*gate_up_ba_[first]);\n"
     "    } else {\n"
     "      plan_quant(4);\n"
     "      pool->do_work_stealing_job((int)qtasks.size(), nullptr,\n"
     "          [this, &qtasks](int t) { const QTask& q = qtasks[t]; gate_up_ba_[q.e]->from_rows(q.r0, q.r1, m_local_input_ptr_[q.e]); }, nullptr);\n"
     "    }\n"),
):
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
p.write_bytes(s.encode('utf-8'))
print(hashlib.sha256(s.encode('utf-8')).hexdigest())
