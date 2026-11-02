"""Native edit: KT_EXPERT_ARENA_REBALANCE_NODE0=3, a sixteenth of node 0's stripes to node 3. argv: expert_arena_spill.hpp."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
PAIRS = [
    ('  const bool rebalance = rebalance_env && std::strcmp(rebalance_env, "1") == 0;\n  if (rebalance) {\n',
     '  const bool rebalance = rebalance_env && std::strcmp(rebalance_env, "1") == 0;\n'
     "  // Value 3: a sixteenth of node 0's stripes to node 3, its own socket's spill node, in place of the eighth across the compute nodes.\n"
     "  // The eighth left node 0's part 32% slower than the others; with no rebalance its usable memory fell to the guard in a 35K prefill.\n"
     '  const bool rebalance_socket = rebalance_env && std::strcmp(rebalance_env, "3") == 0;\n'
     '  if (rebalance || rebalance_socket) {\n'),
    ('    if (stagger) {\n      if (bytes > local_bytes) place(offset, bytes, local_bytes, bytes - local_bytes, &stagger_node);\n',
     '    if (rebalance_socket && local == 1UL) {\n'
     '      const size_t extra = (bytes / 16 / hugepage) * hugepage;\n'
     '      const unsigned long target = 1UL << 3;\n'
     '      if (extra) place(offset, bytes, local_bytes - extra, extra, &target);\n'
     '    }\n'
     '    if (stagger) {\n      if (bytes > local_bytes) place(offset, bytes, local_bytes, bytes - local_bytes, &stagger_node);\n'),
]
for old, new in PAIRS:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
p.write_text(s)
