"""CPU-only checks of real prefix-policy ASTs; no Torch/server/model imports.

The eviction harness models leaf removal and session ordering, not tensor freeing,
HiCache transfer, or the full scheduler. Passing is not GPU/integration acceptance.
Run with: python -B tools/test_prefix_cache_policy.py [sglang-root]
"""
from __future__ import annotations

import ast
import argparse
import logging
import math
import os
from pathlib import Path
import time
import types
import sys
import unittest
from abc import ABC, abstractmethod
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch


SRT = Path(__file__).resolve().parents[1] / "third_party/sglang/python/sglang/srt"


def extract(path, names, namespace=None, owner=None, assignments=()):
    """Compile selected, unmodified definitions from the actual draft source."""
    path = SRT / path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if owner:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    picked = [
        n for n in body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in assignments for t in n.targets)
    ]
    found = {n.name for n in picked if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if found != set(names):
        raise AssertionError(f"missing source definitions in {path}: {set(names) - found}")
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + picked, type_ignores=[])
    result = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), result)
    return result


def policy_namespace():
    return extract(
        "mem_cache/evict_policy.py",
        {"EvictionStrategy", "LRUStrategy", "CacheTTLPolicy", "TTLStrategy",
         "cache_ttl_enabled", "normalize_cache_class", "cache_class_from_kv_hints",
         "cache_class_of_req", "with_cache_class_hint"},
        {"ABC": ABC, "abstractmethod": abstractmethod, "_os": os, "_math": math, "_time": time},
        assignments={"CACHE_CLASS_USER", "CACHE_CLASS_AGENT", "CACHE_CLASSES",
                     "CACHE_CLASS_HINT_TYPE", "CACHE_CLASS_HINT_VERSION"},
    )


class Clock:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


class Node:
    def __init__(self, name, parent=None, tokens=1, session_ref=0, locked=False):
        self.name, self.parent = name, parent
        self.children = {}
        self.key = list(range(tokens))
        self.last_access_time = 0
        self.session_ref, self.locked = session_ref, locked
        self.write_through_pending_id = None
        self.component_data = {"swa": NS(value=list(range(tokens)), host_value=None)}
        if parent is not None:
            parent.children[name] = self


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.ns = policy_namespace()
        self.Policy = self.ns["CacheTTLPolicy"]
        self.clock = Clock()
        self.policy = self.Policy(agent_s=10, user_s=100, sweep_s=5, clock=self.clock)
        self.root = Node("root", tokens=0)

    def test_default_off_ignores_invalid_optional_environment(self):
        with patch.dict(os.environ, {"SGLANG_CACHE_TTL_AGENT_S": "not-a-number"}, clear=True):
            self.assertFalse(self.ns["cache_ttl_enabled"]())
            self.assertIsNone(self.Policy.from_env())
        with patch.dict(os.environ, {"SGLANG_CACHE_TTL": "true"}, clear=True):
            self.assertIsNone(self.Policy.from_env())

    def test_environment_defaults_and_overrides(self):
        with patch.dict(os.environ, {"SGLANG_CACHE_TTL": "1"}, clear=True):
            p = self.Policy.from_env()
            self.assertEqual((p.agent_s, p.user_s, p.sweep_s, p.relend_tokens), (1800, 43200, 60, 0))
        with patch.dict(os.environ, {"SGLANG_CACHE_TTL": "1", "SGLANG_CACHE_TTL_AGENT_S": "3.5",
                                   "SGLANG_CACHE_TTL_USER_S": "12", "SGLANG_CACHE_TTL_SWEEP_S": "2",
                                   "SGLANG_CACHE_TTL_RELEND_TOKENS": "17"}, clear=True):
            p = self.Policy.from_env()
            self.assertEqual((p.agent_s, p.user_s, p.sweep_s, p.relend_tokens), (3.5, 12, 2, 17))

    def test_finite_positive_ttls_and_sweep_nonnegative_relend(self):
        for field in ("agent_s", "user_s", "sweep_s"):
            for value in (0, -1, math.inf, -math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.Policy(**{field: value})
        with self.assertRaises(ValueError):
            self.Policy(relend_tokens=-1)
        with patch.dict(os.environ, {"SGLANG_CACHE_TTL": "1", "SGLANG_CACHE_TTL_SWEEP_S": "nan"}, clear=True):
            with self.assertRaises(ValueError):
                self.Policy.from_env()

    def test_mixed_shared_prefix_preserves_later_class_deadline(self):
        shared = Node("shared", self.root)
        agent, user = Node("agent", shared), Node("user", shared)
        self.policy.stamp_path(agent, self.root, "agent")
        self.clock.now = 5
        self.policy.stamp_path(user, self.root, "user")
        self.assertEqual(self.policy.deadline(shared, 5), 105)
        self.assertEqual(self.policy.deadline(agent, 5), 10)
        self.assertFalse(hasattr(self.root, "_ttl_user_ts"))
        self.clock.now = 10
        self.assertFalse(self.policy.is_expired(agent))
        self.clock.now = 10.001
        self.assertTrue(self.policy.is_expired(agent))
        self.assertFalse(self.policy.is_expired(shared))
        self.clock.now = 104
        self.policy.stamp_path(agent, self.root, "agent")
        self.assertEqual(self.policy.deadline(shared, 104), 114)

    def test_split_chain_inherits_children_before_they_are_removed(self):
        fragment = Node("fragment", self.root)
        inner = Node("inner", fragment)
        tail = Node("tail", inner)
        tail._ttl_agent_ts, tail._ttl_user_ts = 2, 3
        self.assertEqual(self.policy.resolve_unstamped(self.root, 90), 2)
        inner.children.clear()
        self.assertEqual(self.policy.deadline(fragment, 200), 103)
        self.assertEqual(self.policy.deadline(inner, 200), 103)
        self.assertTrue(self.policy.is_expired(fragment, 104))

    def test_split_parent_touched_by_agent_preserves_existing_user_lifetime(self):
        fragment = Node("new-split-parent", self.root)
        old_user_tail = Node("existing-user-tail", fragment)
        old_user_tail._ttl_user_ts = 3
        self.clock.now = 20
        self.policy.stamp_path(fragment, self.root, "agent")
        self.assertEqual(self.policy.deadline(fragment, 20), 103)
        self.assertEqual(fragment._ttl_user_ts, 3)
        self.assertEqual(fragment._ttl_agent_ts, 20)
        self.assertFalse(self.policy.is_expired(fragment, 31))

    def test_nested_split_then_agent_touch_keeps_original_user_stamp(self):
        outer = Node("outer-fragment", self.root)
        inner = Node("inner-fragment", outer)
        tail = Node("existing-user-tail", inner)
        tail._ttl_user_ts = 3
        self.clock.now = 20
        self.policy.stamp_path(outer, self.root, "agent")
        self.assertEqual(self.policy.deadline(outer, 20), 103)
        self.assertEqual(inner._ttl_user_ts, 3)
        self.assertEqual(outer._ttl_user_ts, 3)
        self.assertEqual(self.policy.deadline(inner, 20), 103)
        self.assertTrue(self.policy.is_expired(outer, 104))

    def test_fresh_agent_leaf_does_not_invent_user_lifetime(self):
        leaf = Node("fresh-agent", self.root)
        self.clock.now = 20
        self.policy.stamp_path(leaf, self.root, "agent")
        self.assertIsNone(getattr(leaf, "_ttl_user_ts", None))
        self.assertEqual(self.policy.deadline(leaf, 20), 30)
        self.assertTrue(self.policy.is_expired(leaf, 31))

    def test_unstamped_leaf_is_conservative_once_and_then_expires(self):
        node = Node("internal", self.root)
        self.assertEqual(self.policy.deadline(node, 20), 120)
        self.assertEqual(self.policy.deadline(node, 119), 120)
        self.assertTrue(self.policy.is_expired(node, 121))

    def test_ttl_strategy_expired_first_and_inner_order_preserved(self):
        old, live, older_expired = [Node(n, self.root) for n in ("old", "live", "older")]
        self.policy.stamp_path(old, self.root, "agent", 0)
        self.policy.stamp_path(older_expired, self.root, "agent", 0)
        self.policy.stamp_path(live, self.root, "user", 0)
        old.last_access_time, live.last_access_time, older_expired.last_access_time = 100, -10, 20
        self.clock.now = 11
        inner = self.ns["LRUStrategy"]()
        inner.threshold = 7
        strategy = self.ns["TTLStrategy"](inner, self.policy)
        self.assertEqual(sorted([live, old, older_expired], key=strategy.get_priority), [older_expired, old, live])
        self.assertEqual(strategy.threshold, 7)


class SweepTests(unittest.TestCase):
    setUp = PolicyTests.setUp

    def make_cache(self, nodes, enabled=True):
        ns = extract("mem_cache/unified_radix_cache.py", {"ttl_sweep", "maybe_ttl_sweep", "_cache_ttl_stamp"},
                     {"BASE_COMPONENT_TYPE": "full", "EvictParams": NS, "time": time, "logger": Mock()},
                     owner="UnifiedRadixCache")
        cache = type("ExtractedCache", (), {k: ns[k] for k in ("ttl_sweep", "maybe_ttl_sweep", "_cache_ttl_stamp")})()
        leaves = {n for n in nodes if not n.children and not n.locked}
        arena = {n.name: n for n in [self.root] + nodes}
        cache.tree_core = NS(root_node=self.root, evictable_device_leaves=leaves, _node_arena=arena)
        cache.components = {"full": NS(session_ref=lambda n: n.session_ref)}
        cache.enable_session_radix_cache = True
        cache._cache_ttl = self.policy if enabled else None
        cache.disable = False
        cache.full_evictable_size = lambda: sum(len(n.key) for n in arena.values())
        cache.evictable_size = cache.full_evictable_size
        cache.token_to_kv_pool_allocator = NS(available_size=lambda: 100)
        cache._cache_ttl_relend = Mock()
        cache._cache_ttl_relend_check = Mock()
        cache.victims = []
        strategy = self.ns["TTLStrategy"](self.ns["LRUStrategy"](), self.policy)

        def evict(params, allocation_reclaim_satisfied):
            # Fixture models normal session partition + eviction order/removal only.
            while not allocation_reclaim_satisfied():
                node = min(leaves, key=lambda n: (n.session_ref > 0, strategy.get_priority(n)))
                self.assertFalse(node.locked)
                self.assertEqual(node.session_ref, 0)
                cache.victims.append(node)
                leaves.remove(node)
                arena.pop(node.name)
                node.parent.children.pop(node.name)
                if node.parent is not self.root and not node.parent.children and not node.parent.locked:
                    leaves.add(node.parent)
            return NS(num_tokens_evicted=sum(len(n.key) for n in cache.victims))

        cache._evict = Mock(side_effect=evict)
        return cache

    def test_sweep_cascades_expired_parents_preserves_live_locked_and_session(self):
        parent = Node("parent", self.root, tokens=3)
        tail = Node("tail", parent, tokens=2)
        live = Node("live", self.root, tokens=5)
        locked = Node("locked", self.root, locked=True)
        held = Node("held", self.root, session_ref=1)
        for n in (tail, locked, held):
            self.policy.stamp_path(n, self.root, "agent", 0)
        self.policy.stamp_path(live, self.root, "user", 0)
        cache = self.make_cache([parent, tail, live, locked, held])
        self.clock.now = 11
        self.assertEqual(cache.ttl_sweep(), 5)
        self.assertEqual(cache.victims, [tail, parent])
        self.assertEqual(set(cache.tree_core._node_arena), {"root", "live", "locked", "held"})
        self.assertEqual(self.policy.stats, {"sweeps": 1, "swept_tokens": 5, "swept_nodes": 2, "relends": 0})
        self.assertEqual(cache.ttl_sweep(), 0)
        self.assertEqual(cache._evict.call_count, 1)

    def test_sweep_off_and_disabled_never_enter_evictor(self):
        cache = self.make_cache([], enabled=False)
        self.assertEqual(cache.ttl_sweep(), 0)
        self.assertEqual(cache.maybe_ttl_sweep(), 0)
        cache._cache_ttl, cache.disable = self.policy, True
        self.assertEqual(cache.ttl_sweep(), 0)
        cache._evict.assert_not_called()

    def test_cadence_uses_real_sweep_and_no_early_run(self):
        leaf = Node("expired", self.root, tokens=4)
        self.policy.stamp_path(leaf, self.root, "agent", -20)
        cache = self.make_cache([leaf])
        self.assertEqual(cache.maybe_ttl_sweep(), 0)
        self.clock.now = 4.999
        self.assertEqual(cache.maybe_ttl_sweep(), 0)
        cache._evict.assert_not_called()
        self.clock.now = 5
        self.assertEqual(cache.maybe_ttl_sweep(), 4)
        self.assertEqual(self.policy.next_sweep, 10)
        self.assertEqual(cache.maybe_ttl_sweep(), 0)
        self.assertEqual(cache._evict.call_count, 1)

    def test_sweep_failure_disables_periodic_retry(self):
        cache = self.make_cache([])
        cache.ttl_sweep = Mock(side_effect=RuntimeError("synthetic eviction failure"))
        cache.maybe_ttl_sweep()
        self.clock.now = 5
        self.assertEqual(cache.maybe_ttl_sweep(), 0)
        self.assertFalse(self.policy.sweep_on)
        self.clock.now = 50
        self.assertEqual(cache.maybe_ttl_sweep(), 0)
        cache.ttl_sweep.assert_called_once()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.ns = policy_namespace()
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail
        self.error = HTTPException
        ns = extract("entrypoints/openai/serving_base.py", {"apply_cache_class"},
                     dict(self.ns, HTTPException=HTTPException), owner="OpenAIServingBase")
        self.apply = ns["apply_cache_class"]
        # Value objects stand in for msgspec only; actual policy merge code is exercised.
        hints = types.ModuleType("sglang.srt.managers.kv_hints")
        hints.KvHintAction = NS
        hints.KvHintsEnvelope = NS
        hints.decode_kv_hints_envelope = lambda value: NS(**value) if isinstance(value, dict) else value
        self.modules = patch.dict("sys.modules", {hints.__name__: hints})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def call(self, body=None, header=None, existing=None, enabled="1"):
        adapted = NS(rid="request-id", kv_hints=existing)
        raw = NS(headers={} if header is None else {"x-cache-class": header})
        with patch.dict(os.environ, {"SGLANG_CACHE_TTL": enabled}, clear=True):
            self.apply(None, adapted, NS(cache_class=body), raw)
        return adapted

    def test_header_over_body_and_existing_envelope_identity(self):
        other = NS(action_type="keep-me", payload={"id": 17})
        old = NS(action_type="cache_class", payload={"class": "user"})
        envelope = NS(protocol_version="3", message_id="original", actions=[other, old])
        adapted = self.call(body="invalid-but-overridden", header="  AgEnT ", existing=envelope)
        self.assertEqual(self.ns["cache_class_of_req"](adapted), "agent")
        self.assertEqual(adapted.kv_hints.protocol_version, "3")
        self.assertEqual(adapted.kv_hints.message_id, "original")
        self.assertIs(adapted.kv_hints.actions[0], other)
        self.assertEqual(len(adapted.kv_hints.actions), 2)
        self.assertEqual(envelope.actions, [other, old])

    def test_body_blank_header_fallback_absent_and_off(self):
        self.assertEqual(self.ns["cache_class_of_req"](self.call(body="user", header=" ")), "user")
        self.assertEqual(self.ns["cache_class_of_req"](self.call(body="agent")), "agent")
        self.assertIsNone(self.call().kv_hints)
        existing = object()
        self.assertIs(self.call(body="bad", header="bad", existing=existing, enabled="0").kv_hints, existing)

    def test_invalid_header_or_body_is_400(self):
        for body, header in (("user", "unknown"), ("unknown", None)):
            with self.subTest(body=body, header=header), self.assertRaises(self.error) as ctx:
                self.call(body=body, header=header)
            self.assertEqual(ctx.exception.status_code, 400)

    def test_last_hint_wins_and_invalid_last_clears_class(self):
        actions = [NS(action_type="cache_class", payload={"class": "agent"}),
                   NS(action_type="ignored", payload={}),
                   NS(action_type="cache_class", payload={"class": "user"})]
        self.assertEqual(self.ns["cache_class_from_kv_hints"](NS(actions=actions)), "user")
        actions.append(NS(action_type="cache_class", payload="malformed"))
        self.assertIsNone(self.ns["cache_class_from_kv_hints"](NS(actions=actions)))


class InitGuardTests(unittest.TestCase):
    def test_actual_init_policy_block_only_enables_python_tp1_pp1(self):
        path = SRT / "mem_cache/unified_radix_cache.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedRadixCache")
        init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Attribute) and t.attr == "_cache_ttl" for t in n.targets))
        block = init.body[start:start + 2]
        self.assertIsInstance(block[1], ast.If)
        code = compile(ast.Module(body=block, type_ignores=[]), str(path), "exec")
        class PythonTree:
            pass
        for enabled, disabled, python, tp, pp in (
            ("1", False, True, 1, 1), ("0", False, True, 1, 1),
            ("1", True, True, 1, 1), ("1", False, False, 1, 1),
            ("1", False, True, 2, 1), ("1", False, True, 1, 2),
        ):
            with self.subTest(enabled=enabled, disabled=disabled, python=python, tp=tp, pp=pp):
                ns = policy_namespace()
                core = PythonTree() if python else NS()
                original = ns["LRUStrategy"]()
                core.eviction_strategy = original
                cache = NS(disable=disabled, tree_core=core, tp_world_size=tp, pp_size=pp)
                ns.update(self=cache, UnifiedTreeCore=PythonTree, logger=Mock())
                with patch.dict(os.environ, {"SGLANG_CACHE_TTL": enabled}, clear=True):
                    exec(code, ns)
                expected = enabled == "1" and not disabled and python and tp == pp == 1
                self.assertEqual(cache._cache_ttl is not None, expected)
                if expected:
                    self.assertIsInstance(core.eviction_strategy, ns["TTLStrategy"])
                    self.assertIs(core.eviction_strategy.inner, original)
                else:
                    self.assertIs(core.eviction_strategy, original)


class HostAndSwaTests(unittest.TestCase):
    def host_pages(self, env, c128=None, size=0):
        ns = extract("mem_cache/hybrid_cache/hybrid_pool_assembler.py", {"_deepseek_v4_num_host_pages"},
                     {"get_memory": lambda: NS(hicache_size=size, hicache_ratio=4), "logger": logging.getLogger(__name__)})
        alloc = NS(size_full=131072)
        if c128 is not None:
            alloc.c128_attn_allocator = NS(num_pages=c128)
        with patch.dict(os.environ, env, clear=True):
            return ns["_deepseek_v4_num_host_pages"](params=NS(token_to_kv_pool_allocator=alloc),
                    kvcache=NS(size=131072, swa_size=9728), page_size=256, swa_page_size=256)

    def test_host_defaults_zero_and_token_page_rounding(self):
        self.assertEqual(self.host_pages({}), (2048, 152, 2048))
        self.assertEqual(self.host_pages({"KT_HICACHE_FULL_HOST_TOKENS": "0", "KT_HICACHE_SWA_HOST_TOKENS": "0"}), (2048, 152, 2048))
        self.assertEqual(self.host_pages({"KT_HICACHE_FULL_HOST_TOKENS": "7000000", "KT_HICACHE_SWA_HOST_TOKENS": "65536"}), (27344, 256, 27344))
        self.assertEqual(self.host_pages({"KT_HICACHE_FULL_HOST_TOKENS": "257", "KT_HICACHE_SWA_HOST_TOKENS": "1"}), (2, 1, 2))
        self.assertEqual(self.host_pages({"KT_HICACHE_FULL_HOST_TOKENS": "257"}, c128=3), (2, 152, 12))

    def test_invalid_host_tokens_and_unsupported_absolute_size(self):
        for name in ("KT_HICACHE_FULL_HOST_TOKENS", "KT_HICACHE_SWA_HOST_TOKENS"):
            for value in ("-1", "nan", "1.5"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.host_pages({name: value})
        with self.assertRaises(ValueError):
            self.host_pages({}, size=1)

    def swa(self, tail="0", write_back=False, window=128, buffer=False, has_host=True):
        path = "mem_cache/unified_cache/components/swa.py"
        with patch.dict(os.environ, {"KT_HICACHE_SWA_TAIL_ONLY": tail}, clear=True):
            flags = extract(path, set(), {"os": os}, assignments={"_KT_SWA_TAIL_ONLY"})
        methods = {"_collect_unbacked_swa_nodes", "_kt_child_holds_swa", "_collect_unbacked_swa_nodes_in_window"}
        ns = extract(path, methods, flags, owner="SWAComponent")
        component = type("ExtractedSwa", (), {name: ns[name] for name in methods})()
        root = Node("root", tokens=0)
        parent, leaf = Node("parent", root, 1792), None
        leaf = Node("leaf", parent, 256)
        component.component_type, component.sliding_window_size = "swa", window
        component.tree_core = NS(root_node=root, has_swa_host_pool=has_host,
                                 is_host_memory_buffer_only=buffer, is_write_back=write_back)
        return component, parent, leaf

    def test_tail_only_opt_in_write_back_and_window_coverage(self):
        c, p, leaf = self.swa()
        self.assertEqual(c._collect_unbacked_swa_nodes(p), [p])
        self.assertEqual(c._collect_unbacked_swa_nodes(leaf), [leaf])
        c, p, leaf = self.swa(tail="1")
        self.assertEqual(c._collect_unbacked_swa_nodes(p), [])
        self.assertEqual(c._collect_unbacked_swa_nodes(leaf), [leaf])
        c, p, leaf = self.swa(tail="1", window=300)
        self.assertEqual(c._collect_unbacked_swa_nodes(leaf), [leaf, p])
        c, p, leaf = self.swa(tail="1", write_back=True)
        self.assertEqual(c._collect_unbacked_swa_nodes(p), [p])

    def test_host_backed_child_pending_backup_and_missing_pool(self):
        c, p, leaf = self.swa(tail="1", window=300)
        leaf.component_data["swa"] = NS(value=None, host_value=list(range(256)))
        self.assertEqual(c._collect_unbacked_swa_nodes(p), [])
        self.assertEqual(c._collect_unbacked_swa_nodes(leaf), [p])
        p.write_through_pending_id = 1
        self.assertEqual(c._collect_unbacked_swa_nodes(leaf), [])
        c.tree_core.has_swa_host_pool = False
        self.assertEqual(c._collect_unbacked_swa_nodes(p), [])

    def test_buffer_mode_retains_single_node_backup(self):
        c, p, leaf = self.swa(tail="1", buffer=True, window=300)
        self.assertEqual(c._collect_unbacked_swa_nodes(p), [p])
        self.assertEqual(c._collect_unbacked_swa_nodes(leaf), [leaf])
        leaf.component_data["swa"].value = None
        self.assertEqual(c._collect_unbacked_swa_nodes(leaf), [])


class TraceTests(unittest.TestCase):
    def trace(self, enabled, minimum="2"):
        path = SRT / "mem_cache/kt_hicache_trace.py"
        ns = {}
        with patch.dict(os.environ, {"KT_HICACHE_TRACE": enabled, "KT_HICACHE_TRACE_MIN": minimum}, clear=True):
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        ns["_log"] = Mock()
        return ns

    def fixture(self):
        match_result, host_result, leaf_result = object(), object(), object()
        tree = NS(node_by_id=Mock(return_value=NS(backuped=False, key=[1, 2])),
                  evict_device_leaf=Mock(return_value=leaf_result))
        cache = NS(match_prefix=Mock(return_value=match_result),
                   _execute_kv_backup=Mock(return_value=host_result),
                   evict_host=Mock(return_value=7), load_back=Mock(return_value=True), tree_core=tree)
        return cache, (match_result, host_result, leaf_result)

    def test_trace_off_leaves_all_methods_untouched(self):
        ns = self.trace("0")
        cache, _ = self.fixture()
        before = vars(cache).copy()
        self.assertFalse(ns["install"](cache))
        self.assertEqual(vars(cache), before)
        ns["_log"].assert_not_called()

    def test_trace_off_ignores_malformed_trace_only_threshold(self):
        for value in ("bad", "", "-1", "1.5"):
            with self.subTest(value=value):
                ns = self.trace("0", value)
                cache, _ = self.fixture()
                before = vars(cache).copy()
                self.assertFalse(ns["install"](cache))
                self.assertEqual(vars(cache), before)

    def test_trace_enabled_rejects_invalid_threshold_and_allows_zero(self):
        for value in ("bad", "", "-1", "1.5"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                self.trace("1", value)
        self.assertEqual(self.trace("1", "0")["MIN_TOKENS"], 0)

    def test_trace_on_preserves_call_payload_return_objects_and_single_install(self):
        ns = self.trace("1")
        cache, results = self.fixture()
        originals = vars(cache).copy()
        leaf_method = cache.tree_core.evict_device_leaf
        self.assertTrue(ns["install"](cache))
        wrapped = cache.match_prefix
        self.assertFalse(ns["install"](cache))
        self.assertIs(cache.match_prefix, wrapped)
        params, extra, req = NS(key=[1, 2]), object(), object()
        self.assertIs(cache.match_prefix(params, extra, flag=True), results[0])
        originals["match_prefix"].assert_called_once_with(params, extra, flag=True)
        device, transfers, sidecars = [5, 6], {"swa": [NS(device_indices=[7])]}, object()
        self.assertIs(cache._execute_kv_backup(9, device, transfers, sidecars), results[1])
        originals["_execute_kv_backup"].assert_called_once_with(9, device, transfers, sidecars)
        self.assertEqual(cache.evict_host(8, "swa", marker=extra), 7)
        originals["evict_host"].assert_called_once_with(8, "swa", marker=extra)
        self.assertIs(cache.load_back(9, 12, req=req), True)
        originals["load_back"].assert_called_once_with(9, 12, req=req)
        self.assertIs(cache.tree_core.evict_device_leaf(9, False), results[2])
        leaf_method.assert_called_once_with(9, False)
        self.assertEqual(ns["STATS"]["match"], 1)
        self.assertEqual(ns["STATS"]["backup_tokens"], 2)
        self.assertEqual(ns["STATS"]["leaf_delete_tokens"], 2)

    def test_trace_preserves_original_exception(self):
        ns = self.trace("1")
        cache, _ = self.fixture()
        failure = RuntimeError("synthetic match failure")
        cache.match_prefix.side_effect = failure
        original = cache.match_prefix
        ns["install"](cache)
        with self.assertRaises(RuntimeError) as ctx:
            cache.match_prefix(NS(key=[1, 2]))
        self.assertIs(ctx.exception, failure)
        original.assert_called_once()
        self.assertEqual(ns["STATS"]["match"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sglang_root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "third_party/sglang")
    args, test_args = parser.parse_known_args()
    SRT = args.sglang_root.resolve() / "python/sglang/srt"
    unittest.main(argv=[sys.argv[0]] + test_args, verbosity=2)
