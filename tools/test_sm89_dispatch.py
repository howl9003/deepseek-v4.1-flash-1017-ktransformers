"""CPU-only contract tests for the applied SM89 patch; no torch installation needed.

python3 tools/test_sm89_dispatch.py /path/to/patched/sglang
"""

import ast
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = (
    pathlib.Path(sys.argv.pop(1))
    if len(sys.argv) > 1
    else pathlib.Path("third_party/sglang")
)
BACKEND = ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
MODULE = ROOT / "python/sglang/kernels/ops/attention/sm89_dual_scope.py"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in found} == set(names)
    for node in found:
        node.decorator_list = []
    module = ast.Module(body=found, type_ignores=[])
    code = compile(
        ast.fix_missing_locations(module),
        str(path),
        "exec",
        flags=__import__("__future__").annotations.compiler_flag,
    )
    exec(code, namespace)
    return namespace


class Tensor:
    def __init__(self, rows, width=584):
        self.rows = list(rows)
        self.shape = (len(self.rows), 1, 1, width)
        self.ndim = 4

    def __getitem__(self, sl):
        return Tensor(self.rows[sl], self.shape[-1])

    def element_size(self):
        return 1


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.platform = types.SimpleNamespace(device_capability=(8, 9), is_sm120=False)
        self.ns = load_functions(
            BACKEND,
            [
                "_use_sm89_sparse_attention",
                "_create_flashmla_metadata",
                "_use_sparse_prefill",
            ],
            {
                "_is_cuda": True,
                "_is_xpu": False,
                "get_platform": lambda: self.platform,
                "is_cp_active": lambda _: False,
                "_LARGE_INDEXER_QUERY_THRESHOLD": 32,
                "envs": types.SimpleNamespace(
                    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=types.SimpleNamespace(
                        get=lambda: False
                    )
                ),
            },
        )

    def test_exact_device_gate(self):
        for cap in [
            (7, 0),
            (8, 0),
            (8, 6),
            (8, 9),
            (9, 0),
            (10, 0),
            (10, 3),
            (12, 0),
            (12, 1),
        ]:
            with self.subTest(cap=cap):
                self.platform.device_capability = cap
                self.assertEqual(self.ns["_use_sm89_sparse_attention"](), cap == (8, 9))
        self.ns["_is_cuda"] = False
        self.platform.device_capability = (8, 9)
        self.assertFalse(self.ns["_use_sm89_sparse_attention"]())

    def test_metadata_avoids_flashmla_only_for_portable_devices(self):
        native = types.ModuleType("sgl_kernel.flash_mla")
        native.get_mla_metadata = lambda: ("native-metadata",)
        parent = types.ModuleType("sgl_kernel")
        parent.flash_mla = native
        # Newer pinned SGLang exposes the same metadata factory here instead.
        interface = types.ModuleType(
            "sglang.kernels.ops.attention.flash_mla_interface"
        )
        interface.get_mla_metadata = native.get_mla_metadata
        packages = {}
        for name in (
            "sglang",
            "sglang.kernels",
            "sglang.kernels.ops",
            "sglang.kernels.ops.attention",
        ):
            packages[name] = types.ModuleType(name)
            packages[name].__path__ = []
        with patch.dict(
            sys.modules,
            {
                "sgl_kernel": parent,
                "sgl_kernel.flash_mla": native,
                **packages,
                "sglang.kernels.ops.attention.flash_mla_interface": interface,
            },
        ):
            for cap in [(8, 9), (9, 0), (10, 0), (12, 0)]:
                self.platform.device_capability = cap
                self.platform.is_sm120 = cap == (12, 0)
                expected = None if cap in [(8, 9), (12, 0)] else "native-metadata"
                self.assertEqual(self.ns["_create_flashmla_metadata"](), expected)

    def test_sparse_prefill_native_paths_preserved(self):
        obj = types.SimpleNamespace(
            trtllm_attn=False,
            forward_metadata=types.SimpleNamespace(late_layer_tail=None),
        )
        batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_extend_without_speculative=lambda: True
            )
        )
        for cap in [(8, 9), (9, 0), (10, 0), (12, 0)]:
            self.platform.device_capability = cap
            self.platform.is_sm120 = cap == (12, 0)
            self.assertEqual(
                self.ns["_use_sparse_prefill"](obj, batch, num_qo_tokens=128),
                cap in [(9, 0), (10, 0)],
            )


class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.cache = Tensor(range(4))

        def invoke(**kw):
            self.calls.append(kw)
            return kw["q"]

        self.ns = load_functions(
            MODULE,
            ["flash_mla_sparse_sm89"],
            {
                "_V4_BYTES_PER_TOKEN": 584,
                # the selectors off (SGLANG_SM89_V41_KV, SGLANG_SM89_FP4_COMPRESSED_KV)
                "_MAIN_WIDTHS": (584,),
                "_EXTRA_WIDTHS": (584,),
                "_MAX_ROWS": 3,
                "_DUAL_SCOPE": True,
                "supported": lambda *args: True,
                "sm89_dual_scope_decode": invoke,
                "torch": types.SimpleNamespace(
                    cat=lambda chunks, dim: Tensor(
                        [v for c in chunks for v in c.rows], 512
                    )
                ),
            },
        )
        self.args = dict(
            q=Tensor(range(8), 512),
            k_cache=self.cache,
            indices=Tensor(range(8)),
            topk_length=Tensor(range(8)),
            attn_sink=object(),
            head_dim_v=512,
            softmax_scale=0.125,
            extra_k_cache=self.cache,
            extra_indices=Tensor(range(8)),
            extra_topk_length=Tensor(range(8)),
        )

    def test_chunked_query_metadata_and_cache_identity(self):
        out = self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertEqual(out.rows, list(range(8)))
        self.assertEqual(
            [c["q"].rows for c in self.calls], [[0, 1, 2], [3, 4, 5], [6, 7]]
        )
        for call in self.calls:
            for key in ["indices", "topk_length", "extra_indices", "extra_topk_length"]:
                self.assertEqual(call[key].rows, call["q"].rows)
            self.assertIs(call["k_cache"], self.cache)
            self.assertIs(call["extra_k_cache"], self.cache)
            self.assertIs(call["attn_sink"], self.args["attn_sink"])

    def test_swa_only_and_absent_lengths(self):
        for key in [
            "topk_length",
            "extra_k_cache",
            "extra_indices",
            "extra_topk_length",
        ]:
            self.args[key] = None
        self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertTrue(
            all(
                c["extra_indices"] is None and c["topk_length"] is None
                for c in self.calls
            )
        )

    def test_unrecognised_layout_fails_before_kernel(self):
        for width in [528, 288]:
            self.args["k_cache"] = Tensor(range(4), width)
            with self.assertRaisesRegex(ValueError, "V4 584-byte"):
                self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertEqual(self.calls, [])

    def test_selector_layouts_reach_the_dual_scope_kernels(self):
        # SGLANG_SM89_V41_KV (it implies SGLANG_SM89_FP4_COMPRESSED_KV): V41 main, V41_FP4 extra
        self.ns.update(_MAIN_WIDTHS=(584, 528), _EXTRA_WIDTHS=(584, 288))
        main, extra = Tensor(range(4), 528), Tensor(range(4), 288)
        self.args.update(k_cache=main, extra_k_cache=extra)
        self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertEqual(len(self.calls), 3)
        for call in self.calls:
            self.assertIs(call["k_cache"], main)
            self.assertIs(call["extra_k_cache"], extra)

    def test_selector_layouts_refuse_the_portable_path(self):
        self.ns.update(
            _MAIN_WIDTHS=(584, 528), _EXTRA_WIDTHS=(584, 288), supported=lambda *a: False
        )
        for main, extra in [(528, 288), (584, 288), (528, 584)]:
            self.args.update(
                k_cache=Tensor(range(4), main), extra_k_cache=Tensor(range(4), extra)
            )
            with self.assertRaisesRegex(ValueError, "portable SM89 path"):
                self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertEqual(self.calls, [])

    def test_fp4_cache_needs_its_selector_and_stays_the_extra_cache(self):
        self.args["extra_k_cache"] = Tensor(range(4), 288)
        with self.assertRaisesRegex(ValueError, "V4 584-byte"):
            self.ns["flash_mla_sparse_sm89"](**self.args)
        self.ns.update(_EXTRA_WIDTHS=(584, 288))
        self.args.update(k_cache=Tensor(range(4), 288), extra_k_cache=self.cache)
        with self.assertRaisesRegex(ValueError, "V4 584-byte"):
            self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertEqual(self.calls, [])

    def test_query_shape_fallback(self):
        self.ns["supported"] = lambda *args: False
        fallback = types.ModuleType(
            "sglang.kernels.ops.attention.flash_mla_sm120_triton"
        )
        fallback.flash_mla_sparse_decode_triton = lambda **kw: (
            self.ns["sm89_dual_scope_decode"](**kw),
            None,
        )
        with patch.dict(sys.modules, {fallback.__name__: fallback}):
            out = self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertEqual(out.rows, list(range(8)))

    def test_multisequence_query_rejected_before_fallback(self):
        self.args["q"].shape = (2, 4, 64, 512)
        with self.assertRaisesRegex(ValueError, "expects q"):
            self.ns["flash_mla_sparse_sm89"](**self.args)
        self.assertEqual(self.calls, [])

    def test_partials_are_owned_by_each_call(self):
        allocations = []

        def allocate(*args, **kwargs):
            result = object()
            allocations.append((args, kwargs, result))
            return result

        fake = types.SimpleNamespace(float32="float32", empty=allocate)
        ns = load_functions(MODULE, ["_partials"], {"torch": fake, "_D": 512})
        first = ns["_partials"](8, 1, 64, "cuda:0")
        second = ns["_partials"](8, 1, 64, "cuda:0")
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        self.assertEqual(allocations[0][0], (8, 1, 64, 512))
        self.assertEqual(allocations[1][0], (8, 1, 64))


LAYOUT = ROOT / "python/sglang/kernels/ops/attention/dsv4/kv_layout.py"
POOL = ROOT / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
SELECTORS = ("SGLANG_SM89_V41_KV", "SGLANG_SM89_FP4_COMPRESSED_KV")


def load_kv_layout():
    """kv_layout.py imports only the standard library."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("sm89_test_kv_layout", LAYOUT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class KVLayoutSelectorTest(unittest.TestCase):
    """Tier 0's sm89-fp4-compressed-kv and sm89-v41-window-kv: which layouts a pool gets."""

    def setUp(self):
        self.kl = load_kv_layout()
        self.cap, self.mode, self.option = (8, 9), "auto", "auto"
        get = lambda f: types.SimpleNamespace(get=f)  # noqa: E731
        self.ns = load_functions(
            POOL,
            ["resolve_compressed_kv_layout", "select_dsv4_kv_layout"],
            {
                "KVLayout": self.kl.KVLayout,
                "sm89_fp4_compressed_kv": self.kl.sm89_fp4_compressed_kv,
                "sm89_v41_kv": self.kl.sm89_v41_kv,
                "_is_hip": False,
                "torch": types.SimpleNamespace(
                    cuda=types.SimpleNamespace(
                        is_available=lambda: True,
                        get_device_capability=lambda *a: self.cap,
                    ),
                    version=types.SimpleNamespace(cuda="13.0"),
                ),
                "envs": types.SimpleNamespace(
                    SGLANG_DSV4_KV_LAYOUT=get(lambda: self.mode),
                    SGLANG_DSV4_COMPRESSED_KV_LAYOUT=get(lambda: self.option),
                    SGLANG_SM120_FLASHMLA_BACKEND=get(lambda: "flashinfer"),
                ),
                "get_exec": lambda: types.SimpleNamespace(
                    kernel=types.SimpleNamespace(dsv4_attn_backend="auto")
                ),
            },
        )
        self.env = patch.dict(os.environ, {})
        self.env.start()
        for name in SELECTORS:
            os.environ.pop(name, None)

    def tearDown(self):
        self.env.stop()

    def layouts(self):
        main, option = self.ns["select_dsv4_kv_layout"]()
        resolve = self.ns["resolve_compressed_kv_layout"]
        return main, {r: resolve(main, r, option) for r in (1, 2, 4, 128)}

    def test_selectors_off_keep_v4_everywhere(self):
        V4 = self.kl.KVLayout.V4
        self.assertEqual(self.layouts(), (V4, dict.fromkeys((1, 2, 4, 128), V4)))
        self.assertFalse(self.kl.is_valid_kv_layout_pair(V4, self.kl.KVLayout.V41_FP4))

    def test_fp4_compressed_kv(self):
        K = self.kl.KVLayout
        os.environ["SGLANG_SM89_FP4_COMPRESSED_KV"] = "1"
        self.assertEqual(
            self.layouts(), (K.V4, {1: K.V41_FP4, 2: K.V41_FP4, 4: K.V4, 128: K.V4})
        )
        self.assertTrue(self.kl.is_valid_kv_layout_pair(K.V4, K.V41_FP4))
        self.assertFalse(self.kl.is_valid_kv_layout_pair(K.V4, K.V41))
        # an explicit fp8 compressed layout next to the V4 main cache conflicts
        self.mode, self.option = "v4", "fp8"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.layouts()
        self.mode, self.option = "auto", "auto"
        # GPUs that keep the V4 main cache refuse it; SM100 already has V41_FP4 there
        for cap in [(9, 0), (12, 0), (7, 5)]:
            self.cap = cap
            with self.assertRaisesRegex(ValueError, "SM89"):
                self.layouts()
        self.cap = (10, 0)
        self.assertEqual(self.layouts()[1][1], K.V41_FP4)

    def test_v41_window_kv_implies_fp4(self):
        K = self.kl.KVLayout
        os.environ["SGLANG_SM89_V41_KV"] = "1"
        self.assertTrue(self.kl.sm89_fp4_compressed_kv())
        self.assertEqual(
            self.layouts(), (K.V41, {1: K.V41_FP4, 2: K.V41_FP4, 4: K.V41, 128: K.V41})
        )
        for mode, option in [("v4", "auto"), ("v41_new", "auto"), ("auto", "fp8")]:
            self.mode, self.option = mode, option
            with self.assertRaisesRegex(ValueError, "conflict"):
                self.layouts()
        self.mode, self.option = "v41", "fp4"
        self.assertEqual(self.layouts()[0], K.V41)
        for cap in [(9, 0), (10, 0), (12, 0), (7, 5)]:
            self.cap = cap
            with self.assertRaisesRegex(ValueError, "SM89"):
                self.layouts()


class FakeTensor:
    def __init__(self, *shape):
        self.shape, self.ndim, self.device = tuple(shape), len(shape), "cuda:0"

    def stride(self, dim=None):
        return tuple(1 for _ in self.shape) if dim is None else 1

    def reshape(self, rows, rest):
        return FakeTensor(rows, int(__import__("math").prod(self.shape)) // rows)

    def view(self, dtype):
        return self

    def contiguous(self):
        return self

    def unsqueeze(self, dim):
        return self

    def __getitem__(self, index):
        return self


class Kernel:
    """A Triton kernel stand-in: `kernel[grid](*args, **kwargs)` is recorded."""

    def __init__(self, name, launches):
        self.name, self.launches = name, launches
        self.fn = self

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((self.name, grid, kwargs))


class SplitKFlagsTest(unittest.TestCase):
    """The V41 / V41_FP4 flags reach the kernel SGLANG_SM89_ATTN_SPLITK=auto picks: split 8
    up to four rows, split 4 up to eight, unsplit above."""

    def setUp(self):
        self.launches = []
        names = [
            "_fused_gather_attn_dsv4_dual_scope_kernel",
            "_fused_gather_attn_dsv4_dual_scope_splitk_kernel",
            "_combine_splitk_kernel_8_optimized",
            "_combine_splitk_kernel",
            "_combine_splitk_kernel_2",
        ]
        F = types.SimpleNamespace(
            _bucket_total_tokens=lambda rows: rows,
            **{n: Kernel(n, self.launches) for n in names},
        )
        self.ns = load_functions(
            MODULE,
            ["sm89_dual_scope_decode", "_split_k", "_partials", "_launch"],
            {
                "torch": types.SimpleNamespace(
                    empty=lambda *shape, **kw: FakeTensor(*shape),
                    empty_like=lambda t: t,
                    uint8="uint8",
                    float32="float32",
                    bfloat16="bfloat16",
                ),
                "triton": types.SimpleNamespace(cdiv=lambda a, b: -(-a // b)),
                "_F": F,
                "_SPLITK_MODE": "auto",
                "_AUTOTUNE": False,
                "_BLOCK_H": 16,
                "_BLOCK_N": 64,
                "_NUM_WARPS": 8,
                "_D": 512,
                "_FP4_BYTES_PER_TOKEN": 288,
                "_V41_BYTES_PER_TOKEN": 528,
                "_STAGED_ROWS": 0,  # optional 02-sm89-staged-gather, off
                "_should_use_fused_splitk": lambda *a: False,
            },
        )

    def attention_launch(self, rows, main_width, extra_width):
        self.launches.clear()
        extra = None if extra_width is None else FakeTensor(512, 64, 1, extra_width)
        self.ns["sm89_dual_scope_decode"](
            FakeTensor(rows, 64, 512),
            FakeTensor(1024, 64, 1, main_width),
            FakeTensor(rows, 1, 128),
            None,
            None,
            512,
            None,
            extra,
            None if extra is None else FakeTensor(rows, 1, 512),
            None,
        )
        attention = [l for l in self.launches if "gather_attn" in l[0]]
        self.assertEqual(len(attention), 1)
        return attention[0]

    def test_flags_follow_the_cache_widths_in_every_split(self):
        for rows, kernel in [
            (1, "_splitk_kernel"),
            (4, "_splitk_kernel"),
            (6, "_splitk_kernel"),
            (16, "_dual_scope_kernel"),
        ]:
            for main, extra, flags in [
                (528, 288, (True, True)),
                (584, 288, (False, True)),
                (584, 584, (False, False)),
                (528, None, (True, False)),
            ]:
                with self.subTest(rows=rows, main=main, extra=extra):
                    name, grid, kw = self.attention_launch(rows, main, extra)
                    self.assertTrue(name.endswith(kernel), name)
                    self.assertEqual((kw["MAIN_V41"], kw["EXTRA_FP4"]), flags)


class LauncherDefaultsTest(unittest.TestCase):
    def test_ada_defaults_preserve_native_devices_and_user_overrides(self):
        launcher = pathlib.Path(__file__).resolve().parents[1] / "serve.sh"
        blocks = re.findall(r'if \[\[ "\$cap" == "8\.9" \]\]; then\n.*?\nfi',
                            launcher.read_text(encoding="utf-8"), re.DOTALL)
        self.assertEqual(len(blocks), 1)
        names = ["SGLANG_OPT_USE_TOPK_V2", "SGLANG_OPT_USE_TILELANG_MHC_PRE",
                 "SGLANG_OPT_DEEPGEMM_HC_PRENORM", "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH",
                 "SGLANG_DSV41_HOST_KV", "SGLANG_SM89_V41_KV"]
        script = blocks[0] + "\n" + "\n".join(
            'printf "%s\\n" "${' + name + '-unset}"' for name in names)
        clean = {k: v for k, v in os.environ.items() if k not in names}
        for cap in ["8.9", "9.0", "10.0", "12.0"]:
            for override in [None, "0", "1"]:
                with self.subTest(cap=cap, override=override):
                    env = dict(clean, cap=cap)
                    if override is not None:
                        env.update(dict.fromkeys(names, override))
                    got = subprocess.check_output([shutil.which("bash") or "bash", "-c", script], env=env,
                                                  text=True).splitlines()
                    expected = ([override] * 6 if override is not None else
                                ["0", "0", "0", "1", "unset", "1"] if cap == "8.9" else ["unset"] * 6)
                    self.assertEqual(got, expected)

    def test_fp32_logits_on_sm89_by_default(self):
        launcher = pathlib.Path(__file__).resolve().parents[1] / "serve.sh"
        lines = [l for l in launcher.read_text(encoding="utf-8").splitlines()
                 if "fp32_head" in l and not l.lstrip().startswith("#") and "printf" not in l]
        self.assertEqual(len(lines), 4)
        script = "extra_args=()\n" + "\n".join(lines) + '\nprintf "%s\\n" "${extra_args[*]}"'
        clean = {k: v for k, v in os.environ.items() if k != "SGLANG_FP32_LM_HEAD"}
        for cap in ["8.9", "9.0", "10.0", "12.0"]:
            for override in [None, "0", "1"]:
                with self.subTest(cap=cap, override=override):
                    env = dict(clean, cap=cap)
                    if override is not None:
                        env["SGLANG_FP32_LM_HEAD"] = override
                    got = subprocess.check_output([shutil.which("bash") or "bash", "-c", script], env=env,
                                                  text=True).strip()
                    on = override == "1" or (override is None and cap == "8.9")
                    self.assertEqual(got, "--enable-fp32-lm-head" if on else "")


class HostKVDefaultsTest(unittest.TestCase):
    """Exercise the upstream layout-aware host-KV contract used by Ada.

    This checks default selection and staging geometry, not GPU gather execution.
    The class is absent from tier 0, so those configurations skip these checks.
    """

    def setUp(self):
        self.source = ROOT / "python/sglang/srt/mem_cache/dsv41_host_kv.py"
        if not self.source.exists():
            self.skipTest("host-kv tier-1 feature is not applied")

    def test_upstream_host_kv_default_and_explicit_opt_out(self):
        ns = load_functions(self.source, ["host_kv_enabled"], {"os": os})
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(ns["host_kv_enabled"]())
            os.environ["SGLANG_DSV41_HOST_KV"] = "0"
            self.assertFalse(ns["host_kv_enabled"]())
            os.environ["SGLANG_DSV41_HOST_KV"] = "1"
            self.assertTrue(ns["host_kv_enabled"]())

    def test_host_staging_uses_each_pools_padded_geometry(self):
        fake_torch = types.SimpleNamespace(
            uint8="uint8",
            int32="int32",
            zeros=lambda *shape, **kwargs: (shape, kwargs),
            full=lambda shape, fill, **kwargs: (shape, fill, kwargs),
        )
        ns = load_functions(self.source, ["__init__"], {"torch": fake_torch, "os": os})
        # V4 uses 576 data + 8 scale bytes, not V41_FP4's 256 + 32.
        # Include page padding explicitly; staging must use the supplied stride.
        layouts = {1: (576, 8, 128, 74752), 2: (576, 8, 8, 5120),
                   4: (256, 32, 128, 36864)}
        obj = types.SimpleNamespace()
        ns["__init__"](obj, layouts=layouts, topk=512, max_rows=16, device="cpu")
        for ratio, (data, scale, entries, page_bytes) in layouts.items():
            with self.subTest(ratio=ratio):
                self.assertEqual(obj.geometry[ratio], (data, scale, entries, page_bytes))
                self.assertEqual(obj.entries_per_page[ratio], entries)
                self.assertEqual(obj.stage[ratio][0], (16 * 512 // entries, page_bytes))
                self.assertEqual(obj.stage[ratio][1], {"dtype": "uint8", "device": "cpu"})
                self.assertEqual(obj.remap[ratio][0:2], ((16, 512), -1))


if __name__ == "__main__":
    unittest.main()
