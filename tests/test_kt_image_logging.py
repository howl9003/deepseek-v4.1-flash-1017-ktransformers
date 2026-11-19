"""Interleave the native constructor's actual printf with the foreground image writer."""
import ast
import contextlib
import ctypes
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import unittest

KT = Path(__file__).resolve().parents[1]


def emit():
    # build_bs.py generates the native constructor. Require its complete record in
    # one printf, then exercise that same format with the platform's C runtime.
    tree = ast.parse((KT / "build_bs.py").read_bytes())
    patch = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "mt_b2" for t in n.targets))
    fmt = ast.literal_eval(re.search(r'printf\(("(?:\\.|[^"\\])*")', patch)[1]).encode("ascii")
    assert fmt.startswith(b"[kt-parts]") and fmt.endswith(b"\n")
    libc = ctypes.CDLL("msvcrt" if os.name == "nt" else None)
    native = sys.stdout

    class Interleave:
        called = False

        def write(self, text):
            size = native.write(text)
            native.flush()
            if not self.called:
                self.called = True
                libc.printf(fmt, 36, 2, b"320,192", 32)
                libc.fflush(None)
            return size

        def flush(self):
            native.flush()

    hook = runpy.run_path(str(KT / "python/utils/kt_image.py"), init_globals={"os": os})
    os.environ.pop("KT_IMAGE_STRICT", None)
    with contextlib.redirect_stdout(Interleave()):
        hook["_kt_image_fail"](35, "fixture")


class ImageLogging(unittest.TestCase):
    def test_interleaved_native_constructor(self):
        result = subprocess.check_output([sys.executable, "-I", "-B", "-u", __file__, "emit"], text=True)
        self.assertEqual(result.splitlines(keepends=True), [
            "[KT image] layer 35: GGUF path: fixture\n",
            "[kt-parts] layer 36: KT_TP_PART_BLOCKS: 2 parts of 320,192 rows (granule 32)\n"])


if __name__ == "__main__":
    if sys.argv[1:] == ["emit"]:
        emit()
    else:
        unittest.main()
