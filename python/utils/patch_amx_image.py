"""pathb-image-r1 (15 Nov, concept-r1 IMAGE_LOAD_DESIGN_r1.md section 4): the image hook for a SERVED kt-pkg.
pathb-image-r2 (16 Nov, STARTUP_CUTS_r1.md section 3): and the second hook, at the GGUF loader's creation in
NativeMoEWrapper.__init__ (IMAGE_INIT_HOOK after IMAGE_INIT_ANCHOR, build_bs.py's strings byte for byte).

The served kt-pkg is his installed kt_kernel package (with the installed host adapter: utils/amx.py, recipe_gguf_*) plus our
module and the two contract_v3 edits of path-b-r1/b1/patch_amx_v3.py; a stage swaps the .so only. The image hook lives in
amx.py, so a kt-pkg that is to load the image must take the same two things build_bs.py puts into the build tree's amx.py:
the hook at the top of NativeMoEWrapper.load_weights (IMAGE_HOOK, after IMAGE_ANCHOR) and python/utils/kt_image.py
appended. The strings are build_bs.py's IMAGE_ANCHOR and IMAGE_HOOK, byte for byte (checked by the imgid job and by
staged-imgid/test_imgid_local.py).

  python3 -B patch_amx_image.py <kt-pkg>/kt_kernel/utils/amx.py <kt_image.py> [<out amx.py>]

Refuses (exit 1, the file untouched) unless amx.py carries the contract_v3 selector (patch_amx_v3.py was applied), the
anchor once and no image hook yet. Prints the sha256 before and after. With <out>, writes there and leaves amx.py as is.
"""
import hashlib
import sys
from pathlib import Path

IMAGE_ANCHOR = '            return self._load_weights_mxfp4_cache(physical_to_logical_map_cpu)\n'
IMAGE_HOOK = (IMAGE_ANCHOR +
              '        # pathb-nodescale (concept-r1 NODE_SCALING_r1.md section 3): the block-ordered expert image\n'
              '        if self.method == "MXFP4" and os.environ.get("KT_IMAGE_DIR"):\n'
              '            if _kt_image_load_layer(self, physical_to_logical_map_cpu):\n'
              '                return\n')
# pathb-image-r2: the second hook, at the first wrapper's GGUF loader creation in NativeMoEWrapper.__init__
IMAGE_INIT_ANCHOR = ('        if NativeMoEWrapper._native_loader_instance is None:\n'
                     '            NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)\n'
                     '        self.loader = NativeMoEWrapper._native_loader_instance\n')
IMAGE_INIT_HOOK = ('        # pathb-image-r2 (concept-r1 STARTUP_CUTS_r1.md section 3): no GGUF loader while the image serves the layer\n'
                   '        # (load_weights makes one on a fallback); the layer\'s image task is queued here (KT_IMAGE_PRELOAD)\n'
                   '        if method == "MXFP4" and os.environ.get("KT_IMAGE_DIR") and _kt_image_init(self):\n'
                   '            self.loader = None\n'
                   '        else:\n'
                   '            if NativeMoEWrapper._native_loader_instance is None:\n'
                   '                NativeMoEWrapper._native_loader_instance = NativeMoEWrapper._create_loader(method, weight_path)\n'
                   '            self.loader = NativeMoEWrapper._native_loader_instance\n')
V3_SELECTOR = 'AVX2ContractV3_MOE = getattr(_moe_mod, "AVX2ContractV3_MOE", None)'


def patched(src, kt_image_py):
    """The patched text, or raise ValueError with the reason."""
    if V3_SELECTOR not in src or '    if forced == "contract_v3":\n' not in src:
        raise ValueError("amx.py has no contract_v3 selector (apply path-b-r1/b1/patch_amx_v3.py first)")
    if "_kt_image_load_layer" in src or "_kt_image_init" in src:
        raise ValueError("amx.py already has an image hook")
    if src.count(IMAGE_ANCHOR) != 1:
        raise ValueError(f"the hook's anchor occurs {src.count(IMAGE_ANCHOR)} times, not once")
    if src.count(IMAGE_INIT_ANCHOR) != 1:
        raise ValueError(f"the init hook's anchor occurs {src.count(IMAGE_INIT_ANCHOR)} times, not once")
    if not src.endswith("\n"):
        raise ValueError("amx.py does not end with a newline")
    return src.replace(IMAGE_ANCHOR, IMAGE_HOOK).replace(IMAGE_INIT_ANCHOR, IMAGE_INIT_HOOK) + kt_image_py


def main(argv):
    if len(argv) not in (2, 3):
        raise SystemExit(__doc__)
    amx, kti = Path(argv[0]), Path(argv[1])
    out = Path(argv[2]) if len(argv) == 3 else amx
    raw = amx.read_bytes()
    if b"\r" in raw:
        raise SystemExit("refused: amx.py has CR bytes")
    try:
        new = patched(raw.decode("utf-8"), kti.read_bytes().decode("utf-8"))
    except ValueError as ex:
        raise SystemExit(f"refused: {ex}")
    out.write_bytes(new.encode("utf-8"))
    print(f"patched {amx} -> {out}: sha256 {hashlib.sha256(raw).hexdigest()} -> "
          f"{hashlib.sha256(new.encode('utf-8')).hexdigest()} (kt_image.py {hashlib.sha256(kti.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main(sys.argv[1:])
