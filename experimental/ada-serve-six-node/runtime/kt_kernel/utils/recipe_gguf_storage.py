"""Opt-in original MXFP4 storage source for the donor's unchanged CPU arithmetic.

No Native1017 execution profile or ModelRunner/state dependency. Reuses KT's
existing loader lifetime and worker-owned copies, not a second worker team.
Copyright 2026 Strata adaptation contributors. SPDX-License-Identifier: Apache-2.0
"""
import os
from pathlib import Path
import re


def create_mxfp4_loader(weight_path, ordinary_loader):
    path=Path(weight_path)
    manifest=os.environ.get("KT_MXFP4_GGUF_MANIFEST")
    pin=os.environ.get("KT_MXFP4_GGUF_MANIFEST_SHA256")
    if path.suffix.lower() != ".gguf":
        if manifest or pin:
            raise ValueError("GGUF manifest configured but kt_weight_path is not a GGUF file")
        return ordinary_loader(weight_path)
    if not path.is_file() or not manifest or not re.fullmatch("[0-9a-f]{64}",pin or ""):
        raise ValueError("MXFP4 GGUF requires a file, KT_MXFP4_GGUF_MANIFEST and pinned KT_MXFP4_GGUF_MANIFEST_SHA256")
    from .recipe_gguf_reader import NativeMXFP4GGUFReader
    return NativeMXFP4GGUFReader(path,manifest,pin)
