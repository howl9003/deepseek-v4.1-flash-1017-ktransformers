"""Default-off Phase G policy. This module has no GPU or torch import."""
import os

# Import-time like other feature selectors: never toggle an already-built pool/graph.
SINGLE_ROUTE = os.environ.get("SGLANG_SINGLE_ROUTE", "0") == "1"


def validate_single_route_pool(mode, compressed, backend, capability):
    """Validate the deliberately narrow V41 FP8 candidate, before allocation."""
    if tuple(capability) not in ((8, 9), (12, 0), (12, 1)):
        raise ValueError("SINGLE_ROUTE draft supports SM89 and SM120/121 only")
    if mode not in ("auto", "v4", "v41"):
        raise ValueError("SINGLE_ROUTE requires non-interleaved V41 FP8")
    if compressed not in ("auto", "fp8"):
        raise ValueError("SINGLE_ROUTE rejects FP4 compressed KV caches")
    if backend != "flashmla":
        raise ValueError("SINGLE_ROUTE requires the flashmla backend entry point")
    return "v41", "fp8"


def reader_layout(layout, page_size):
    """Only addressing/format; QK and softmax arithmetic are not selected here."""
    if page_size <= 0 or layout.value not in ("v4", "v41"):
        raise ValueError("portable reader supports non-interleaved V4/V41 FP8 only")
    data_stride, scale_base, scale_stride = layout.row_addressing(page_size)
    return data_stride, scale_base, scale_stride, layout.tile_size, layout.value == "v41"
