"""Default-off logical expert order for a fixed-placement candidate.

The input route IDs address the layer's global physical experts, including when
the payload was fetched through a resident or hot-shadow local row. Snapshot
the active physical-to-logical permutation during eager warm-up, then reuse its
device storage in graphs. No placement file or reference ORDER_MAP is read.
This candidate does not enable EPLB or dynamic hot placement.
"""
import os
import torch

_MODE = os.environ.get('KT_CONTRACT_V2_COMBINE_ORDER', '')
if _MODE not in ('', 'logical'):
    raise ValueError('KT_CONTRACT_V2_COMBINE_ORDER must be unset or logical; legacy ORDER_MAP is retired')
_cache = {}


def order_for(wrapper, device):
    if not _MODE:
        return None
    from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata
    metadata = get_global_expert_location_metadata()
    if metadata is None:
        raise RuntimeError('logical combine requires active expert-location metadata')
    layer = wrapper.kt_config.layer_idx
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('logical combine table requires the route CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    key = (id(metadata), device.index, layer)
    if key in _cache:
        return _cache[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('logical combine table must be initialized during eager warm-up before graph capture')
    if os.environ.get('KT_HOT_DYNAMIC', '0') != '0':
        raise RuntimeError('this logical-order candidate requires fixed placement during the run')
    rows = metadata.physical_to_logical_map_cpu
    if rows.device.type != 'cpu' or rows.ndim != 2 or tuple(rows.shape) != (40, 384):
        raise ValueError('logical combine requires a CPU [40, 384] expert map')
    if not 0 <= layer < 40:
        raise ValueError('logical combine layer is outside the pinned model')
    row = rows[layer].tolist()
    if sorted(row) != list(range(384)):
        raise ValueError('logical combine map must be a 384-expert permutation')
    table = torch.tensor(row, dtype=torch.int64, device=device)
    _cache[key] = table
    return table
