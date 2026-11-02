"""Frozen arithmetic tables and constants of the DeepSeek-V4.1-Flash-1017 contract v2 (mirror of
contract-v2-r1/contract_v2.py, shipped with the SGLang overlay so the GPU kernels need nothing else)."""
import hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MARGIN = 20

def _frozen(name, count, expected):
    data = (HERE / name).read_bytes()
    if len(data) != count * 4 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('shared arithmetic table has changed: ' + name)
    return np.frombuffer(data, dtype='<f4')

FP8 = _frozen('fp8_table.bin', 256, '422eccfaa21e72a6b26855bb10cdcfead6c1ce3262ecd813c99d8cbf9677f2e2')
SILU = _frozen('silu_table.bin', 65536, '4e08cff10bed7793df85e094f8ec9d9700cce52412bd5e54070eaf221596f9ea')
W_INT = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int64)   # FP4 * 2
_c = np.arange(256); _e = (_c & 127) >> 3; _m = _c & 7
N_OF_CODE = np.where(_e == 0, _m, (8 + _m) << np.maximum(_e - 1, 0)).astype(np.int64); N_OF_CODE[[127, 255]] = 0
assert np.all(np.abs(FP8[np.r_[0:127, 128:255]]) == (N_OF_CODE * 2.0 ** -9)[np.r_[0:127, 128:255]])
