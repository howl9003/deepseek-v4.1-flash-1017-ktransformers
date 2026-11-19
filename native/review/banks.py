import ctypes
import sys
import torch

_lib = ctypes.CDLL(sys.argv[2])
P, I = ctypes.c_void_p, ctypes.c_int
_lib.v3_repack.argtypes = [P,P,P,P,I,I,I,P]
_lib.v3_repack.restype = I

def _p(t):
    return None if t is None else ctypes.c_void_p(t.data_ptr())


def _s():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _ck(rc, name):
    if rc != 0:
        raise RuntimeError(f"{name} failed rc={rc}")


def repack(W, WS):
    """Repack."""
    E, N, K2 = W.shape
    Wr = torch.empty_like(W); Sr = torch.empty_like(WS)
    _ck(_lib.v3_repack(_p(W.contiguous()), _p(WS.contiguous()), _p(Wr), _p(Sr), E, N, K2 * 2, _s()), "repack")
    return Wr, Sr
