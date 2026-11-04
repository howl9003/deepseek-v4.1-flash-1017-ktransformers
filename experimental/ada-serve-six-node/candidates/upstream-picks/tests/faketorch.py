"""A numpy stand-in for the slice of torch the upstream-picks-r1 overlays touch, so the patched files run on a CPU
without torch. Tensors hold numpy arrays (BF16 as float32); `cuda` tensors are ordinary arrays tagged with a device.
Streams run in order; events are recorded and queried through a switch the tests control. Not a general torch."""
import itertools
import types

import numpy as np


class dtype:
    def __init__(self, name, np_dtype, itemsize):
        self.name, self.np, self.itemsize = name, np.dtype(np_dtype), itemsize

    def __repr__(self):
        return 'torch.' + self.name


bfloat16 = dtype('bfloat16', np.float32, 2)
float32 = dtype('float32', np.float32, 4)
float64 = dtype('float64', np.float64, 8)
int32 = dtype('int32', np.int32, 4)
int64 = dtype('int64', np.int64, 8)
uint8 = dtype('uint8', np.uint8, 1)
int8 = dtype('int8', np.int8, 1)
bool = dtype('bool', np.bool_, 1)   # noqa: A001 (torch.bool)
long = int64
float8_e4m3fn = dtype('float8_e4m3fn', np.uint8, 1)
float8_e8m0fnu = dtype('float8_e8m0fnu', np.uint8, 1)
_BY_NP = {np.dtype(np.float32): float32, np.dtype(np.float64): float64, np.dtype(np.int32): int32,
          np.dtype(np.int64): int64, np.dtype(np.uint8): uint8, np.dtype(np.int8): int8, np.dtype(np.bool_): bool}


class device:
    def __init__(self, spec='cpu', index=None):
        if isinstance(spec, device):
            spec, index = spec.type, spec.index
        if isinstance(spec, str) and ':' in spec:
            spec, index = spec.split(':')[0], int(spec.split(':')[1])
        self.type, self.index = spec, index if index is not None else (0 if spec == 'cuda' else None)

    def __eq__(self, other):
        other = device(other) if isinstance(other, str) else other
        return isinstance(other, device) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self):
        return hash((self.type, self.index))

    def __repr__(self):
        return f'device({self.type}:{self.index})'


CUDA = device('cuda', 0)
_PTRS = {}            # data_ptr -> Tensor, every tensor ever made (views included)
_BASES = {}           # data_ptr -> Tensor made by empty/zeros/full (what a native pointer stands for)
ALLOCS = []           # (shape, dtype, device, pinned) of every allocation, for buffer-growth checks


def _dev(d):
    if d is None:
        return device('cpu')
    return d if isinstance(d, device) else device(d)


class Tensor:
    def __init__(self, arr, dt=None, dev=None, pinned=False):
        self.a = arr
        self._dtype = dt or _BY_NP[arr.dtype]
        self.device = _dev(dev)
        self.pinned = pinned
        _PTRS[self.data_ptr()] = self

    # -- metadata
    @property
    def shape(self):
        return tuple(self.a.shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return self.a.ndim

    @property
    def is_cuda(self):
        return self.device.type == 'cuda'

    def dim(self):
        return self.a.ndim

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]

    def numel(self):
        return int(self.a.size)

    def element_size(self):
        return self._dtype.itemsize

    def data_ptr(self):
        return self.a.__array_interface__['data'][0]

    def is_contiguous(self):
        return self.a.flags['C_CONTIGUOUS']

    def __len__(self):
        return self.a.shape[0]

    def __repr__(self):
        return f'FakeTensor({self.shape}, {self._dtype}, {self.device})'

    # -- views and copies
    def _wrap(self, arr, dt=None):
        return Tensor(arr, dt or (self._dtype if arr.dtype == self.a.dtype else None), self.device, self.pinned)

    def __getitem__(self, idx):
        idx = idx.a if isinstance(idx, Tensor) else idx
        if isinstance(idx, tuple):
            idx = tuple(i.a if isinstance(i, Tensor) else i for i in idx)
        return self._wrap(self.a[idx])

    def __setitem__(self, idx, value):
        self.a[idx] = value.a if isinstance(value, Tensor) else value

    def contiguous(self):
        return self if self.is_contiguous() else self._wrap(np.ascontiguousarray(self.a))

    def clone(self):
        return self._wrap(self.a.copy())

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return self._wrap(self.a.reshape(shape))

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], dtype):
            return self._wrap(self.a.view(shape[0].np), shape[0])
        return self.reshape(*shape)

    def expand(self, *shape):
        return self._wrap(np.broadcast_to(self.a, shape))

    def to(self, dt=None, **kw):
        if isinstance(dt, dtype):
            return Tensor(self.a.astype(dt.np), dt, self.device, self.pinned)
        if dt is not None:
            return Tensor(self.a.copy(), self._dtype, _dev(dt))
        return self

    def float(self):
        return self.to(float32)

    def long(self):
        return self.to(int64)

    def copy_(self, src, non_blocking=False):
        np.copyto(self.a, src.a.astype(self.a.dtype, copy=False) if isinstance(src, Tensor) else src)
        return self

    def zero_(self):
        self.a[...] = 0
        return self

    def fill_(self, v):
        self.a[...] = v
        return self

    def new_full(self, shape, v):
        return Tensor(np.full(shape, v, dtype=self.a.dtype), self._dtype, self.device)

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, dtype=self.a.dtype), self._dtype, self.device)

    def all(self):
        return Tensor(np.array(bool_(self.a.all())))

    def any(self):
        return Tensor(np.array(bool_(self.a.any())))

    def item(self):
        return self.a.item()

    def __int__(self):
        return int(self.a)

    __index__ = __int__

    def __bool__(self):
        return builtins_bool(self.a)

    # -- arithmetic and comparisons
    def _bin(self, other, op):
        o = other.a if isinstance(other, Tensor) else other
        r = op(self.a, o)
        return Tensor(np.asarray(r), None, self.device)

    def __ge__(self, o): return self._bin(o, np.greater_equal)
    def __gt__(self, o): return self._bin(o, np.greater)
    def __le__(self, o): return self._bin(o, np.less_equal)
    def __lt__(self, o): return self._bin(o, np.less)
    def __eq__(self, o): return self._bin(o, np.equal)          # noqa: E704
    def __ne__(self, o): return self._bin(o, np.not_equal)      # noqa: E704
    def __and__(self, o): return self._bin(o, np.logical_and)
    def __or__(self, o): return self._bin(o, np.logical_or)
    def __add__(self, o): return self._bin(o, np.add)
    def __sub__(self, o): return self._bin(o, np.subtract)
    def __mul__(self, o): return self._bin(o, np.multiply)
    __hash__ = object.__hash__


builtins_bool = __builtins__['bool'] if isinstance(__builtins__, dict) else __builtins__.bool
bool_ = np.bool_


def _arr(v):
    return v.a if isinstance(v, Tensor) else v


def _new(arr, dt, dev, pinned=False):
    t = Tensor(arr, dt, dev, pinned)
    _BASES[t.data_ptr()] = t
    ALLOCS.append((tuple(arr.shape), dt, _dev(dev), pinned))
    return t


def empty(*shape, dtype=float32, device=None, pin_memory=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    # garbage, not zeros: a buffer read before it is written shows up in the checks
    arr = np.full(shape, np.nan if dtype.np.kind == 'f' else 77, dtype=dtype.np)
    return _new(arr, dtype, device, pin_memory)


def zeros(*shape, dtype=float32, device=None, pin_memory=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return _new(np.zeros(shape, dtype=dtype.np), dtype, device, pin_memory)


def full(shape, v, dtype=float32, device=None):
    return _new(np.full(shape, v, dtype=dtype.np), dtype, device)


def zeros_like(t, dtype=None):
    dt = dtype or t.dtype
    return Tensor(np.zeros(t.shape, dtype=dt.np), dt, t.device)


def full_like(t, v):
    return Tensor(np.full(t.shape, v, dtype=t.a.dtype), t.dtype, t.device)


def arange(n, dtype=int64, device=None):
    return Tensor(np.arange(n, dtype=dtype.np), dtype, device)


def tensor(data, dtype=None, device=None):
    arr = np.array(data, dtype=dtype.np if dtype else None)
    return Tensor(arr, dtype, device)


def cat(ts, dim=0):
    return Tensor(np.concatenate([t.a for t in ts], axis=dim), ts[0].dtype, ts[0].device)


def where(cond, a, b, out=None):
    ca, aa, ba = _arr(cond), _arr(a), _arr(b)
    ref = a if isinstance(a, Tensor) else b
    r = np.where(ca, aa, ba).astype(ref.a.dtype if isinstance(ref, Tensor) else np.result_type(aa, ba), copy=False)
    if out is not None:
        np.copyto(out.a, r)
        return out
    return Tensor(r, ref.dtype if isinstance(ref, Tensor) else None, ref.device if isinstance(ref, Tensor) else None)


def isfinite(t):
    return Tensor(np.isfinite(t.a), bool, t.device)


def _assert_async(cond, msg=''):
    assert builtins_bool(_arr(cond)), msg


# -- cuda: streams run in order on the host
class Event:
    created = 0

    def __init__(self, *a, **kw):
        Event.created += 1
        self.done = False
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True
        self.done = cuda.events_complete

    def query(self):
        return self.done or cuda.events_complete

    def synchronize(self):
        self.done = True


class Stream:
    _ids = itertools.count(1)

    def __init__(self):
        self.cuda_stream = next(Stream._ids)
        self.syncs = 0

    def wait_event(self, e):
        pass

    def wait_stream(self, s):
        pass

    def record_event(self, e=None):
        e = e or Event()
        e.record(self)
        return e

    def synchronize(self):
        self.syncs += 1
        cuda.syncs += 1


cuda = types.SimpleNamespace()
cuda.capturing = False
cuda.events_complete = False
cuda.syncs = 0
cuda._stream = Stream()
cuda.Event = Event
cuda.is_current_stream_capturing = lambda: cuda.capturing
cuda.current_stream = lambda device=None: cuda._stream
cuda.synchronize = lambda device=None: setattr(cuda, 'syncs', cuda.syncs + 1)
cuda.is_available = lambda: True


class _Ctx:
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


cuda.device = _Ctx
cuda.stream = _Ctx
nn = types.SimpleNamespace(Module=object, Parameter=lambda t, requires_grad=False: t)


def reset():
    _PTRS.clear()
    _BASES.clear()
    ALLOCS.clear()
    cuda.capturing = False
    cuda.events_complete = False
    cuda.syncs = 0
    cuda._stream = Stream()
    Event.created = 0


def lookup(ptr):
    return _BASES[ptr]
