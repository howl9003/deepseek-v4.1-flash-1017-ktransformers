"""Default-off C1 integration; exact precompiled kernel, immutable per-step flags.

No compilation in the server. The kernel's sticky atomic flag is shared across
the candidate's serial forward-stream workspaces. A per-result clone is queued
after each forward on that stream, before the next forward may run. Existing
asynchronous result readback copies the clone and records its usual event.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
import torch

_enabled = os.environ.get('SGLANG_FP8_PATHA_EXPERT_STATIC', '0') == '1'
_pa = None
_flag = None
_device = None
_warming = False
_binary_sha = '4d44ef332bfd9aba8cdb963225fd55ec9c0b2b5281f42adcc93fed92a3e08f61'
_python_sha = '6b215f8a669d6f85abbdd17991e316e4455991c69d141d424d1d2ad6ca83ed75'

def _module():
    global _pa
    if not _enabled:
        raise RuntimeError('C1 static expert selector is off')
    if _pa is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('C1 must be prepared outside capture')
        root=Path(os.environ['SGLANG_FP8_PATHA_EXPERT_STATIC_DIR']).resolve()
        path=root/'patha_expert_static.py'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=_python_sha:
            raise RuntimeError('C1 Python source pin mismatch')
        binary=Path(os.environ['SGLANG_FP8_PATHA_EXPERT_STATIC_BINARY']).resolve()
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=_binary_sha:
            raise RuntimeError('C1 native binary pin mismatch')
        name='patha_expert_static'
        if name in sys.modules:
            if Path(sys.modules[name].__file__).resolve()!=path:
                raise RuntimeError('C1 module already loaded from a different path')
            pa=sys.modules[name]
        else:
            spec=importlib.util.spec_from_file_location(name,path)
            pa=importlib.util.module_from_spec(spec);sys.modules[name]=pa
            spec.loader.exec_module(pa)
        knobs={'PATHA_EXPERT_LUT':0,'PATHA_STAGES':3,'PATHA_EXPERT_BPC':8,'PATHA_EXPERT_TC':0}
        if pa.build_knobs()!=knobs:
            raise RuntimeError('C1 requires the qualified LUT-off configuration')
        extname='patha_expert_static_sm89_e0_s3_b8_t0'
        if binary.stem!=extname:
            raise RuntimeError('C1 native filename does not match the qualified module')
        spec=importlib.util.spec_from_file_location(extname,binary)
        ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
        if {k:int(ext.build_config(i)) for i,k in enumerate(knobs)}!=knobs:
            raise RuntimeError('C1 native build configuration mismatch')
        ext.configure()
        pa._EXT=ext                    # Existing loader returns immediately; no JIT.
        directory=Path(os.environ['SGLANG_FP8_PATHA_EXPERT_STATIC_ORIGIN_DIR']).resolve()
        if not directory.is_dir():raise RuntimeError('C1 native-origin directory was not prepared')
        receipt={'pid':os.getpid(),'binary':str(binary),'binary_sha256':_binary_sha,
                 'python_source':str(path),'python_sha256':_python_sha,
                 'knobs':knobs,'loader':'direct_pinned_binary_no_compiler'}
        with (directory/(str(os.getpid())+'.json')).open('x') as f:
            json.dump(receipt,f,indent=2);f.write('\n')
        _pa=pa
    return _pa

def warmup(device):
    global _flag,_device,_warming
    if not _enabled:
        return
    d=torch.device(device)
    if d.index is None:d=torch.device('cuda',torch.cuda.current_device())
    if _flag is not None:
        if d!=_device:raise RuntimeError('C1 candidate supports one CUDA device per process')
        return
    if _warming or torch.cuda.is_current_stream_capturing():
        raise RuntimeError('C1 workspace warmup must precede capture')
    _warming=True
    try:
        pa=_module()
        for topk in range(1,7):pa.warmup(d,M_list=tuple(range(1,9)),topk=topk)
        shared=torch.zeros((1,),dtype=torch.int32,device=d)
        for ws in pa._WORKSPACES.values():
            if ws.device!=d:raise RuntimeError('C1 workspace belongs to a different device')
            ws.flag=shared
        _flag,_device=shared,d
        torch.cuda.synchronize(d)       # Complete initial zero before capture on another stream.
    finally:
        _warming=False

def expert_routes(x,w13,s13,w2,s2,ids,weights,*,out=None):
    warmup(x.device)
    return _module().expert_routes_patha(x,w13,s13,w2,s2,ids,weights,out=out)

def snapshot_after_forward():
    if not _enabled or _flag is None:
        raise RuntimeError('C1 forward completed without an armed error flag')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('C1 result snapshot must be outside the captured graph')
    # Own storage is essential: async copy can overlap the next graph replay.
    return _flag.clone()

def check_result(result):
    value=result.patha_static_flags
    if value is None:
        raise RuntimeError('C1 result is missing its error snapshot; output discarded')
    # copy_done was synchronized by the existing result processor; the non-overlap
    # fallback is a direct scalar read after its existing token synchronization.
    bits=int(value.item())
    if bits:
        raise RuntimeError('C1 expert flags %#x (%s); output discarded' % (bits,_module().describe_flags(bits)))
