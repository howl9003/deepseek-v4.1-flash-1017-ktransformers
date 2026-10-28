"""One launch, ascending expert-ID FP32 adds; no changed rounding boundary."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _combine(R,I,Y,K:tl.constexpr,H:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,BK:tl.constexpr,BH:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*BH+tl.arange(0,BH)
    slot=tl.arange(0,BK)
    ids=tl.load(I+row*S0+slot*S1,slot<K,-1).to(tl.int64)
    # Pair (expert ID, original slot) preserves stable order for duplicate IDs.
    sentinel=tl.full((),0x7fffffffffffffff,tl.int64)
    keys=tl.where((slot<K)&(ids>=0),ids*BK+slot,sentinel)
    acc=tl.full((BH,),0,tl.float32);previous=tl.full((),-1,tl.int64)
    for _ in range(K):
        key=tl.min(tl.where(keys>previous,keys,sentinel),0)
        valid=key!=sentinel
        chosen=tl.where(valid,key%BK,0)
        term=tl.load(R+(row*K+chosen)*H+col,valid&(col<H),0).to(tl.float32)
        acc=tl.where(valid,acc+term,acc)
        previous=key
    tl.store(Y+row*H+col,acc,col<H)

def combine(routes,ids):
    if not routes.is_cuda or routes.dtype!=torch.bfloat16 or routes.ndim!=3 or not routes.is_contiguous():
        raise ValueError('routes must be contiguous CUDA BF16 [tokens, topk, hidden]')
    t,k,h=routes.shape
    if not 1<=k<=64 or ids.shape!=(t,k) or ids.dtype not in (torch.int32,torch.int64) or ids.device!=routes.device:
        raise ValueError('route IDs must match the route shape and device')
    out=torch.empty((t,h),dtype=torch.float32,device=routes.device)
    _combine[(t,tr.cdiv(h,256))](routes,ids,out,k,h,*ids.stride(),tr.next_power_of_2(k),256,enable_fp_fusion=False)
    return out
