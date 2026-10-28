"""Grouped FP8 Tensor Core wo_a experiment; quantization overhead is included.

The input is already E4M3 QDQ with g32 powers-of-two scales. Re-encoding
therefore preserves its value; tests must prove this for each actual input.
"""
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

@triton.jit
def _parts(X,W,XS,WS,P,M,NG:tl.constexpr,N:tl.constexpr,K:tl.constexpr,S:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 group=tl.program_id(2)//S;split=tl.program_id(2)%S
 m=tl.program_id(0)*BM+tl.arange(0,BM)
 n=tl.program_id(1)*BN+tl.arange(0,BN)
 steps:tl.constexpr=triton.cdiv(K//BK,S)
 acc=tl.zeros((BM,BN),tl.float32)
 for t in range(split*steps,tl.minimum((split+1)*steps,K//BK)):
  k=t*BK+tl.arange(0,BK)
  a=tl.load(X+(m[:,None]*NG+group)*K+k[None,:],m[:,None]<M,other=0.)
  b=tl.load(W+(group*N+n[None,:])*K+k[:,None],n[None,:]<N,other=0.)
  sa=tl.load(XS+(m*NG+group)*(K//BK)+t,m<M,other=0.)
  sb=tl.load(WS+(group*(N//BK)+n//BK)*(K//BK)+t,n<N,other=0.)
  acc+=tl.dot(a,b)*sa[:,None]*sb[None,:]
 tl.store(P+((split*M+m[:,None])*NG+group)*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _merge(P,Y,TOTAL,S:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 v=tl.load(P+i,i<TOTAL,other=0.)
 for s in tl.static_range(1,S):v+=tl.load(P+s*TOTAL+i,i<TOTAL,other=0.)
 tl.store(Y+i,v,i<TOTAL)

def grouped(x,w,scale,*,splits=4,bn=128,bm=None):
 assert x.is_cuda and w.is_cuda and scale.is_cuda
 assert x.dtype==torch.bfloat16 and w.dtype==torch.float8_e4m3fn and scale.dtype==torch.float32
 assert x.ndim==w.ndim==scale.ndim==3 and x.is_contiguous() and w.is_contiguous() and scale.is_contiguous()
 m,g,k=x.shape;wg,n,wk=w.shape
 assert g==wg and k==wk and scale.shape==(g,n//32,k//32) and n%32==k%32==0
 assert splits in (1,4,8)
 out=torch.empty((m,g,n),device=x.device,dtype=x.dtype)
 if not m:return out
 q,qs=sglang_per_token_group_quant_fp8(x.view(m*g,k),32,scale_ue8m0=True)
 qs=qs.contiguous();p=torch.empty((splits,m,g,n),device=x.device,dtype=torch.float32)
 bm=(16 if m<=32 else 64) if bm is None else bm
 _parts[(triton.cdiv(m,bm),triton.cdiv(n,bn),g*splits)](q,w,qs,scale,p,m,g,n,k,splits,bm,bn,32,num_warps=4,num_stages=3,enable_fp_fusion=False)
 _merge[(triton.cdiv(m*g*n,256),)](p,out,m*g*n,splits,256,num_warps=4,enable_fp_fusion=False)
 return out
