"""Experimental BF16 x BF16 -> FP32 with reduction independent of row count.

No automatic activation or monkey patch. Fixed K slices and ordered FP32 merge;
M stays runtime-valued so different row counts share the same arithmetic kernel.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _partials(X,W,P,M,N:tl.constexpr,K:tl.constexpr,S:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM)
 n=tl.program_id(1)*BN+tl.arange(0,BN)
 split=tl.program_id(2)
 span:tl.constexpr=triton.cdiv(K,S*BK)*BK
 ks=split*span+tl.arange(0,BK)
 acc=tl.zeros((BM,BN),tl.float32)
 for off in range(0,span,BK):
  k=ks+off
  x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),other=0.)
  w=tl.load(W+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),other=0.)
  acc=tl.dot(x,w,acc)
 tl.store(P+split*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _merge(P,Y,TOTAL,S:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 result=tl.load(P+i,i<TOTAL,other=0.)
 for s in tl.static_range(1,S):result=result+tl.load(P+s*TOTAL+i,i<TOTAL,other=0.)
 tl.store(Y+i,result,i<TOTAL)

def linear(x,w,*,splits=8,bn=32,bk=64):
 assert x.is_cuda and w.is_cuda and x.device==w.device
 assert x.dtype==w.dtype==torch.bfloat16 and x.ndim==w.ndim==2
 assert x.is_contiguous() and w.is_contiguous() and x.shape[1]==w.shape[1]
 assert splits in (1,4,8) and bn in (32,64) and bk in (32,64)
 m,k=x.shape;n=w.shape[0]
 if not m:return torch.empty((0,n),device=x.device,dtype=torch.float32)
 partial=torch.empty((splits,m,n),device=x.device,dtype=torch.float32)
 _partials[(triton.cdiv(m,16),triton.cdiv(n,bn),splits)](x,w,partial,m,n,k,splits,16,bn,bk,num_warps=4,enable_fp_fusion=False)
 if splits==1:return partial[0]
 output=torch.empty((m,n),device=x.device,dtype=torch.float32)
 _merge[(triton.cdiv(m*n,256),)](partial,output,m*n,splits,256,num_warps=4,enable_fp_fusion=False)
 return output
