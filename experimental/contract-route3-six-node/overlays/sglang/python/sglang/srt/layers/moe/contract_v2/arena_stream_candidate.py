"""Contract BufferB zero-copy staging, with packed weights and scales at explicit offsets.

One mapping/registration per native projection. Copy each complete expert block
once, then read its original packed weights and FP32 power-of-two scales on GPU.
Derived native metadata travels with the block but never affects GPU arithmetic.
"""
import mmap
import time
import torch


def validate_layout(parts, hidden, intermediate, experts):
    if not parts or intermediate % len(parts) or intermediate // len(parts) % 32:
        raise RuntimeError('unaligned contract NUMA slices')
    rows=intermediate//len(parts)
    weight_bytes=rows*hidden//2
    scale_bytes=rows*(hidden//32)*4
    for part in parts:
        if len(part)!=3: raise RuntimeError('contract needs three interleaved arenas per part')
        for fd,size,stride,weight_offset,scale_offset in part:
            if (fd<0 or stride%64 or size!=stride*experts or weight_offset!=0
                or scale_offset%4 or scale_offset<weight_bytes
                or scale_offset+scale_bytes>stride):
                raise RuntimeError('invalid contract arena byte layout')
    return rows


def map_arenas(st, method):
    from sglang.srt.layers.moe.kt_ep_wrapper import KT_LAYERS
    from sglang.srt.layers.moe.kt_stream_prefill import _rank_segments, COLLAPSE, _collapse_arenas
    if st.rank!=0 or st.world!=1:
        raise RuntimeError('contract arena export currently requires one GPU rank')
    st._contract_arena_layout={}
    st._registered=set()
    all_fds=[]
    for lid,wrapper in sorted(KT_LAYERS.items()):
        parts=wrapper.wrapper.moe.contract_expert_arena_infos()
        validate_layout(parts,st.K,st.N,wrapper.global_num_experts)
        segments=_rank_segments(0,1,len(parts),st.N)
        if st._contract_arena_layout and st._segs!=segments:
            raise RuntimeError('contract layer NUMA layouts differ')
        st._segs=segments
        st._seg_whole=len(parts)==1
        st._contract_arena_layout[lid]=parts
        mappings=[]
        for part in parts:
            arenas=[]
            for fd,size,stride,_,_ in part:
                mm=mmap.mmap(fd,size,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE)
                arenas.append((torch.frombuffer(mm,dtype=torch.uint8),stride,mm))
                all_fds.append((fd,size))
            mappings.append(arenas)
        st._arena_parts[lid]=mappings
        st._arenas[lid]=mappings[0]
    if not st._contract_arena_layout: raise RuntimeError('no contract native arenas')
    if COLLAPSE: _collapse_arenas(all_fds)


def stage(st,lid,gi,experts,stats):
    st._register_arenas(lid)
    slot=st.slots[gi%2]
    parts=st._arena_parts[lid]
    layout=st._contract_arena_layout[lid]
    signature=tuple(tuple((v[2],v[3],v[4]) for v in part) for part in layout)
    if not hasattr(slot,'contract_raws'):
        from sglang.srt.layers.moe import kt_lend
        with kt_lend.scratch_region():
            slot.contract_raws=[[torch.empty((st.G,stride),dtype=torch.uint8,device=st.device)
                                 for _,stride,_ in part] for part in parts]
        slot.contract_layout=signature
        slot.raw=slot.contract_raws[0][0]  # Existing group helper only uses this as a ready marker.
    elif slot.contract_layout!=signature:
        raise RuntimeError('contract stream slot layout changed after graph capture')
    base,n=experts[0],len(experts)
    if experts!=list(range(base,base+n)) or not 0<n<=st.G:
        raise RuntimeError('contract DMA requires a contiguous expert group')
    start=time.perf_counter()
    with torch.cuda.stream(st.transfer_stream):
        if slot.compute_used: st.transfer_stream.wait_event(slot.compute_done)
        for part,raws in zip(parts,slot.contract_raws):
            for (host,stride,_),raw in zip(part,raws):
                raw[:n].view(-1).copy_(host[base*stride:(base+n)*stride],non_blocking=True)
        slot.dma_done.record(st.transfer_stream)
    stats['dma_enq']+=time.perf_counter()-start


def pack_payload(st,slot,n):
    from .adapter import _scale_codes
    N,K=st.N,st.K
    w13=torch.empty((n,2*N,K//2),dtype=torch.uint8,device=st.device)
    s13=torch.empty((n,2*N,K//32),dtype=torch.uint8,device=st.device)
    w2=torch.empty((n,K,N//2),dtype=torch.uint8,device=st.device)
    s2=torch.empty((n,K,N//32),dtype=torch.uint8,device=st.device)
    next_row=0
    for (_,lo,count,d,m),raws,layout in zip(st._segs,slot.contract_raws,slot.contract_layout):
        if d!=next_row or lo%32 or count%32 or m%32 or lo+count>m:
            raise RuntimeError('contract slices must cover aligned rows exactly once')
        next_row+=count
        for pi,offset in ((0,0),(1,N)):
            raw=raws[pi][:n]
            scale_offset=layout[pi][2]
            w13[:,offset+d:offset+d+count].copy_(raw[:,lo*K//2:(lo+count)*K//2].reshape(n,count,K//2))
            scales=raw[:,scale_offset:scale_offset+m*(K//32)*4].view(torch.float32)
            sf=scales[:,lo*(K//32):(lo+count)*(K//32)].reshape(n,count,K//32)
            s13[:,offset+d:offset+d+count].copy_(_scale_codes(sf))
        raw=raws[2][:n]
        w2[:,:,d//2:(d+count)//2].copy_(raw[:,:K*m//2].reshape(n,K,m//2)[:,:,lo//2:(lo+count)//2])
        scales=raw[:,layout[2][2]:layout[2][2]+K*(m//32)*4].view(torch.float32)
        sf=scales.reshape(n,K,m//32)[:,:,lo//32:(lo+count)//32]
        s2[:,:,d//32:(d+count)//32].copy_(_scale_codes(sf))
    if next_row!=N: raise RuntimeError('incomplete contract streamed payload')
    return w13,s13,w2,s2


def load_hot(st,layer,expert,dst,order,dyn):
    """Load a resident/hot slot from the same interleaved native memory.

    Called on the existing hot-lender stream, whose events protect destination
    reuse. Keep its scratch reuse and exact exponent extraction unchanged.
    """
    st._register_arenas(layer.layer_idx)
    w13,s13,w2,s2=(dst[k].view(torch.uint8) for k in
                   (('w13',False),('w13',True),('w2',False),('w2',True)))
    n,k=w13.shape[0]//2,w2.shape[0]
    a,b=(0,1) if order==0 else (1,0)
    stage=dyn['stage']
    for (_,lo,count,d,m),arenas,layout in zip(st._segs,st._arena_parts[layer.layer_idx],st._contract_arena_layout[layer.layer_idx]):
        kb,kg=w13.shape[1],s13.shape[1]
        for source,row0 in ((a,0),(b,n)):
            host,stride,_=arenas[source]
            scale=layout[source][4]
            w13[row0+d:row0+d+count].view(-1).copy_(host[expert*stride+lo*kb:expert*stride+(lo+count)*kb],non_blocking=True)
            cnt=count*kg
            stage[:cnt].view(torch.uint8).copy_(host[expert*stride+scale+4*lo*kg:expert*stride+scale+4*(lo+count)*kg],non_blocking=True)
            s13[row0+d:row0+d+count].view(-1).copy_(((stage[:cnt].view(torch.int32)>>23)&255).to(torch.uint8))
        host,stride,_=arenas[2]
        scale=layout[2][4]
        if dyn.get('stage_b') is None or dyn['stage_b'].numel()<k*m//2:
            dyn['stage_b']=torch.empty(k*m//2,dtype=torch.uint8,device=w2.device)
        sb=dyn['stage_b'][:k*m//2]
        sb.copy_(host[expert*stride:expert*stride+k*m//2],non_blocking=True)
        w2[:,d//2:(d+count)//2].copy_(sb.view(k,m//2)[:,lo//2:(lo+count)//2])
        cnt=k*(m//32)
        if stage.numel()<cnt:dyn['stage']=stage=torch.empty(cnt,dtype=torch.float32,device=w2.device)
        stage[:cnt].view(torch.uint8).copy_(host[expert*stride+scale:expert*stride+scale+4*cnt],non_blocking=True)
        s2[:,d//32:(d+count)//32].copy_(((stage[:cnt].view(torch.int32).view(k,m//32)[:,lo//32:(lo+count)//32]>>23)&255).to(torch.uint8))
