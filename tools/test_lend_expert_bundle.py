"""CPU regression using production lend/remap functions and identity sentinels."""
import ast,contextlib,hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace as NS
HERE=Path(__file__).resolve().parent
if len(sys.argv)>1:
    fixed=(Path(sys.argv[1])/'python/sglang/srt/layers/moe/kt_lend.py').read_text(encoding='utf-8')
else:
    text=(HERE.parent/'patches/sglang-tier1/13-prefill-lend.patch').read_text(encoding='utf-8')
    section=text.split('diff --git a/python/sglang/srt/layers/moe/kt_lend.py ',1)[1].split('\ndiff --git ',1)[0]
    fixed='\n'.join(line[1:] for line in section.splitlines() if line.startswith('+') and not line.startswith('+++'))+'\n'
class Tensor:
    def __init__(self,shape,dtype='uint8',payload=0):
        self.shape=tuple(shape) if isinstance(shape,(tuple,list)) else (shape,)
        self.dtype,self.payload=dtype,payload
        self.device=NS(type='cuda');self.is_cuda=True;self.storage={}
    def numel(self):return math.prod(self.shape)
    def element_size(self):return 1
    def is_contiguous(self):return True
    def copy_(self,other):self.payload=other.payload;return self
    @property
    def data(self):return self
    @data.setter
    def data(self,other):self.payload=other.payload
class Uninitialized:pass
class Layer:
    def __init__(self,n):
        self.params={'w13_weight':Tensor((n,4608,2560)),'w2_weight':Tensor((n,5120,1152)),
                     'w13_weight_scale_inv':Tensor((n,4608,160)),'w2_weight_scale_inv':Tensor((n,5120,72))}
    def named_parameters(self):return self.params.items()
    def get_parameter(self,name):return self.params[name]
    def modules(self):return [self]
def place(span,item):return span.storage.setdefault(item[1],Tensor(item[2],item[3]))
base=NS(Tensor=Tensor,Uninitialized=Uninitialized,Layer=Layer,place=place)
functions={'_nbytes','_align','_pack_resident','_point','bind_model','_lendable_size'}
nodes=[n for n in ast.parse(fixed).body if isinstance(n,ast.FunctionDef) and n.name in functions]
code=compile(ast.Module(body=nodes,type_ignores=[]),'fixed-kt-lend','exec')
results=[]
for enabled in (False,True):
    for count in (1,2,4):
        layer=base.Layer(count)
        torch=NS(Tensor=base.Tensor,uint8='uint8',nn=NS(parameter=NS(UninitializedParameter=base.Uninitialized)),
            empty=lambda shape,**kw:base.Tensor(shape,kw.get('dtype','uint8')),
            cuda=NS(synchronize=lambda *a:None,empty_cache=lambda:None))
        ns=dict(torch=torch,math=math,ENABLED=True,MIN_BYTES=1<<20,SKIP=['engram'],PFONLY=[],RESIDENT=['hc_'],
            _DYNAMIC_RESIDENT_SCALES=enabled,STATE={'phase':'load'},_decoder_layers=lambda m:[layer],
            _place=base.place,_tms=lambda:NS(region=lambda **kw:contextlib.nullcontext()),TAG_WEIGHTS='weights',
            _hook_entries=lambda *a:None,scratch_empty=lambda shape,dtype,dev:base.Tensor(shape,dtype),_log=lambda *a:None)
        exec(code,ns);ns['bind_model'](None)
        layout=ns['STATE']['layouts'][0];mirror=ns['STATE']['dev_spans'][0]
        for previous in (55,361,55,361):
            for p in layer.params.values():p.payload=previous
            ns['_point'](layer,mirror,layout)
            values={name:p.payload for name,p in layer.params.items()}
            if enabled or count==4:assert set(values.values())=={0}
            else:assert len(set(values.values()))==2
            results.append(dict(enabled=enabled,n=count,previous=previous,payloads=values))
        if enabled:
            assert len(layout)==4
            assert not ns['_lendable_size']('router.weight',base.Tensor((16,16)))
            assert not ns['_lendable_size']('w2_weight_scale_inv',base.Tensor((0,16)))
            assert ns['_lendable_size']('mlp.experts.w13_weight_scale',base.Tensor((1,16)))
            half=base.Tensor((1,16));half.element_size=lambda:2
            assert not ns['_lendable_size']('mlp.experts.w13_weight_scale_inv',half)
        else:
            assert [x[1] for x in layout]=={1:['w13_weight','w2_weight'],2:['w13_weight','w2_weight','w13_weight_scale_inv'],4:list(layer.params)}[count]
tree=ast.parse(fixed)
# One size predicate is used at construction, relayout and post-load rebound check.
for fn in ('bind_model','_relayout','after_module'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_lendable_size' for n in ast.walk(node))
constant=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_DYNAMIC_RESIDENT_SCALES' for t in n.targets))
for missing in (None,'KT_KV_HOT_LEND','KT_KV_HOT_DYNAMIC_SLOTS','KT_KV_HOT_DYNAMIC_RESIDENT'):
    env={k:'1' for k in ('KT_KV_HOT_LEND','KT_KV_HOT_DYNAMIC_SLOTS','KT_KV_HOT_DYNAMIC_RESIDENT') if k!=missing}
    ns={'os':NS(environ=env)};exec(compile(ast.Module(body=[constant],type_ignores=[]),'flag','exec'),ns)
    assert ns['_DYNAMIC_RESIDENT_SCALES']==(missing is None)
out=dict(status='PASS',transitions=len(results),cases=results,fixed_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
    limitations='CPU metadata/storage doubles; actual fixed GPU payload gate required')


print('PASS: 24 repeated transitions; N1/N2/N4, dynamic on/off, unchanged disabled layouts, unrelated/empty/non-byte exclusions, all three selection sites')
