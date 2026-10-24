"""Exercise the actual request classes with Pydantic, without OpenAI/Torch.

Executes protocol.py through ChatCompletionRequest, retaining complete class bodies,
validators, field annotations and the real PDRoutingFields base. The unrelated OpenAI
Responses imports and sglang.utils import are omitted; neither supplies a field type
used by these two request classes. No conversion method is called in these tests.
"""
from __future__ import annotations
import ast
import copy
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import pydantic
import test_prefix_cache_policy as policy_tests

SRT=Path(__file__).resolve().parents[1]/'third_party/sglang/python/sglang/srt'


def schema(*, upstream=False):
    path=SRT/'entrypoints/openai/protocol.py'
    parsed=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    body=[]
    for original in parsed.body:
        node=copy.deepcopy(original)
        if isinstance(node,ast.ImportFrom) and (
            (node.module or '').startswith('openai.') or node.module=='sglang.utils'):
            continue
        if upstream and isinstance(node,ast.ClassDef) and node.name in ('CompletionRequest','ChatCompletionRequest'):
            node.body=[item for item in node.body if not (
                isinstance(item,ast.AnnAssign) and isinstance(item.target,ast.Name)
                and item.target.id=='cache_class')]
        body.append(node)
        if isinstance(node,ast.ClassDef) and node.name=='ChatCompletionRequest':break
    name='_prefix_schema_upstream' if upstream else '_prefix_schema_draft'
    module=types.ModuleType(name)
    sys.modules[name]=module
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before=schema(upstream=True)
        cls.after=schema()
        policy_tests.SRT=SRT

    def payloads(self):
        for name,base in [('CompletionRequest',{'prompt':'test'}),
                          ('ChatCompletionRequest',{'messages':[{'role':'user','content':'test'}]})]:
            for value in (None,'agent',' USER ',17,1.25,True,{'class':'agent'},['agent']):
                yield name,dict(base,cache_class=value),value

    def test_actual_models_accept_same_off_mode_values_as_upstream(self):
        for name,payload,value in self.payloads():
            with self.subTest(name=name,value=value):
                before=getattr(self.before,name).model_validate(payload)
                after=getattr(self.after,name).model_validate(payload)
                self.assertIsNone(before.model_config.get('extra'))
                self.assertIsNone(after.model_config.get('extra'))
                self.assertEqual(before.model_dump(),after.model_dump(exclude={'cache_class'}))
                self.assertEqual(after.cache_class,value)

    def test_off_hook_ignores_even_invalid_class_objects(self):
        ns=policy_tests.policy_namespace()
        class HTTPError(Exception):
            def __init__(self,**kw):self.status_code=kw['status_code']
        hook=policy_tests.extract('entrypoints/openai/serving_base.py',{'apply_cache_class'},
             dict(ns,HTTPException=HTTPError),owner='OpenAIServingBase')['apply_cache_class']
        for name,payload,_ in self.payloads():
            req=getattr(self.after,name).model_validate(payload)
            adapted=types.SimpleNamespace(kv_hints=object(),rid='schema-probe')
            old=adapted.kv_hints
            with patch.dict(os.environ,{},clear=True):hook(None,adapted,req,None)
            self.assertIs(adapted.kv_hints,old)

    def test_enabled_hook_rejects_invalid_typed_values_as_400(self):
        ns=policy_tests.policy_namespace()
        class HTTPError(Exception):
            def __init__(self,**kw):self.status_code=kw['status_code']
        hook=policy_tests.extract('entrypoints/openai/serving_base.py',{'apply_cache_class'},
             dict(ns,HTTPException=HTTPError),owner='OpenAIServingBase')['apply_cache_class']
        for name,base in [('CompletionRequest',{'prompt':'test'}),
                          ('ChatCompletionRequest',{'messages':[{'role':'user','content':'test'}]})]:
            for value in (17,1.25,True,{'class':'agent'},['agent']):
                req=getattr(self.after,name).model_validate(dict(base,cache_class=value))
                with patch.dict(os.environ,{'SGLANG_CACHE_TTL':'1'},clear=True):
                    with self.assertRaises(HTTPError) as error:
                        hook(None,types.SimpleNamespace(kv_hints=None,rid='schema-probe'),req,None)
                self.assertEqual(error.exception.status_code,400)


if __name__=='__main__':
    if len(sys.argv)>1 and not sys.argv[1].startswith('-'):
        SRT=Path(sys.argv.pop(1))/'python/sglang/srt'
    print(json.dumps({'pydantic':pydantic.__version__,'schema_source':str(SRT/'entrypoints/openai/protocol.py')}))
    unittest.main(verbosity=2)
