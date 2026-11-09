"""Disposable, read-only live measurement of the deployed simple-search function."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, cast

os.environ['USER_USAGE_TRACKING_ENABLED'] = 'false'
os.environ['PGOPTIONS'] = '-c default_transaction_read_only=on'

from sqlalchemy import text
import tiktoken
from google.genai.models import AsyncModels
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.models import User
import onyx.context.search.utils as search_utils
from onyx.reranking.openrouter import OpenRouterRerankClient
from onyx.server.features.regulatory.simple_search_api import simple_search
from onyx.server.features.regulatory.simple_search_models import SimpleSearchRequest
from onyx.tracing.framework import set_trace_processors
from onyx.tracing.framework.processor_interface import TracingProcessor
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span
from onyx.tracing.framework.traces import Trace
from shared_configs.contextvars import get_current_tenant_id

QUERIES = [
    'dahilde işleme rejimi',
    'gümrük kıymetinde royalti ve lisans ödemeleri',
    'ithalatta gözetim uygulaması',
    'menşe şahadetnamesi ve EUR.1 dolaşım belgesi',
    'gümrükte fazla çalışma ücretleri',
    'geçici ithalat rejimi süre uzatımı',
    'antrepo rejiminde eşyanın devri',
    'geri gelen eşyanın gümrük vergisinden muafiyeti',
    'İthal ettiğim ürünün gümrük vergisi eksik hesaplanmışsa sonradan hangi ceza uygulanır?',
    'Yurt dışından getirilen makineyi Türkiye’de işledikten sonra yeniden ihraç etmek için hangi gümrük rejimi uygulanır?',
]
active: dict[str, Any] = {}
lock = threading.Lock()
tokenizer = tiktoken.get_encoding('cl100k_base')

def emit(kind: str, value: dict[str, Any]) -> None:
    print(kind + ' ' + json.dumps(value, ensure_ascii=False, default=str), flush=True)

class Collector(TracingProcessor):
    def on_trace_start(self, trace: Trace) -> None: pass
    def on_trace_end(self, trace: Trace) -> None: pass
    def on_span_start(self, span: Span[Any]) -> None: pass
    def shutdown(self) -> None: pass
    def force_flush(self) -> None: pass
    def on_span_end(self, span: Span[Any]) -> None:
        data = span.span_data
        if not isinstance(data, GenerationSpanData): return
        cfg = data.model_config or {}
        if cfg.get('flow') not in ('catalog_search_summary', 'catalog_query_preparation'): return
        with lock:
            active.setdefault('llm_events', []).append({'span_id':span.span_id,'flow':cfg.get('flow'),'model':data.model,'usage':data.usage})

set_trace_processors([Collector()])
SqlEngine.set_app_name('simple_search_ten_query_benchmark')
SqlEngine.init_engine(pool_size=2,max_overflow=2)
original_cache_read = search_utils.get_cached_query_embeddings

def observe_cache_read(*args: Any, **kwargs: Any) -> Any:
    result = original_cache_read(*args, **kwargs)
    active['embedding_cache_hits'] = sum(value is not None for value in result)
    active['embedding_cache_misses'] = sum(value is None for value in result)
    return result

search_utils.get_cached_query_embeddings = observe_cache_read
original_rerank = OpenRouterRerankClient._post_json

def observe_rerank(self: OpenRouterRerankClient, *, url: str, api_key: str, request_payload: dict[str, Any]) -> Any:
    docs = request_payload.get('documents', [])
    q = request_payload.get('query', '')
    event: dict[str, Any] = {'model':request_payload.get('model'),'documents':len(docs), 'estimated_input_tokens_cl100k':sum(len(tokenizer.encode(q))+len(tokenizer.encode(doc)) for doc in docs)}
    try:
        response = original_rerank(self, url=url, api_key=api_key, request_payload=request_payload)
        if isinstance(response, dict):
            event.update({'response_id':response.get('id'),'usage':response.get('usage'),'meta':response.get('meta'),'response_keys':list(response.keys())})
        return response
    except Exception as exc:
        event['error_type'] = type(exc).__name__
        raise
    finally:
        with lock: active.setdefault('rerank_events', []).append(event)

OpenRouterRerankClient._post_json = observe_rerank
original_embed = AsyncModels.embed_content

async def observe_embed(self: AsyncModels, *args: Any, **kwargs: Any) -> Any:
    response = await original_embed(self, *args, **kwargs)
    event: dict[str, Any] = {'model':kwargs.get('model'), 'statistics':[e.statistics.model_dump(mode='json') if e.statistics else None for e in (response.embeddings or [])]}
    dump = response.model_dump(mode='json', exclude={'embeddings','sdk_http_response'})
    event['metadata'] = dump
    with lock: active.setdefault('embedding_events', []).append(event)
    return response

AsyncModels.embed_content = observe_embed
start,stop = int(sys.argv[1]),int(sys.argv[2])
assert 0 <= start < stop <= 10
assert os.environ.get('POSTGRES_DB') == 'customs-regulations-dev', 'Unexpected database scope'
emit('BENCH_SETUP', {'start':start,'stop':stop,'utc':datetime.now(timezone.utc).isoformat(),'database':os.environ.get('POSTGRES_DB'),'method':'deployed simple_search function; isolated process; unchanged providers/retrieval; read-only SQL'})
for index in range(start,stop):
    active = {'index':index+1,'query':QUERIES[index],'llm_events':[],'embedding_events':[],'rerank_events':[]}
    began=time.monotonic()
    try:
        with get_session_with_tenant(tenant_id=get_current_tenant_id()) as session:
            session.execute(text('SET TRANSACTION READ ONLY'))
            result=simple_search(SimpleSearchRequest(query=QUERIES[index]), cast(User,None), session)
            active.update({'request_id':str(result.request_id),'summary_usage':result.summary_usage.model_dump(mode='json') if result.summary_usage else None,'summary_error_code':result.summary_error_code,'result_count':result.search.total,'returned_items':len(result.search.items),'summary_present':bool(result.summary),'query_rewritten':result.query_rewritten,'search_query':result.search_query})
    except Exception as exc:
        active['error_type']=type(exc).__name__
        import traceback
        active['error_frames']=[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)[-8:]]
    active['elapsed_seconds']=round(time.monotonic()-began,3)
    emit('BENCH_QUERY',active)
    if active.get('error_type') and not any(active[k] for k in ('llm_events','embedding_events','rerank_events')):
        raise SystemExit('Measurement setup failed before a provider call')
emit('BENCH_DONE', {'utc':datetime.now(timezone.utc).isoformat(),'start':start,'stop':stop})
