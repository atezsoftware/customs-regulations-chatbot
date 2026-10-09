"""Explicit bounded DEV provider probe; credentials and source data stay in the pod."""
import json
import re
import time
import httpx
from sqlalchemy import select
from onyx.db.legal_review_dev_preflight import inspect_dev, _inspection_engine

report = inspect_dev()
from onyx.auth.schemas import UserRole
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.legal_review_providers import resolve_legal_review_jev, _check_cost_limit
from onyx.db.llm import can_user_access_llm_provider, fetch_user_group_ids
from onyx.db.models import LLMProvider, User
from onyx.db.persona import get_default_behavior_persona

keys = {}
with _inspection_engine(), get_session_with_current_tenant() as session:
    admin = session.scalars(select(User).where(User.role == UserRole.ADMIN).order_by(User.id).limit(1)).first()
    if admin is None:
        raise SystemExit('existing_dev_admin_missing')
    persona = get_default_behavior_persona(session)
    groups = fetch_user_group_ids(session, admin)
    jev = resolve_legal_review_jev(admin, persona=persona, db_session=session)
    if jev and jev.route == 'openrouter':
        keys['jev'] = jev.api_key.get_secret_value()
    for provider in session.scalars(select(LLMProvider).where(LLMProvider.provider == 'openai').order_by(LLMProvider.id)):
        if not can_user_access_llm_provider(provider, groups, persona, is_admin=True):
            continue
        if (provider.api_base or '').strip().rstrip('/') not in ('', 'https://api.openai.com/v1'):
            continue
        if not any(m.name == 'gpt-6-luna' and m.is_visible for m in provider.model_configurations):
            continue
        if provider.api_key:
            key = provider.api_key.get_value(apply_mask=False).strip()
            if key:
                _check_cost_limit(session, key)
                keys['openai'] = key
                break

from onyx.legal_review.models import LegalDimension
old_names = [f'evidence:issue_{i}:{d.value}' for i in range(1, 4) for d in LegalDimension] + ['request_coverage', 'source_conditions', 'issue_dependencies']
text = 'The supplied document explicitly requires a declaration before release.'
instructions = 'Does the supplied document explicitly require a declaration before release?'
probes = []
if 'jev' in keys:
    for name,state,instruction in [('jev_object_state', {'original_evidence':text},instructions),('jev_long_instruction',text,instructions+(' Check only the supplied declaration requirement.'*27))]:
        probes.append((name, 'jev', 'https://openrouter.ai/api/v1/systemone', {
            'model': 'typesafe/jev-1.13', 'state': state,
            'questions': {check_name: {'type':'noul','instructions':instruction} for check_name in old_names}
        }))
results = []
for name, credential_name, endpoint, payload in probes:
    start = time.monotonic()
    result = {'probe':name, 'questions':len(payload['questions']), 'request_bytes':len(json.dumps(payload,ensure_ascii=False).encode())}
    try:
        with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as client:
            response = client.post(endpoint, headers={'Authorization':'Bearer '+keys[credential_name]}, json=payload)
        result['status'] = response.status_code
        data = response.json()
        if response.is_success:
            result['model'] = data.get('model')
            result['answer_count'] = len(data.get('answers', []))
            result['usage'] = data.get('usage')
            result['answer_types'] = sorted({a.get('type') for a in (data.get('answers',[]).values() if isinstance(data.get('answers'),dict) else data.get('answers',[]))})
        else:
            error = data.get('error', {})
            if isinstance(error,dict):
                for field in ('code','param','type'):
                    value = error.get(field)
                    if isinstance(value,(str,int)) and re.fullmatch(r'[A-Za-z0-9_.\[\]: -]{1,160}',str(value)):
                        result[field]=value
                message=str(error.get('message',''))
            else:
                message=str(error)
            for secret in keys.values():
                message=message.replace(secret,'[redacted]')
            message=re.sub(r'Bearer\s+\S+|sk-[A-Za-z0-9_-]+','[redacted]',message)
            result['message']=message[:500]
    except Exception as error:
        result['failure_type']=type(error).__name__
    result['elapsed_seconds']=round(time.monotonic()-start,3)
    results.append(result)
print(json.dumps({'environment':'dev','db_read_only':True,'probe_results':results},ensure_ascii=False,sort_keys=True))
