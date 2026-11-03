"""Read-only DEV pod/log diagnostics; emits no log bodies, SQL or provider payloads."""
import datetime as dt
import json
import re
import subprocess

NAMESPACE='customs-regulations-dev'
START='2026-11-03T14:00:00Z'
END='2026-11-03T14:35:00Z'

def command(args: list[str], input_text: str | None = None, timeout: int = 90) -> str:
    result=subprocess.run(args,input=input_text,capture_output=True,text=True,timeout=timeout,check=False)
    if result.returncode:
        failure={'diagnostic_command_failed':args[:3],'exit_code':result.returncode};print(json.dumps(failure));return json.dumps(failure)
    return result.stdout

def safe_findings(raw: str, origin: str)->list[dict[str,object]]:
    rows=[]
    for line in raw.splitlines():
        try:
            value=json.loads(line)
            if isinstance(value,dict):line=str(value.get('log') or value.get('message') or '')
        except ValueError:pass
        classes=re.findall(r'(?:\b[a-z_]+\.)*\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Lost|Conflict))\s*:',line)
        capability=re.search(r'ASv3 capability ([a-z_]+) failed',line)
        target='37e9c668-6ae3-404a-a8d9-5ae7154fa95a' in line and ('500' in line or '502' in line)
        if not classes and not capability and not target:continue
        timestamp=re.search(r'2026-11-03[T ][0-9:.+Z-]+',line)
        tags=[tag for tag,pattern in {'redis':'redis','pool_limit':'queuepool|pool limit|pool timeout','connection_reset':'connection reset','connection_refused':'connection refused|error 111','connection_closed':'connection.*closed|closed.*connection','cache':'cache','codec':'checkpoint.*budget|checkpoint.*decode','multiple_results':'multiple rows|multiple results','float_json':'out of range float|json compliant','timeout':'timed out|timeout','database':'psycopg|sqlalchemy','authentication':'authentication|permission denied'}.items() if re.search(pattern,line,re.I)]
        rows.append({'origin':origin,'timestamp':timestamp[0] if timestamp else None,'exception_classes':classes,'capability':capability[1]if capability else None,'target_http_error':target,'categories':tags,'line_sha256':__import__('hashlib').sha256(line.encode()).hexdigest()})
    return rows[:250]

pods=json.loads(command(['kubectl','get','pods','-n',NAMESPACE,'-o','json']))
findings=[]
for pod in pods.get('items',[]):
    statuses={s['name']:s for s in pod.get('status',{}).get('containerStatuses',[])}
    candidates=[c for c in pod['spec']['containers'] if 'customs-regulations-backend-lite-dev:'in c['image'] or 'redis'in c['image'].lower()]
    for container in candidates:
        status=statuses.get(container['name'],{})
        print(json.dumps({'pod':pod['metadata']['name'],'container':container['name'],'image':container['image'],'phase':pod.get('status',{}).get('phase'),'created_at':pod['metadata'].get('creationTimestamp'),'deleting_at':pod['metadata'].get('deletionTimestamp'),'grace_seconds':pod['spec'].get('terminationGracePeriodSeconds'),'restarts':status.get('restartCount'),'ready':status.get('ready'),'last_termination':{k:v for k,v in status.get('lastState',{}).get('terminated',{}).items() if k in{'reason','exitCode','startedAt','finishedAt'}},'node':pod['spec'].get('nodeName')}))
        if 'customs-regulations-backend-lite-dev:'not in container['image']:continue
        args=['kubectl','logs',pod['metadata']['name'],'-n',NAMESPACE,'-c',container['name'],'--since-time='+START,'--timestamps=true','--tail=20000']
        findings.extend(safe_findings(command(args),pod['metadata']['name']))
        from pathlib import Path
        retained=command(['kubectl','exec','-i',pod['metadata']['name'],'-n',NAMESPACE,'-c',container['name'],'--','python','-'],Path('backend/scripts/inspect_tariff_retained_errors.py').read_text())
        for record in retained.splitlines():
            try:
                payload=json.loads(record)
                if any(key in payload for key in ('retained_log_file','active_owned_task','active_worker_count','inspect_exception_class')):print(json.dumps(payload))
            except ValueError:pass
        if status.get('restartCount',0):findings.extend(safe_findings(command(args+['--previous']),pod['metadata']['name']+':previous'))
events=json.loads(command(['kubectl','get','events','-n',NAMESPACE,'-o','json']))
for event in events.get('items',[]):
    at=event.get('lastTimestamp')or event.get('eventTime')or ''
    if START<=at<=END and event.get('reason')in{'Killing','BackOff','Unhealthy','Failed','Evicted','Preempting','NodeNotReady'}:
        print(json.dumps({'event_reason':event.get('reason'),'object':event.get('involvedObject',{}).get('name'),'at':at,'count':event.get('count')}))
# Deleted pods require the existing container log archive, when enabled.
groups=json.loads(command(['aws','logs','describe-log-groups','--region','eu-central-1','--log-group-name-prefix','/aws/containerinsights/atez-dev-cluster','--output','json']))
for group in groups.get('logGroups',[]):
    name=group['logGroupName']
    if not name.endswith('/application'):continue
    start=int(dt.datetime.fromisoformat(START.replace('Z','+00:00')).timestamp()*1000)
    end=int(dt.datetime.fromisoformat(END.replace('Z','+00:00')).timestamp()*1000)
    data=json.loads(command(['aws','logs','filter-log-events','--region','eu-central-1','--log-group-name',name,'--start-time',str(start),'--end-time',str(end),'--filter-pattern','"customs-regulations-dev"','--limit','10000','--no-paginate','--output','json']))
    print(json.dumps({'archive':name,'events_examined':len(data.get('events',[])),'has_next_page':bool(data.get('nextToken'))}))
    for event in data.get('events',[]):findings.extend(safe_findings(event.get('message',''),name))
print(json.dumps({'safe_exception_findings':findings[:300],'read_only':True,'raw_log_bodies_exported':False}))

expected_revision='5a11809571cb930a5f90c2f20833986336ff6f32'
for pod in pods.get('items',[]):
    if pod['metadata'].get('deletionTimestamp')or pod['metadata'].get('labels',{}).get('app')!='dev-customs-regulations-background':continue
    statuses={item['name']:item for item in pod.get('status',{}).get('containerStatuses',[])}
    for container in pod['spec']['containers']:
        if not container['image'].endswith('customs-regulations-backend-lite-dev:'+expected_revision)or not statuses.get(container['name'],{}).get('ready'):continue
        raw_probe=command(['kubectl','exec','-i',pod['metadata']['name'],'-n',NAMESPACE,'-c',container['name'],'--','python','-',expected_revision],Path('backend/scripts/inspect_tariff_native_pdf.py').read_text(),timeout=210)
        for record in raw_probe.splitlines():
            try:
                payload=json.loads(record)
                if 'elapsed'in payload:print(json.dumps({'native_pdf_diagnostic':payload}))
            except ValueError:pass

all_groups=json.loads(command(['aws','logs','describe-log-groups','--region','eu-central-1','--query','logGroups[].logGroupName','--output','json']))
if isinstance(all_groups,list):print(json.dumps({'available_relevant_log_groups':[name for name in all_groups if any(k in name.lower() for k in ('customs','containerinsights','atez-dev'))][:100]}))
