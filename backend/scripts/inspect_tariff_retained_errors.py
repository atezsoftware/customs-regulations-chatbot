"""Read retained DEV log files and emit only source-code literal error messages."""
import ast,json,re
from pathlib import Path
literals=set()
for path in Path('/app/onyx').rglob('*.py'):
 try:tree=ast.parse(path.read_text())
 except (ValueError,OSError,SyntaxError):continue
 for node in ast.walk(tree):
  if isinstance(node,ast.Raise)and isinstance(node.exc,ast.Call)and node.exc.args and isinstance(node.exc.args[0],ast.Constant)and isinstance(node.exc.args[0].value,str):
   text=node.exc.args[0].value
   if 8<=len(text)<=400:literals.add(text)
owned_ids={'6001':'634d428a-1675-4812-a902-229e9f3e3987','6455':'d2d0756e-e996-46cf-991b-aec286ed4aab','history':'37e9c668-6ae3-404a-a8d9-5ae7154fa95a'}
for pattern in ('onyx*.log*','api_server*.log*','celery_worker_user_file_processing*.log*','celery_worker_regulatory_annex*.log*'):
 for path in sorted(Path('/var/log/onyx').glob(pattern))[:8]:
  try:
   with path.open('rb')as stream:
    stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-8_000_000));raw=stream.read().decode('utf-8',errors='replace')
  except OSError:continue
  findings=[];timestamp=None;context=None;owned_context=[];stack=[];progress={}
  for line in raw.splitlines():
   stamp=re.search(r'(?:2026-11-03|11/03/2026)[ T][0-9:.]+',line)
   if stamp:timestamp=stamp[0]
   try:
    record=json.loads(line)
    if isinstance(record,dict):line='\n'.join(value for key in ('message','log','exc_info','stack_trace','exception')if isinstance(value:=record.get(key),str))
   except ValueError:pass
   found=[name for name,value in owned_ids.items()if value in line]
   if found:owned_context=found
   for frame in re.finditer(r'File "/app/(onyx/[A-Za-z0-9_/]+\.py)", line ([0-9]+), in ([A-Za-z0-9_]+)',line):
    stack.append({'path':frame[1],'line':int(frame[2]),'function':frame[3]});stack=stack[-12:]
   for tag in ('writer_publication_finished','contextual enrichment','contextualize','Embedding','index_user_file_impl','parse_time_limit'):
    if tag in line:progress[tag]=progress.get(tag,0)+1
   marker=re.search(r'ASv3 capability ([a-z_]+) failed',line)
   if marker:context=marker[1]
   classes=re.findall(r'\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Lost|Conflict))\s*:',line)
   messages=sorted(text for text in literals if text in line)
   if classes or messages:
    findings.append({'timestamp':timestamp,'classes':classes,'capability_context':context,'owned_context_hint_only':owned_context,'code_stack':stack,'safe_code_literals':messages[:4]});stack=[]
  print(json.dumps({'retained_log_file':path.name,'bytes_examined':len(raw.encode()),'progress_tag_counts':progress,'safe_findings':findings[-100:],'raw_log_exported':False}))
