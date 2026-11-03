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
for pattern in ('api_server*.log*','celery_worker_user_file_processing*.log*','celery_worker_regulatory_annex*.log*'):
 for path in sorted(Path('/var/log/onyx').glob(pattern))[:8]:
  try:
   with path.open('rb')as stream:
    stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-8_000_000));raw=stream.read().decode('utf-8',errors='replace')
  except OSError:continue
  findings=[];timestamp=None;context=None
  for line in raw.splitlines():
   stamp=re.search(r'(?:2026-11-03|11/03/2026)[ T][0-9:.]+',line)
   if stamp:timestamp=stamp[0]
   marker=re.search(r'ASv3 capability ([a-z_]+) failed',line)
   if marker:context=marker[1]
   classes=re.findall(r'\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Lost|Conflict))\s*:',line)
   messages=sorted(text for text in literals if text in line)
   if classes or messages:
    findings.append({'timestamp':timestamp,'classes':classes,'capability_context':context,'safe_code_literals':messages[:4]})
  print(json.dumps({'retained_log_file':path.name,'bytes_examined':len(raw.encode()),'safe_findings':findings[-100:],'raw_log_exported':False}))
