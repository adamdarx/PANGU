from pathlib import Path
import json,subprocess,time,os,datetime
H=Path(__file__).parent
while True:
 if (H/'STOP').exists(): raise SystemExit('Latest user policy disables blanket archival')
 try:
  d=json.loads((H/'INVENTORY.json').read_text());assert d['entries'];break
 except (ValueError,AssertionError,FileNotFoundError):time.sleep(10)
with (H/'worker.log').open('ab',buffering=0) as f:
 p=subprocess.Popen(['/home/adamdarx/.local/pangu-host/bin/python','-u',str(H/'worker.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
(H/'LAUNCH.json').write_text(json.dumps(dict(pid=p.pid,files=d['files'],raw_bytes=d['raw_bytes'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
