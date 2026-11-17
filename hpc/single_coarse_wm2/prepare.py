from pathlib import Path
import hashlib,json,re
r=Path(__file__).parent;out=r/'inputs';out.mkdir(exist_ok=True);rows=[]
def parse(s):
 d={};section=''
 for line in s.splitlines():
  line=line.split('#',1)[0].strip()
  if line.startswith('<'):section=line.strip('<>')
  elif '=' in line:
   k,v=line.split('=',1);d[section,k.strip()]=v.strip()
 return d
allowed={(s,'nx'+str(a)) for s in ['parthenon/mesh','parthenon/meshblock'] for a in (1,2,3)}
for kind in ('stationary','spinning'):
 raw=(r/'source-records'/(kind+'.in')).read_text();original=parse(raw);assert float(original['parthenon/time','tlim'])==500
 for n in (32,48,64):
  lines=[];section=''
  for line in raw.splitlines():
   stripped=line.split('#',1)[0].strip()
   if stripped.startswith('<'):section=stripped.strip('<>')
   if '=' in stripped and (section,stripped.split('=',1)[0].strip()) in allowed:
    value=n if section=='parthenon/mesh' else n//2
    line=re.sub(r'(=\s*)[^#]+',lambda m:m[1]+str(value),line)
   lines.append(line)
  text='\n'.join(lines)+'\n';new=parse(text);assert {k for k in original if original[k]!=new[k]}<=allowed
  name=f'{kind}_n{n}.in';(out/name).write_text(text)
  rows.append(dict(kind=kind,n=n,block=n//2,input=name,sha256=hashlib.sha256(text.encode()).hexdigest(),original_sha256=hashlib.sha256(raw.encode()).hexdigest(),target=500,physical_level=8,min_dx=400/n/256,root_blocks_per_axis=2,changed_parameters=['/'.join(k) for k in sorted(allowed)]))
(r/'PLAN.json').write_text(json.dumps(dict(status='controlled_six_case_candidates_GPU_admission_required',cases=rows,original_production_unchanged=True),indent=2)+'\n')
