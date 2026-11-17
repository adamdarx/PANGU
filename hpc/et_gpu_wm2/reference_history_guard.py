"""Check completed ET norm rows; AH failure rows and axis profiles are separate."""
from pathlib import Path
import math,re
NAMES=('hydrobasex-rho.tsv','asterx-cons_vector.tsv','z4c-hc.tsv','admbasex-lapse.tsv')
def check_norms(folder,full=False):
    for name in NAMES:
        for p in Path(folder).rglob(name):
            with p.open('rb') as f:
                header=f.readline().decode(errors='replace')
                offset=0 if full else max(0,p.stat().st_size-16384);f.seek(offset);data=f.read()
            columns={int(i)-1:key for i,key in re.findall(r'(\d+):([^\s\\]+)',header)}
            lines=data.splitlines(keepends=True)
            if offset:lines=lines[1:]
            for line in lines:
                if not line.endswith(b'\n') or not line.strip() or line.lstrip().startswith(b'#'):continue
                row=[float(x) for x in line.split()]
                if not all(math.isfinite(x) for x in row):
                    raise RuntimeError('Nonfinite norm row: %s time=%.17g'%(p.name,row[1]))
                if columns and len(row)!=max(columns)+1:
                    raise RuntimeError('Malformed completed norm row: '+p.name)
                for i,key in columns.items():
                    if key.endswith('.volume') and row[i]<=0:
                        raise RuntimeError('Nonpositive diagnostic volume: %s time=%.17g'%(p.name,row[1]))
