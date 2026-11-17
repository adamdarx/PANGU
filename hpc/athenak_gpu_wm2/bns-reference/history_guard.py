"""Reject nonfinite complete history rows while leaving an in-progress row alone."""
from pathlib import Path
import math
def check_history(folder,full=False):
    for p in Path(folder).glob('*.hst'):
        with p.open('rb') as f:
            offset=0 if full else max(0,p.stat().st_size-16384);f.seek(offset);blob=f.read()
        lines=blob.splitlines(keepends=True)
        if offset:lines=lines[1:]
        for line in lines:
            if not line.endswith(b'\n') or not line.strip() or line.lstrip().startswith(b'#'):continue
            row=[float(x) for x in line.split()]
            if not all(math.isfinite(x) for x in row):
                raise RuntimeError('Nonfinite complete history row: %s time=%.17g'%(p.name,row[0]))
