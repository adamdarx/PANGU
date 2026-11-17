"""Derive matched Cartesian BBH resolutions from an existing frozen input.

Only the root and block cell counts change. Physical AMR regions, refinement
levels, equations, initial data, CFL and output schedules are retained.
This prepares candidates; it does not approve resources or submit evolution.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re


def parameters(text):
    result={};block=''
    for line in text.splitlines():
        line=line.split('#',1)[0].strip()
        if line.startswith('<'):block=line.strip('<>')
        elif '=' in line:
            key,value=line.split('=',1);result[block,key.strip()]=value.strip()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--resolutions',default='32,48,64')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=args.input.read_text();original=parameters(raw)
    roots=[int(original['parthenon/mesh','nx'+str(a)]) for a in [1,2,3]]
    blocks=[int(original['parthenon/meshblock','nx'+str(a)]) for a in [1,2,3]]
    if len(set(roots))!=1 or len(set(blocks))!=1 or roots[0]%blocks[0]:
        raise ValueError('Need an isotropic root/block layout with an integer block ratio')
    ratio=roots[0]//blocks[0];nghost=int(original['parthenon/mesh','nghost'])
    resolutions=[int(v) for v in args.resolutions.split(',')]
    if sorted(set(resolutions))!=resolutions or resolutions[0]!=roots[0]:
        raise ValueError('Use strictly increasing candidates starting with the archived resolution')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    plan=dict(status='candidates_only_resource_admission_required',source_input=str(args.input.resolve()),
              source_input_sha256=hashlib.sha256(raw.encode()).hexdigest(),root_blocks_per_axis=ratio,
              physics_changes=[],cases=[])
    allowed={(s,'nx'+str(a)) for s in ['parthenon/mesh','parthenon/meshblock'] for a in [1,2,3]}
    finest_level=int(original['parthenon/mesh','numlevel'])-1
    extent=float(original['parthenon/mesh','x1max'])-float(original['parthenon/mesh','x1min'])
    for n in resolutions:
        if n%ratio:raise ValueError('Root resolution must preserve the original root block count')
        block_cells=n//ratio
        if block_cells%2 or block_cells//2<nghost:
            raise ValueError('Need even blocks with a coarse interior large enough for the stencil')
        lines=[];section=''
        for line in raw.splitlines():
            stripped=line.split('#',1)[0].strip()
            if stripped.startswith('<'):section=stripped.strip('<>')
            if '=' in stripped:
                key=stripped.split('=',1)[0].strip()
                if (section,key) in allowed:
                    value=n if section=='parthenon/mesh' else block_cells
                    if original[section,key]!=str(value):
                        line=re.sub(r'(=\s*)[^#]+',lambda m:m.group(1)+str(value)+' ',line).rstrip()
            lines.append(line)
        text='\n'.join(lines)+'\n';new=parameters(text)
        changed={key for key in original if new[key]!=original[key]}
        assert changed== (set() if n==roots[0] else allowed)
        path=output/('quasicircular_n'+str(n)+'_pangu.in');path.write_text(text)
        dx=extent/n
        plan['cases'].append(dict(n=n,block=block_cells,input=path.name,
             input_sha256=hashlib.sha256(text.encode()).hexdigest(),root_dx=dx,
             requested_finest_level=finest_level,requested_min_dx=dx/2**finest_level,
             wavezone_dx=dx/2**int(original['numerical_relativity','amr_radius_0_refinement_level']),
             changed_keys=['/'.join(key) for key in sorted(changed)],
             resource_status='not_admitted',target=float(original['parthenon/time','tlim'])))
    assert (output/plan['cases'][0]['input']).read_text()==raw
    (output/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
