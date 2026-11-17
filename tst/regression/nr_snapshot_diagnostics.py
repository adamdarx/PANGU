"""Recompute a strict Schwarzschild AH from a restart with no evolution steps."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import tempfile

import h5py
import numpy as np


def launch(exe,folder,arguments):
    folder.mkdir()
    with (folder/'run.log').open('wb') as log:
        subprocess.run([str(exe),*arguments],cwd=folder,stdout=log,
                       stderr=subprocess.STDOUT,check=True,timeout=600)
    text=(folder/'run.log').read_text()
    if 'Driver completed.' not in text:raise RuntimeError('Driver completion missing')
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--workdir',required=True,type=Path)
    a=p.parse_args();exe=a.executable.resolve();parent=a.workdir.resolve()
    parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='run-',dir=parent))
    overrides=['parthenon/mesh/nghost=4','parthenon/time/integrator=rk4',
        'numerical_relativity/rk4_variant=classical','numerical_relativity/cfl=1e-8',
        'numerical_relativity/finite_difference_order=4',
        'numerical_relativity/weyl_enabled=false','numerical_relativity/waveform_enabled=false',
        'numerical_relativity/horizon_enabled=true','numerical_relativity/horizon_count=2',
        'parthenon/time/nlim=1','parthenon/output3/file_type=rst','parthenon/output3/dt=1']
    launch(exe,work/'source',['-i',str(a.input.resolve()),*overrides])
    checkpoint=next((work/'source').glob('*.final.rhdf'))
    with h5py.File(checkpoint) as saved:
        cached=saved['Params'].attrs['numerical_relativity/horizon_last_search_times']
        assert len(cached)==2 and cached[0]==saved['Info'].attrs['Time']
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source=next((work/'source').glob('*.final.phdf'))
    launch(exe,work/'ordinary-restart',['-r',str(checkpoint),
        'pangu/diagnostics_only=false','parthenon/time/nlim=1',
        'parthenon/time/tlim=20'])
    ordinary=next((work/'ordinary-restart').glob('*.final.rhdf'))
    with h5py.File(checkpoint) as left,h5py.File(ordinary) as right:
        for name in ['horizon_radii','horizon_found','horizon_last_search_times',
                     'horizon_tracker_indices','horizon_centers']:
            key='numerical_relativity/'+name
            assert np.array_equal(left['Params'].attrs[key],right['Params'].attrs[key]),key
    assert not list((work/'ordinary-restart').rglob('nr_horizon*.csv'))
    text=launch(exe,work/'diagnostics',['-r',str(checkpoint),'pangu/diagnostics_only=true',
        'parthenon/time/nlim=100','parthenon/time/tlim=20',
        'numerical_relativity/horizon_enabled=true','numerical_relativity/horizon_count=1',
        'numerical_relativity/horizon_start_time_0=0','numerical_relativity/horizon_stop_time_0=1e30',
        'numerical_relativity/horizon_tracker_0=-1','numerical_relativity/horizon_ntheta=40',
        'numerical_relativity/horizon_lmax=12','numerical_relativity/horizon_iterations=500',
        'numerical_relativity/horizon_convergence=mass_and_expansion',
        'numerical_relativity/horizon_expansion_tolerance=1e-4',
        'numerical_relativity/horizon_output_grid=false',
        'numerical_relativity/diagnostic_directory=diagnostics'])
    assert 'evolved_steps=0' in text
    final=next((work/'diagnostics').glob('*.final.phdf'))
    with h5py.File(source) as left,h5py.File(final) as right:
        for name in ['Time','NCycle']:
            assert left['Info'].attrs[name]==right['Info'].attrs[name],name
        assert int(left['Info'].attrs['NCycle'])==1 and float(left['Info'].attrs['Time'])>0
        x,y=left['nr.z4c'][...],right['nr.z4c'][...]
        delta=float(np.max(np.abs(x-y)))
        assert delta<5e-14*max(1,float(np.max(np.abs(x)))),delta
        stamp=float(right['Info'].attrs['Time']);cycle=int(right['Info'].attrs['NCycle'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    with (work/'diagnostics/diagnostics/nr_horizon.csv').open() as stream:
        rows=list(csv.DictReader(row for row in stream if not row.startswith('#')))
    assert len(rows)==1
    row={k:float(v) for k,v in rows[0].items()}
    assert row['found']==1 and row['scaled_rms_expansion']<=1e-4,row
    assert abs(row['irreducible_mass']-1)<1e-3,row
    report=dict(status='snapshot_diagnostics_passed',workdir=str(work),time=stamp,cycle=cycle,evolved_steps=0,
                source_checkpoint_sha256=digest,source_checkpoint_unchanged=True,
                cached_diagnostic_cadence_and_geometry_reset=True,
                ordinary_restart_diagnostic_cache_preserved=True,
                independent_NR_max_absolute_change=delta,horizon=row,
                scope='One strict analytic Schwarzschild restart check; archived binary snapshots require their own measurements.')
    (work/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
