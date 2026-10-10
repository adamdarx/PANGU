#!/usr/bin/env python3
"""MKS FM seed/CT, MPI decomposition and restart regression (small test meshes)."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--executable', required=True, type=Path)
    ap.add_argument('--cases', required=True, type=Path)
    ap.add_argument('--workdir', required=True, type=Path)
    ap.add_argument('--mpirun', default='mpirun')
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    exe = str(args.executable.resolve())
    common = ['parthenon/mesh/nx1=32', 'parthenon/mesh/nx2=24', 'parthenon/mesh/nx3=8',
              'parthenon/meshblock/nx1=16', 'parthenon/meshblock/nx2=24',
              'parthenon/meshblock/nx3=8', 'parthenon/time/tlim=0.0002',
              'parthenon/time/dt_force=0.0001', 'parthenon/time/nlim=-1',
              'parthenon/output2/variables=mhd.prim,mhd.b_cell,mhd.divb,electrons.prim,electrons.cons']

    def run(name, commands, ranks=2, success=True):
        directory = args.workdir / name
        directory.mkdir(exist_ok=True)
        p = subprocess.run([args.mpirun, '-np', str(ranks), exe, *commands], cwd=directory,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (directory / 'run.log').write_text(p.stdout)
        if success and p.returncode:
            raise RuntimeError(f'{name} exit={p.returncode}: {p.stdout[-2000:]}')
        if not success:
            assert p.returncode != 0, f'{name} unexpectedly accepted'
        return directory, p.stdout

    def read(directory, initial=False):
        label = '00000' if initial else 'final'
        with h5py.File(directory / f'gr_torus_sane.prim.{label}.phdf') as f:
            arrays = {k: f[k][...] for k in ['mhd.prim', 'mhd.b_cell', 'mhd.divb',
                                            'electrons.prim', 'electrons.cons']}
            assert all(np.isfinite(x).all() for x in arrays.values())
            assert arrays['electrons.prim'].shape[1] == 7
            assert np.max(np.abs(arrays['mhd.divb'])) < 1e-11
            return arrays

    report = {}
    for case in sorted(args.cases.glob('*.in')):
        path = str(case.resolve())
        directory, log = run(case.stem, ['-i', path, *common])
        data = read(directory)
        beta = float(re.search(r' beta=([\d.eE+\-]+)', log).group(1))
        assert abs(beta - 100) < 1e-8
        report[case.stem] = {'beta': beta, 'max_divB': float(np.max(np.abs(data['mhd.divb'])))}

    sane = read(args.workdir / 'sane_a09375', True)
    mad = read(args.workdir / 'mad_ap09375', True)
    assert not np.allclose(sane['mhd.b_cell'], mad['mhd.b_cell'], atol=1e-12)
    retro = read(args.workdir / 'mad_am09375', True)['mhd.prim']
    dense = retro[:, 0] > 0.1
    assert np.any(dense) and np.all(retro[:, 3][dense] > 0), 'retrograde disk must retain positive rotation'
    reference = str((args.cases / 'sane_a09375.in').resolve())
    legacy, _ = run('legacy', ['-i', reference, *common, 'problem/magnetic_topology=sane'])
    # Remove the selector entirely to exercise old input behavior.
    legacy_input = args.workdir / 'legacy.in'
    legacy_input.write_text('\n'.join(line for line in Path(reference).read_text().splitlines()
                                      if not line.startswith('magnetic_topology')) + '\n')
    implicit, _ = run('implicit', ['-i', str(legacy_input.resolve()), *common])
    for k, v in read(legacy).items():
        assert np.array_equal(v, read(implicit)[k]), k

    mad_input = str((args.cases / 'mad_ap09375.in').resolve())
    default_input = args.workdir / 'mad_defaults.in'
    default_input.write_text('\n'.join(line for line in Path(mad_input).read_text().splitlines()
                                       if not line.startswith(('potential_r_pow', 'potential_falloff',
                                                               'potential_cutoff', 'potential_rho_pow'))) + '\n')
    defaults, _ = run('mad_defaults', ['-i', str(default_input.resolve()), *common])
    for k, v in read(args.workdir / 'mad_ap09375').items():
        assert np.array_equal(v, read(defaults)[k]), k
    split, _ = run('split', ['-i', mad_input, *common, 'parthenon/time/tlim=0.0001'])
    rst = str((split / 'gr_torus_sane.restart.final.rhdf').resolve())
    preflight, _ = run('preflight', ['-r', rst, '-m', '2'])
    assert not list(preflight.glob('*.rhdf')), 'preflight wrote output'
    resumed, _ = run('resumed', ['-r', rst, 'parthenon/time/tlim=0.0002'])
    for k, v in read(args.workdir / 'mad_ap09375').items():
        assert np.allclose(v, read(resumed)[k], rtol=1e-12, atol=1e-13), k
    one, _ = run('one_rank', ['-i', mad_input, *common], ranks=1)
    for k, v in read(args.workdir / 'mad_ap09375', True).items():
        assert np.allclose(v, read(one, True)[k], rtol=1e-12, atol=1e-13), k
    run('bad_topology', ['-i', reference, *common, 'problem/magnetic_topology=invalid'], success=False)
    run('stale_sane_controls', ['-i', reference, *common, 'problem/magnetic_topology=mad'], success=False)
    print(json.dumps(report, indent=2))
    (args.workdir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: seven seeds, finite states, six electrons, CT, beta, SANE compatibility, MPI and restart')


if __name__ == '__main__':
    main()
