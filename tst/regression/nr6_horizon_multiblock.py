#!/usr/bin/env python3
"""CUDA regression: Schwarzschild horizon must not depend on block decomposition.

One tiny RK4 step isolates the diagnostic from secular evolution errors. This
does not establish long-time mass conservation. An existing output directory
is never overwritten. Run with the NR CUDA executable and the NR-6 stationary
diagnostic input; the overrides below disable unrelated waveform work.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path,
                        help='Also check four-rank ownership on non-binary block faces')
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='run-', dir=args.workdir.resolve()))
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    overrides = [
        'parthenon/mesh/nghost=4', 'parthenon/time/nlim=1',
        'parthenon/time/integrator=rk4', 'numerical_relativity/rk4_variant=classical',
        'numerical_relativity/cfl=1e-8', 'numerical_relativity/finite_difference_order=4',
        'numerical_relativity/weyl_enabled=false',
        'numerical_relativity/waveform_enabled=false',
        'numerical_relativity/horizon_enabled=true',
        'numerical_relativity/horizon_start_time_0=0',
        'numerical_relativity/horizon_stop_time_0=1e30',
        'numerical_relativity/horizon_output_grid=false',
        'numerical_relativity/horizon_ntheta=40', 'numerical_relativity/horizon_lmax=12',
        'numerical_relativity/horizon_iterations=300',
        'numerical_relativity/horizon_convergence=mass_and_expansion',
        'numerical_relativity/horizon_expansion_tolerance=1e-4',
        'numerical_relativity/diagnostic_directory=diagnostics',
    ]
    results = []
    for block in (64, 16):
        folder = workdir / f'b{block}'
        folder.mkdir()
        cmd = [str(args.executable.resolve()), '-i', str(args.input.resolve())]
        cmd += overrides + [f'parthenon/meshblock/nx{axis}={block}' for axis in (1, 2, 3)]
        (folder / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        with (folder / 'run.log').open('w') as log:
            subprocess.run(cmd, cwd=folder, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
        with (folder / 'diagnostics/nr_horizon.csv').open() as stream:
            rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
        if len(rows) != 1:
            raise RuntimeError(f'Expected exactly one horizon measurement: {folder}')
        row = {key: float(value) for key, value in rows[0].items()}
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError(f'Nonfinite measurement: {folder}')
        if row['found'] != 1 or row['scaled_rms_expansion'] > 1e-4:
            raise RuntimeError(f'Unconverged horizon: {folder}: {row}')
        if abs(row['irreducible_mass'] - 1.0) > 1e-3:
            raise RuntimeError(f'Wrong Schwarzschild mass: {folder}: {row}')
        results.append({'block': block, **row})
    for key in ('irreducible_mass', 'rms_expansion', 'radius'):
        if abs(results[0][key] - results[1][key]) > 1e-10:
            raise RuntimeError(f'Decomposition-dependent horizon {key}: {results}')
    if args.mpiexec:
        folder = workdir / 'mpi-nonbinary'
        folder.mkdir()
        command = [str(args.mpiexec.resolve()), '-n', '4', str(args.executable.resolve()),
                   '-i', str(args.input.resolve()), *overrides]
        for axis in (1, 2, 3):
            command += [f'parthenon/mesh/nx{axis}=64',
                        f'parthenon/meshblock/nx{axis}=16',
                        f'parthenon/mesh/x{axis}min=-2.0833333333333335',
                        f'parthenon/mesh/x{axis}max=2.0833333333333335']
        env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,smcuda,tcp',
                   OMPI_MCA_opal_cuda_support='true')
        (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        with (folder / 'run.log').open('w') as log:
            subprocess.run(command, cwd=folder, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=180)
        with (folder / 'diagnostics/nr_horizon.csv').open() as stream:
            rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
        if len(rows) != 1:
            raise RuntimeError(f'Expected one MPI horizon measurement: {folder}')
        row = {key: float(value) for key, value in rows[0].items()}
        if (not all(math.isfinite(value) for value in row.values()) or row['found'] != 1 or
                row['scaled_rms_expansion'] > 1e-4 or
                abs(row['irreducible_mass'] - 1.0) > 1e-3):
            raise RuntimeError(f'Non-binary MPI ownership regression: {row}')
        results.append({'block': 16, 'ranks': 4, 'nonbinary_faces': True, **row})
    (workdir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print('NR-6 horizon multiblock CUDA regression: PASS')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
