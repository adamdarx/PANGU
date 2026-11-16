"""Spherical non-PG3+1 checks for the RN scalarization reproduction driver."""
from pathlib import Path
import argparse
import csv
import json
import math

import h5py
import numpy as np

from scalar_ems_coordinates import run


def inspect(path):
    with h5py.File(path) as f:
        for key in ['scalar.state', 'nr.z4c', 'nr.adm', 'nr.tmunu', 'nr.constraints', 'ems.radial']:
            assert np.isfinite(f[key][...]).all(), (key, path)
        assert len(f['Levels']) == 1
        radial = f['ems.radial'][0, :, 0, 0]
        r, R, E, Q, expansion, mass, jac = radial
        z = f['nr.z4c'][0, :, 0, 0]
        state = f['scalar.state'][0, :, 0, 0]
        assert np.min(z[0]) > 0 and np.min(z[18]) > 0
        horizon = np.flatnonzero((expansion[:-1] <= 0) & (expansion[1:] > 0))
        inner = r[horizon[-1]+1]*1.1 if len(horizon) else 0
        mask = (r > inner) & (r < 50)
        H = f['nr.constraints'][0, 1, 0, 0]
        weight = r*r*jac
        return dict(time=float(f['Info'].attrs['Time']), cycle=int(f['Info'].attrs['NCycle']),
                    radial_points=len(r), scalar_max=float(np.max(np.abs(state))),
                    H_exterior_RMS=float(np.sqrt(np.sum(weight[mask]*H[mask]**2)/np.sum(weight[mask]))),
                    analytic_charge_error=float(np.max(np.abs(Q-Q[0]))),
                    coordinate_volume_weighting='r^2 dr, excluding inside the outer AH and ghosts')


def differences(left, right):
    with h5py.File(left) as a, h5py.File(right) as b:
        assert float(a['Info'].attrs['Time']) == float(b['Info'].attrs['Time'])
        return {key: float(np.max(np.abs(a[key][...]-b[key][...])))
                for key in ['scalar.state', 'nr.z4c', 'nr.adm', 'nr.tmunu']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    root, exe, work = args.root.resolve(), args.pangu.resolve(), args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    deck = root/'plugins/scalar_field/examples/ems_spherical_rn_scalarization.in'
    report = dict(status='running', cases={}, scope='One radial mesh, native BSSN-limit Einstein point RHS, chi=psi^-4; bounded checks precede nonlinear reproduction')

    def save():
        (work/'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    common = ['parthenon/time/tlim=.5', 'parthenon/time/ncycle_out=500']
    flat = run(exe, root, work/'minkowski', common+[
        'plugin/scalar_field/initial_data=minkowski', 'plugin/scalar_field/outer_radius=20',
        'plugin/scalar_field/radial_map_a=0'], deck=deck)
    with h5py.File(flat) as f:
        assert np.max(np.abs(f['scalar.state'][...])) == 0
        assert np.max(np.abs(f['nr.constraints'][...])) == 0
    report['cases']['minkowski'] = inspect(flat)
    save()
    rn_errors = []
    for n in [128, 256]:
        folder = work/f'rn-{n}'
        out = run(exe, root, folder, common+[
            'plugin/scalar_field/initial_data=reissner_nordstrom',
            f'parthenon/mesh/nx1={n}', f'parthenon/meshblock/nx1={n}'], deck=deck)
        initial = next(folder.glob('*.00000.phdf'))
        row = dict(initial=inspect(initial), final=inspect(out))
        assert row['final']['scalar_max'] == 0
        with (folder/'ems_spherical_history.csv').open() as f:
            history = list(csv.DictReader(f))
        exact_R = 2+math.sqrt(3)
        initial_R = float(history[0]['horizon_areal_radius'])
        final_R = float(history[-1]['horizon_areal_radius'])
        assert abs(initial_R/exact_R-1) < 2.e-4
        assert abs(final_R/exact_R-1) < 2.e-4
        row['AH_areal_relative_error'] = abs(final_R/exact_R-1)
        rn_errors.append(row['initial']['H_exterior_RMS'])
        report['cases'][f'rn-{n}'] = row
        save()
    order = math.log(rn_errors[0]/rn_errors[1], 2)
    assert order > 3.2, (rn_errors, order)
    report['RN_initial_H_order'] = order
    small = common+['parthenon/mesh/nx1=128', 'parthenon/meshblock/nx1=128',
                    'parthenon/output2/dt=.25']
    phase0 = run(exe, root, work/'phase0', small+['plugin/scalar_field/phase=0'], deck=deck)
    phase = math.pi/3
    rotated = run(exe, root, work/'phase-rotated', small+[
        'plugin/scalar_field/phase='+str(phase)], deck=deck)
    with h5py.File(phase0) as a, h5py.File(rotated) as b:
        x, y = a['scalar.state'][...], b['scalar.state'][...]
        expected = x.copy()
        expected[:, :2] = math.cos(phase)*x[:, :2]
        expected[:, 2:] = math.sin(phase)*x[:, :2]
        error = float(np.max(np.abs(y-expected)))
        assert error < 5.e-15, error
        geometry_error = float(np.max(np.abs(a['nr.z4c'][...]-b['nr.z4c'][...])))
        assert geometry_error < 5.e-12, geometry_error
    report['U1_rotation'] = dict(scalar_max_difference=error, geometry_max_difference=geometry_error)
    candidates = [p for p in (work/'phase0').glob('*.rhdf') if '.final.' not in p.name]
    candidates = [p for p in candidates if p.name.endswith('.00001.rhdf')]
    assert len(candidates) == 1, candidates
    resumed = run(exe, root, work/'restart', [], checkpoint=candidates[0])
    delta = differences(phase0, resumed)
    assert max(delta.values()) < 5.e-12, delta
    report['restart'] = delta
    pg = run(exe, root, work/'pg-rn', [
        'plugin/scalar_field/initial_data=reissner_nordstrom',
        'parthenon/mesh/nx1=512', 'parthenon/meshblock/nx1=512',
        'parthenon/mesh/x1max=12', 'parthenon/time/tlim=.01'],
        deck=root/'plugins/scalar_field/examples/ems_coupled_rn.in')
    with h5py.File(pg) as f:
        assert 'nr.z4c' not in f
        assert np.max(np.abs(f['scalar.state'][:, :2])) == 0
    report.update(status='bounded_spherical_checks_passed', physical_reproduction='Long nonlinear evolution and comparison with paper Fig6 remain separate')
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
