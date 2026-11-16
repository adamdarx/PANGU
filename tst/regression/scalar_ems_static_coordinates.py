"""Check native 3D constraints for RN and a mapped archived static EMS reference.

This initializes and re-reads native restarts at t=0. It is not an evolution
qualification. The hairy horizon boundary values come from the qualified PG
paper5_n32768 run at 1000M; reconstructing them is not an endpoint prediction.
SciPy is needed only by the independent initial-data fixture.
"""
from pathlib import Path
import argparse
import json
import math

import h5py
import numpy as np

from ems_puncture_fixture import inject
from ems_static_fixture import StaticProfile, constraint_samples
from scalar_ems_coordinates import run


def inspect(path, profile):
    sums = np.zeros(3)
    volume = 0.
    field_error = 0.
    with h5py.File(path) as f:
        assert float(f['Info'].attrs['Time']) == 0
        for block in range(len(f['Levels'])):
            x, y, z = [f['VolumeLocations/'+a][block] for a in 'xyz']
            rho = np.sqrt(x[None, None, :]**2+y[None, :, None]**2+z[:, None, None]**2)
            mask = (rho > .5) & (rho <= 1.25)
            np.testing.assert_array_equal(f['nr.constraint_mask'][block, 0], mask)
            u = f['scalar.state'][block]
            geo = f['nr.z4c'][block]
            F, phi = profile.evaluate(rho)
            field_error = max(field_error, float(np.max(np.abs(u[0]-phi))),
                              float(np.max(np.abs(geo[0]-(rho/F)**4))))
            assert np.max(np.abs(u[1])) == 0, 'Static slicing requires scalar P=0'
            for key in ['scalar.state', 'nr.z4c', 'nr.constraints', 'nr.tmunu', 'ems.gauss']:
                assert np.isfinite(f[key][block]).all(), (key, path)
            cell_volume = float(np.prod([f['Locations/'+a][block, 1]-
                                        f['Locations/'+a][block, 0] for a in 'xyz']))
            volume += int(mask.sum())*cell_volume
            for n, values in enumerate([f['nr.constraints'][block, 1],
                                        f['nr.constraints'][block, 2],
                                        f['ems.gauss'][block, 0]]):
                sums[n] += float(np.sum(values[mask]**2))*cell_volume
        assert volume > 0 and field_error < 1.e-11
        return dict(time=0., volume=volume, physical_initial_field_error=field_error,
                    H_RMS=float(np.sqrt(sums[0]/volume)),
                    M_RMS=float(np.sqrt(sums[1]/volume)),
                    GaussD_RMS=float(np.sqrt(sums[2]/volume)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pangu', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    root, exe, work = args.root.resolve(), args.pangu.resolve(), args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    disc = math.sqrt(1-.9**2)
    profiles = dict(rn=StaticProfile(1+disc, 0.),
                    hair=StaticProfile(1.4548222905052346, .3689996020918125))
    rho = np.geomspace(.014, 200., 1000)
    F, _ = profiles['rn'].evaluate(rho)
    rn_error = float(np.max(np.abs(F/np.sqrt(rho*rho+rho+disc*disc/4)-1)))
    assert rn_error < 1.e-9
    report = dict(status='running', RN_coordinate_error=rn_error, cases={},
                  norm='Coordinate-volume-weighted active cells in 0.5<rho<=1.25; no ghosts',
                  scope='Native initial constraints at t=0 only; no long-time acceptance')

    def save():
        (work/'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    for name, profile in profiles.items():
        case = dict(ADM_mass=profile.adm_mass, horizon_areal_radius=profile.horizon_radius,
                    horizon_isotropic_radius=profile.isotropic_horizon,
                    expected_irreducible_mass=profile.horizon_radius/2,
                    phi_infinity_estimate=profile.phi_infinity_estimate,
                    boundary_data='Specified horizon data, not independently predicted',
                    radial_constraint=constraint_samples(profile), resolutions={})
        report['cases'][name] = case
        assert min(case['radial_constraint']['orders']) > 3.8, (name, case['radial_constraint'])
        save()
        for n in [32, 48, 64]:
            common = ['parthenon/mesh/refinement=none', 'parthenon/mesh/numlevel=1',
                      *[f'parthenon/mesh/nx{a}={n}' for a in [1, 2, 3]],
                      *[f'parthenon/meshblock/nx{a}=16' for a in [1, 2, 3]],
                      *[f'parthenon/mesh/x{a}{side}={value}' for a in [1, 2, 3]
                        for side, value in [('min', -2), ('max', 2)]],
                      'numerical_relativity/horizon_enabled=false',
                      'numerical_relativity/horizon_count=0',
                      'numerical_relativity/constraint_mask_outer_radius=1.25',
                      'problem/constraint_mask_radius=.5', 'parthenon/time/nlim=0']
            seed = work/(name+f'-seed-{n}')
            run(exe, root, seed, common, deck=root/'plugins/scalar_field/examples/ems_cartesian_rn_hold.in')
            checkpoint = work/(name+f'-initial-{n}.rhdf')
            inject(next(seed.glob('*.final.rhdf')), checkpoint, profile)
            output = run(exe, root, work/(name+f'-native-{n}'),
                         ['parthenon/time/nlim=0'], checkpoint=checkpoint)
            case['resolutions'][str(n)] = inspect(output, profile)
            save()
        case['orders'] = {}
        for metric in ['H_RMS', 'GaussD_RMS']:
            values = [case['resolutions'][str(n)][metric] for n in [32, 48, 64]]
            orders = [math.log(values[i]/values[i+1])/math.log([1.5, 4/3][i]) for i in [0, 1]]
            case['orders'][metric] = orders
            save()
            assert min(orders) > 3.0, (name, metric, values, orders)
    report['status'] = 'native_initial_constraints_passed'
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
