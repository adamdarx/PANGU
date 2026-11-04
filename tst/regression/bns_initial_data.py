#!/usr/bin/env python3
"""Independent TOV limit and Cartesian ADM constraints for BNS table producer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np


def load():
    path = Path(__file__).resolve().parents[2]/'plugins/compact_objects/bns_data.py'
    spec = importlib.util.spec_from_file_location('bns_data',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', required=True, type=Path)
    a = p.parse_args()
    a.workdir.mkdir(parents=True,exist_ok=False)
    m = load()
    profile = m.tov()
    report = dict(status='running', isolated=[], binary=[],
                  sources=['https://arxiv.org/abs/gr-qc/9904041',
                           'https://doi.org/10.12942/lrr-2000-5'],
                  distinct_from_reference='at rest, conformally flat, equal cold EOS stars; not boosted York data')
    def save():
        (a.workdir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        # Analytic spherical source (1-r^2)^2 on r<=1 fixes the Green
        # function's sign, 2*pi normalization, origin and exterior matching.
        radial=np.linspace(0,2,801)
        source=np.maximum(1-radial**2,0)**2
        spherical,moment=m.green_coefficients(radial,source[:,None])
        assert abs(spherical[0,0]-np.pi/3)<1e-9
        assert abs(moment[0]-16*np.pi/105)<1e-9
        np.testing.assert_allclose(spherical[radial>=1,0],moment[0]/radial[radial>=1],rtol=0,atol=1e-10)
        report['analytic_poisson_control']='passed'
        for kwargs in [dict(separation=1),dict(lmax=130,angular_points=384)]:
            try:m.solve(profile,**kwargs)
            except ValueError:pass
            else:raise AssertionError('invalid initial data accepted: '+str(kwargs))
        # Exact TOV and isotropic Schwarzschild exterior are independent
        # analytic/spherical controls for the 2D integral constraint solver.
        for step in [.05,.025,.0125]:
            r,c,mm,s = m.solve(profile,step=step,lmax=0,angular_points=32,single=True)
            ref = np.where(r >= profile['isotropic_radius'],
                           1+profile['mass']/(2*np.maximum(r,1e-20)),
                           m.cubic(profile['psi'],np.minimum(r/profile['step'],len(profile['psi'])-1)))
            error = float(np.max(np.abs(ref-1-c[:,0])))
            mass_error = abs(s['adm_mass']/profile['mass']-1)
            report['isolated'].append(dict(step=step,psi_max_abs=error,mass_relative_error=mass_error))
            assert error < 5e-6 and mass_error < 2e-5
        # Fixed, nonsymmetric Cartesian samples and a 4th-order 3D Laplacian
        # differ from the producer's radial/angular integral transform.
        rng = np.random.default_rng(711)
        points = rng.uniform([-6,-6,6],[6,6,18],(2048,3))
        distance = np.sqrt(points[:,0]**2+points[:,1]**2+(points[:,2]-12)**2)
        points = points[distance < profile['isotropic_radius']-1.5]
        x,y,z = points.T
        for step,lmax,nang in [(.025,48,128),(.025,96,256),(.0125,128,384)]:
            r,c,mm,s = m.solve(profile,step=step,lmax=lmax,angular_points=nang)
            psi = m.evaluate(r,c,mm,x,y,z)
            energy = m.free_source(profile,np.hypot(x,y),z,24)/psi**8
            h = .25
            lap = np.zeros_like(psi)
            for axis in range(3):
                for offset,w in [(-2,-1),(-1,16),(0,-30),(1,16),(2,-1)]:
                    shifted = points.copy()
                    shifted[:,axis] += offset*h
                    lap += w*m.evaluate(r,c,mm,*shifted.T)/(12*h*h)
            ham = -8*psi**-5*lap-16*np.pi*energy
            norm = float(np.sqrt(np.mean(ham**2))/(16*np.pi*energy.max()))
            record = dict(lmax=lmax,step=step,cartesian_hamiltonian_bulk_l2_scaled=norm,
                          adm_mass=s['adm_mass'],baryon_mass=s['baryon_mass'],
                          iteration_residual=s['fixed_point_abs_residual'])
            report['binary'].append(record)
            # Translation/axis/reflection errors can otherwise hide behind
            # solver self-residuals. Check physical parity at all samples.
            np.testing.assert_allclose(psi,m.evaluate(r,c,mm,-x,y,-z),rtol=0,atol=2e-14)
            assert np.isfinite(c).all() and psi.min()>1
            if lmax == 128:
                assert norm < 1e-4
                m.write(a.workdir/'bns.dat',profile,24,r,c,mm,s)
        errors = [b['cartesian_hamiltonian_bulk_l2_scaled'] for b in report['binary']]
        assert errors[2] < errors[1] < errors[0]
        assert errors[0]/errors[-1] > 4
        report['status'] = 'passed_initial_data_only'
        report['production_qualified'] = False
        save()
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
