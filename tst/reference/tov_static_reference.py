"""Independent static TOV mass/radius reference using log specific enthalpy.

Equations/EOS convention: Einstein Toolkit TOVSolver documentation,
https://einsteintoolkit.org/thornguide/EinsteinInitialData/TOVSolver/documentation.html
This is a static initial-data analysis, not a time evolution of the GRMHD system.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def solve_tov(rhoc=1.28e-3, kappa=100., gamma=2., rtol=1e-12):
    if not (rhoc > 0 and kappa > 0 and 1 < gamma <= 2 and rtol > 0):
        raise ValueError('Require positive density/K/tolerance and 1<Gamma<=2')
    pressure = kappa * rhoc ** gamma
    energy = rhoc + pressure / (gamma - 1)
    hc = np.log1p(gamma * kappa * rhoc ** (gamma - 1) / (gamma - 1))
    start = 1e-6
    y0 = [hc - 2*np.pi*(energy + 3*pressure)*start**2/3,
          4*np.pi*energy*start**3/3, 4*np.pi*rhoc*start**3/3]

    def rhs(radius, state):
        h, mass, _ = state
        rho = ((gamma - 1)*np.expm1(max(h, 0))/(gamma*kappa)) ** (1/(gamma - 1))
        p = kappa * rho ** gamma
        e = rho + p / (gamma - 1)
        metric = 1 - 2*mass/radius
        if metric <= 0:
            raise ValueError('Static-star integration encountered 2m/r>=1')
        return [-(mass + 4*np.pi*radius**3*p)/(radius*(radius - 2*mass)),
                4*np.pi*radius**2*e, 4*np.pi*radius**2*rho/np.sqrt(metric)]

    def surface(radius, state):
        return state[0]
    surface.terminal = True
    surface.direction = -1
    result = solve_ivp(rhs, (start, 50*np.sqrt(kappa/100)), y0,
                       method='DOP853', rtol=rtol, atol=rtol*.01, events=surface)
    if not result.success or len(result.t_events[0]) != 1:
        raise ValueError('Surface not found within the declared integration domain')
    radius = float(result.t_events[0][0])
    _, mass, baryon = map(float, result.y_events[0][0])
    isotropic = (radius - mass + np.sqrt(radius*(radius - 2*mass))) / 2
    return dict(ADM_mass=mass, baryon_mass=baryon, areal_surface_radius=radius,
                isotropic_surface_radius=float(isotropic),
                central_lapse=float(np.sqrt(1 - 2*mass/radius)*np.exp(-hc)),
                method='DOP853 in radius; H=log((e+p)/rho); vacuum surface H=0',
                rhoc=rhoc, kappa=kappa, gamma=gamma, rtol=rtol,
                coordinate_start=start, function_evaluations=result.nfev)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rhoc', type=float, default=1.28e-3)
    parser.add_argument('--kappa', type=float, default=100.)
    parser.add_argument('--gamma', type=float, default=2.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    coarse = solve_tov(args.rhoc, args.kappa, args.gamma, 1e-10)
    fine = solve_tov(args.rhoc, args.kappa, args.gamma, 1e-12)
    keys = ['ADM_mass', 'baryon_mass', 'areal_surface_radius', 'isotropic_surface_radius']
    result = dict(status='independent_static_TOV_reference_computed', reference=fine,
                  tolerance_refinement=coarse,
                  relative_refinement_differences={key: abs(coarse[key]/fine[key]-1) for key in keys},
                  limitation='Tolerance refinement is a consistency check, not a certified error bound. Grid initial-data error also includes atmosphere and profile-table interpolation.',
                  equations_source='https://einsteintoolkit.org/thornguide/EinsteinInitialData/TOVSolver/documentation.html')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
