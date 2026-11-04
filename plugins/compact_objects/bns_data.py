#!/usr/bin/env python3
"""Constraint-solved, equal-star, momentarily static head-on initial data.

G=c=M_sun=1. This is *not* the boosted York data of Miller et al. (1999).
Free data: flat conformal metric, K_ij=0, cold Gamma=2 fluid at rest, and
Ebar=sum(psi_TOV**8 * E_TOV) in two non-overlapping isotropic TOV supports.
Solve Delta psi=-2*pi*Ebar*psi**(-3) on all of R^3 using the axisymmetric
Legendre Green function. Recover E=Ebar/psi**8 and rho+K*rho**2=E afterwards.
No finite outer-boundary condition, matter boost, or equilibrium claim is made.
Only even multipoles are retained (identical stars at +/- separation/2).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

# Small dense transforms must not create a laptop-wide BLAS thread pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.interpolate import PchipInterpolator
from scipy.special import eval_legendre


def cumulative_simpson(values, x, initial=0):
    """Uniform-grid quadratic cumulative quadrature (SciPy <1.12 compatible)."""
    h = x[1]-x[0]
    f = np.asarray(values)
    increments = np.empty(len(f)-1)
    pairs = (len(f)-1)//2
    lo, mid, hi = f[:2*pairs:2], f[1:2*pairs+1:2], f[2:2*pairs+2:2]
    increments[:2*pairs:2] = h*(5*lo+8*mid-hi)/12
    increments[1:2*pairs:2] = h*(-lo+8*mid+5*hi)/12
    if len(increments) % 2:
        increments[-1] = h*(-f[-3]+8*f[-2]+5*f[-1])/12
    return np.r_[initial, initial+np.cumsum(increments)]


def tov(rhoc=1.28e-3, kappa=100., step=.005):
    """Independent enthalpy TOV integration, matched to Schwarzschild exterior."""
    hc = np.log1p(2*kappa*rhoc)
    pc = kappa*rhoc**2
    ec = rhoc+pc
    eps = 1.e-6

    def rhs(r, y):
        h, mass, w = y
        rho = np.expm1(max(h, 0.))/(2*kappa)
        pressure = kappa*rho*rho
        f = 1-2*mass/r
        if f <= 0:
            raise ValueError("TOV integration crossed a trapped surface")
        return [-(mass+4*np.pi*r**3*pressure)/(r*r*f),
                4*np.pi*r*r*(rho+pressure), (1/np.sqrt(f)-1)/r]

    def surface(r, y):
        return y[0]
    surface.terminal = True
    surface.direction = -1
    sol = solve_ivp(rhs, (eps, 100.),
                    [hc-2*np.pi*(ec/3+pc)*eps**2, 4*np.pi*ec*eps**3/3,
                     2*np.pi*ec*eps**2/3], events=surface,
                    rtol=2e-12, atol=2e-14, max_step=.02, dense_output=True)
    if not sol.success or not len(sol.t_events[0]):
        raise RuntimeError("TOV surface not found")
    radius = float(sol.t[-1])
    mass, ws = sol.y[1:, -1]
    rs = .5*(radius-mass+np.sqrt(radius*(radius-2*mass)))
    r = np.linspace(eps, radius, 10001)
    h, m, w = sol.sol(r)
    iso = rs*r/radius*np.exp(w-ws)
    rho = np.maximum(np.expm1(h)/(2*kappa), 0)
    psi = np.sqrt(r/iso)
    # A uniform source table is also consumed by the GPU plugin. Both the
    # elliptic solver and importer use the same cubic interpolation below.
    n = int(np.ceil(rs/step))+2
    ri = np.arange(n)*step
    source = PchipInterpolator(np.r_[0., iso],
                              np.r_[psi[0]**8*ec, psi**8*(rho+kappa*rho**2)],
                              extrapolate=False)(np.minimum(ri, rs))
    source[ri >= rs] = 0.
    psiref = np.where(ri >= rs, 1+mass/(2*np.maximum(ri, 1e-30)),
                      PchipInterpolator(np.r_[0., iso], np.r_[psi[0], psi],
                                        extrapolate=False)(np.minimum(ri, rs)))
    baryon = 4*np.pi*simpson(rho*r*r/np.sqrt(1-2*m/r), x=r)
    return dict(rhoc=rhoc, kappa=kappa, gamma=2., mass=float(mass),
                baryon_mass=float(baryon), areal_radius=radius,
                isotropic_radius=float(rs), step=step,
                source=source, psi=psiref)


def cubic(table, coordinate):
    """Uniform, four-point Lagrange interpolation, identical to GPU reader."""
    c = np.asarray(coordinate)
    left = np.clip(np.floor(c).astype(int)-1, 0, len(table)-4)
    t = c-left
    return (-((t-1)*(t-2)*(t-3))/6*table[left]
            +t*(t-2)*(t-3)/2*table[left+1]
            -t*(t-1)*(t-3)/2*table[left+2]
            +t*(t-1)*(t-2)/6*table[left+3])


def free_source(profile, cylindrical_radius, z, separation, single=False):
    result = np.zeros(np.broadcast_shapes(np.shape(cylindrical_radius), np.shape(z)))
    for center in ([0.] if single else [-separation/2, separation/2]):
        distance = np.hypot(cylindrical_radius, z-center)
        result += np.where(distance < profile['isotropic_radius'],
                           np.maximum(cubic(profile['source'],
                                            np.minimum(distance/profile['step'],
                                                       len(profile['source'])-1)), 0), 0)
    return result


def green_coefficients(r, source_l):
    """Legendre solution, radial integrals with fourth-order Simpson quadrature."""
    coeff = np.zeros_like(source_l)
    moments = []
    for j in range(source_l.shape[1]):
        ell = 2*j
        # Products stay well conditioned: compact stellar support is bounded
        # away from the origin for binaries; the l>0 isolated source is zeroed.
        safe = np.where(r > 0, r, 1.)
        inside = cumulative_simpson(source_l[:, j]*r**(ell+2), x=r, initial=0)
        integrand = source_l[:, j]*safe**(1-ell)
        integrand[0] = 0.
        outside = cumulative_simpson(integrand[::-1], x=-r[::-1], initial=0)[::-1]
        c = 2*np.pi/(2*ell+1)
        coeff[:, j] = c*(inside/safe**(ell+1)+r**ell*outside)
        moments.append(c*inside[-1])
    coeff[0, :] = 0
    coeff[0, 0] = 2*np.pi*simpson(source_l[:, 0]*r, x=r)
    return coeff, np.asarray(moments)


def solve(profile, separation=24., step=.025, lmax=48, angular_points=128,
          tolerance=2e-11, single=False):
    if lmax < 0 or lmax > 128 or lmax % 2 or angular_points < 2*lmax+1:
        raise ValueError("use even lmax in [0,128] and >=2*lmax+1 angular points")
    if not single and separation <= 2*profile['isotropic_radius']:
        raise ValueError("initial stellar supports overlap")
    end = (0 if single else separation/2)+profile['isotropic_radius']+2*step
    n = int(np.ceil(end/step))+1
    r = np.arange(n)*step
    mu, weight = np.polynomial.legendre.leggauss(angular_points)
    polys = np.stack([eval_legendre(l, mu) for l in range(0, lmax+1, 2)], axis=1)
    transform = polys.T*weight[None, :]*np.arange(1, 2*lmax+2, 4)[:, None]/2
    ebar = free_source(profile, r[:, None]*np.sqrt(1-mu*mu),
                       r[:, None]*mu, separation, single)
    psi = np.ones_like(ebar)
    history = []
    for iteration in range(200):
        density_l = (ebar/psi**3) @ transform.T
        if single:
            density_l[:, 1:] = 0.
        coeff, moments = green_coefficients(r, density_l)
        candidate = 1+coeff@polys.T
        error = float(np.max(np.abs(candidate-psi)))
        history.append(error)
        if error < tolerance:
            psi = candidate
            break
        psi = .5*(psi+candidate)
    else:
        raise RuntimeError(f"Hamiltonian solve failed: {history[-1]}")
    physical_energy = ebar/psi**8
    rho = 2*physical_energy/(1+np.sqrt(1+4*profile['kappa']*physical_energy))
    baryon = 2*np.pi*simpson(r*r*((rho*psi**6)@weight), x=r)
    report = dict(model='time_symmetric_conformal_tov_pair', single=single,
                  separation=0 if single else separation, radial_step=step,
                  lmax=lmax, angular_points=angular_points, iterations=len(history),
                  fixed_point_abs_residual=history[-1],
                  adm_mass=float(2*moments[0]), baryon_mass=float(baryon),
                  psi_min=float(psi.min()), psi_max=float(psi.max()),
                  rho_max=float(rho.max()), momentum_constraint='analytically zero: Kij=0, v=0',
                  seed={k:v for k,v in profile.items() if k not in ['source', 'psi']},
                  initialization_only=True, production_qualified=False)
    return r, coeff, moments, report


def evaluate(r, coeff, moments, x, y, z):
    radius = np.sqrt(x*x+y*y+z*z)
    mu = np.divide(z, radius, out=np.zeros_like(radius), where=radius > 0)
    psi = np.ones_like(radius)
    for j in range(coeff.shape[1]):
        ell = 2*j
        radial = np.where(radius <= r[-1],
                          cubic(coeff[:, j], np.minimum(radius/r[1], len(r)-1)),
                          moments[j]/np.maximum(radius, r[-1])**(ell+1))
        psi += radial*eval_legendre(ell, mu)
    return psi


def write(path, profile, separation, r, coeff, moments, report):
    """Portable ASCII schema, dimension/finite/range validated by the plugin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('PANGU_BNS_TIME_SYMMETRIC_V1\n')
        f.write(f'{len(r)} {coeff.shape[1]} {len(profile["source"])}\n')
        f.write(' '.join(f'{x:.17e}' for x in
                         [r[1], profile['step'], profile['kappa'], separation,
                          profile['isotropic_radius'], report['fixed_point_abs_residual']])+'\n')
        np.savetxt(f, moments[None, :], fmt='%.17e')
        np.savetxt(f, profile['source'], fmt='%.17e')
        np.savetxt(f, coeff, fmt='%.17e')
    report['table_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['table'] = str(path.resolve())
    path.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--rhoc', type=float, default=1.28e-3)
    p.add_argument('--kappa', type=float, default=100.)
    p.add_argument('--separation', type=float, default=24.)
    p.add_argument('--step', type=float, default=.025)
    p.add_argument('--lmax', type=int, default=48)
    p.add_argument('--angular-points', type=int, default=128)
    a = p.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in [a.rhoc, a.kappa, a.separation, a.step]):
        p.error('physical parameters and grid step must be finite and positive')
    profile = tov(a.rhoc, a.kappa)
    r, c, m, report = solve(profile, a.separation, a.step, a.lmax, a.angular_points)
    write(a.output, profile, a.separation, r, c, m, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
