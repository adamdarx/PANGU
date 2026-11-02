#!/usr/bin/env python3
"""Compare a completed spherical EMS evolution with independent static ODEs.

The static ODE uses the measured horizon data as boundary values. This tests
the remaining exterior equations; it does not independently predict those
boundary values. The paper growth observable is log|d(phi_h)/dt|, not log|phi_h|.
"""
from pathlib import Path
import argparse
import json

import h5py
import numpy as np
from scipy.integrate import solve_ivp, simpson, trapezoid


def parameters(path):
    result, section = {}, None
    for line in path.read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if text.startswith("<") and text.endswith(">"):
            section = text[1:-1]
        elif "=" in text:
            key, value = text.split("=", 1)
            result[(section, key.strip())] = value.strip()
    return result


def analyze(directory, growth_window=(81., 143.)):
    directory = Path(directory)
    config = parameters(directory / "input.in")
    section = "plugin/scalar_field"
    get = lambda name, default: config.get((section, name), default)
    number = lambda name, default: float(get(name, default))
    if get("backend", "cartesian_flat") != "spherical_ems":
        raise ValueError("static comparison requires the nonlinear spherical_ems backend")
    seed_mass, Q, alpha = number("bh_mass", 1), number("bh_charge", .9), number("em_alpha", 5)
    mu, quartic, sextic, fa = (number(k, v) for k, v in (("mass", 0), ("quartic", 0), ("sextic", 0), ("decay_constant", 1)))
    coupling, potential = get("em_coupling", "exponential"), get("potential", "polynomial")
    if coupling not in ("exponential", "quadratic") or potential not in ("polynomial", "axion"):
        raise ValueError("unsupported action in static comparison")
    if fa <= 0:
        raise ValueError("axion scale must be positive")
    if potential == "axion":
        V = lambda p: 2 * mu**2 * fa**2 * np.sin(p / (2 * fa))**2
        DV = lambda p: mu**2 * fa * np.sin(p / fa)
    else:
        V = lambda p: .5*mu**2*p*p + .5*quartic*p**4 + sextic*p**6/6
        DV = lambda p: mu**2*p + 2*quartic*p**3 + sextic*p**5
    invf = lambda p: np.exp(-alpha*p*p) if coupling == "exponential" else 1/(1+alpha*p*p)
    dlogf = lambda p: 2*alpha*p if coupling == "exponential" else 2*alpha*p/(1+alpha*p*p)
    h = np.atleast_1d(np.genfromtxt(directory / "ems_history.csv", delimiter=",", names=True))
    if not np.isfinite(h.view(float)).all() or not np.all(np.diff(h["time"]) > 0):
        raise ValueError("history is nonfinite or not strictly increasing")
    rh, ph = float(h["horizon_radius"][-1]), float(h["horizon_phi"][-1])
    if rh <= 0:
        raise ValueError("static comparison requires a resolved outer horizon")
    final = list(directory.glob("*.final.phdf"))
    if len(final) != 1:
        raise ValueError("expected exactly one completed final snapshot")
    with h5py.File(final[0]) as stream:
        t = float(stream["Info"].attrs["Time"])
        if int(stream["Info"].attrs["NumMeshBlocks"]) != 1:
            raise ValueError("spherical EMS comparison expects one radial block")
        radius = stream["VolumeLocations/x"][:].reshape(-1)
        state = stream["scalar.state"][:].reshape(3, -1)
        geometry = stream["ems.geometry"][:].reshape(4, -1)
    if abs(t - h["time"][-1]) > 1e-10 or not np.isfinite(state).all() or not np.isfinite(geometry).all():
        raise ValueError("final snapshot/history mismatch or nonfinite evolved fields")
    slope_N = 1/rh - Q*Q*invf(ph)/rh**3 - 2*rh*V(ph)
    if slope_N <= 0:
        raise ValueError("static ODE requires a nondegenerate outer horizon")
    slope_phi = (DV(ph) - Q*Q*dlogf(ph)*invf(ph)/(2*rh**4))/slope_N

    def rhs(r, y):
        N, phi, x, delta = y
        nr = (1-N)/r - r*N*x*x - 2*r*V(phi) - Q*Q*invf(phi)/r**3
        dr = -r*x*x
        xr = (DV(phi)-Q*Q*dlogf(phi)*invf(phi)/(2*r**4))/N - (2/r+nr/N-dr)*x
        return [nr, x, xr, dr]

    eps = 1e-7 * max(1., rh)
    outer = min(50., float(radius[-1]))
    solution = solve_ivp(rhs, [rh+eps, outer], [slope_N*eps, ph+slope_phi*eps, slope_phi, 0],
                         method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True, max_step=.05)
    if not solution.success:
        raise ValueError(solution.message)
    mask = (radius > 1.05*rh) & (radius < min(30., outer))
    if np.count_nonzero(mask) < 8:
        raise ValueError("insufficient exterior comparison points")
    reference = solution.sol(radius[mask])
    delta_phi, delta_N = state[0, mask]-reference[1], geometry[2, mask]-reference[0]
    norm = float(np.linalg.norm(reference[1]))
    result = dict(time=t, horizon_radius=rh, horizon_phi=ph, points=int(mask.sum()),
                  comparison_interval=[float(radius[mask][0]), float(radius[mask][-1])],
                  action=dict(seed_bh_mass=seed_mass, Q=Q, alpha=alpha, mass=mu, quartic=quartic, sextic=sextic,
                              decay_constant=fa, coupling=coupling, potential=potential),
                  static_phi_relative_l2=float(np.linalg.norm(delta_phi)/norm) if norm > 0 else None,
                  static_phi_max_difference=float(np.max(abs(delta_phi))),
                  static_metric_N_max_difference=float(np.max(abs(delta_N))),
                  max_charge_error=float(np.max(h["charge_error_max"])),
                  final_mass_constraint=float(h["mass_constraint_max"][-1]),
                  boundary_values="measured horizon data; not independent endpoint predictions",
                  family_acceptance="requires resolution and boundary/flux-budget controls")
    time, luminosity = h["time"], h["outgoing_luminosity"]
    if len(time) < 9:
        raise ValueError("finite-radius flux budget requires at least nine history samples")
    indices = np.unique(np.append(np.arange(0, len(time), 2), len(time)-1))
    integral = float(simpson(luminosity, x=time))
    coarse = float(simpson(luminosity[indices], x=time[indices]))
    delta_mass = float(h["outer_mass"][-1]-h["outer_mass"][0])
    budget = delta_mass+integral
    uncertainty = abs(integral-coarse)
    result["mass_flux_budget"] = dict(delta_outer_mass=delta_mass, simpson_integrated_outgoing_flux=integral,
        trapezoid_integrated_outgoing_flux=float(trapezoid(luminosity, x=time)), residual=budget,
        relative_residual=abs(budget)/abs(float(h["outer_mass"][0])),
        decimated_simpson_difference=uncertainty,
        needs_denser_history=bool(uncertainty > .1*max(abs(budget), 10*np.finfo(float).eps*abs(float(h["outer_mass"][0])))),
        interpretation="finite-radius m(R) plus flux through that same R; no Coulomb-tail addition",
        quadrature_control="step-two decimation of actual nonuniform times, not an independent denser-output run")
    if seed_mass == 1 and Q == .9 and alpha == 5 and mu == 0 and quartic == 0 and sextic == 0 and coupling == "exponential":
        time, phi = h["time"], h["horizon_phi"]
        derivative = np.array([np.polynomial.polynomial.polyfit(
            time[i-2:i+3]-time[i], phi[i-2:i+3], 4)[1] for i in range(2, len(time)-2)])
        time = time[2:-2]
        selected = (time >= growth_window[0]) & (time <= growth_window[1]) & (abs(derivative) > 0)
        selected &= (abs(phi[2:-2]) > 1e-5) & (abs(phi[2:-2]) < 1e-3)
        if np.count_nonzero(selected) < 8:
            result["growth"] = dict(available=False, reason="linear-growth window is not adequately sampled")
        else:
            rate = float(np.polyfit(time[selected], np.log(abs(derivative[selected])), 1)[0])
            result["growth"] = dict(available=True, observable="log(abs(d(phi_h)/dt))", rate=rate,
                paper_rate=.07335, relative_difference=abs(rate/.07335-1),
                interval=[float(time[selected][0]), float(time[selected][-1])], samples=int(selected.sum()),
                scalar_amplitude_window=[1e-5, 1e-3])
    np.savez(directory / "static_reference_v2.npz", r=radius[mask], phi=reference[1], N=reference[0], delta=reference[3])
    temporary = directory / "analysis_ems.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    temporary.replace(directory / "analysis_ems.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2))
