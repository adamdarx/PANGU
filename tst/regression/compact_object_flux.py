#!/usr/bin/env python3
"""Independent analytic/tensor checks of the offline EM diagnostic formulas."""
from pathlib import Path
import importlib.util
import numpy as np


def main():
    path = Path(__file__).resolve().parents[2] / "plugins/compact_objects/flux_diagnostics.py"
    spec = importlib.util.spec_from_file_location("compact_flux", path)
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    rng = np.random.default_rng(90210)
    n = 200
    v = rng.normal(size=(n, 3)) * .1
    B = rng.normal(size=(n, 3))
    W = 1 / np.sqrt(1 - np.sum(v*v, axis=-1))
    flat = diagnostic.flux_densities(np.ones(n), W[:, None] * v, B,
                                    np.broadcast_to(np.eye(3), (n, 3, 3)), np.ones(n), np.zeros((n, 3)))
    # Ideal MHD E=-v cross B; Poynting vector E cross B.
    np.testing.assert_allclose(flat["em_energy_flux_density"], np.cross(-np.cross(v, B), B), rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(flat["electromagnetic_eulerian_energy"],
                               .5*(np.sum(B*B, axis=-1)+np.sum(np.cross(v, B)**2, axis=-1)), rtol=2e-13, atol=2e-14)
    # General non-diagonal spatial metric, nonunit lapse and nonzero shift.
    a = rng.normal(size=(n, 3, 3)) * .3
    gamma = np.einsum("...ki,...kj->...ij", a, a) + np.eye(3)
    alpha = rng.uniform(.4, 1.3, n)
    beta = rng.normal(size=(n, 3)) * .2
    U = rng.normal(size=(n, 3)) * .2
    bt = rng.normal(size=(n, 3))
    rho = rng.uniform(.01, 1, n)
    result = diagnostic.flux_densities(rho, U, bt, gamma, alpha, beta)
    g = np.zeros((n, 4, 4))
    g[:, 1:, 1:] = gamma
    g[:, 0, 1:] = g[:, 1:, 0] = np.einsum("...ij,...j->...i", gamma, beta)
    g[:, 0, 0] = -alpha**2 + np.einsum("...i,...ij,...j->...", beta, gamma, beta)
    u, b = result["fluid_four_velocity"], result["magnetic_four_vector"]
    inner = lambda x, y: np.einsum("...i,...ij,...j->...", x, g, y)
    np.testing.assert_allclose(inner(u, u), -1, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(inner(u, b), 0, rtol=0, atol=2e-14)
    b2 = inner(b, b)
    np.testing.assert_allclose(result["magnetic_four_squared"], b2, rtol=2e-13, atol=2e-14)
    tensor = b2[:, None, None] * (u[:, :, None]*u[:, None, :] + .5*np.linalg.inv(g)) - b[:, :, None]*b[:, None, :]
    mixed = np.einsum("...ij,...jk->...ik", tensor, g)
    weight = np.sqrt(-np.linalg.det(g))
    np.testing.assert_allclose(result["em_energy_flux_density"], -weight[:, None]*mixed[:, 1:, 0], rtol=3e-13, atol=2e-14)
    np.testing.assert_allclose(result["mass_flux_density"], (weight*rho)[:, None]*u[:, 1:], rtol=2e-13, atol=2e-14)
    x, weights = np.polynomial.legendre.leggauss(16)
    phi = np.arange(32) * (2*np.pi/32)
    normals = np.stack(np.broadcast_arrays(np.sqrt(1-x[:, None]**2)*np.cos(phi),
                                            np.sqrt(1-x[:, None]**2)*np.sin(phi), x[:, None]), axis=-1)
    weights = np.broadcast_to(weights[:, None]*(2*np.pi/32), x[:, None].shape[:-1] + (32,))
    radius, flux = 7., 1.3
    value = diagnostic.coordinate_sphere_flux(flux * normals / radius**2, normals, weights, radius)
    np.testing.assert_allclose(value, 4*np.pi*flux, rtol=1e-13, atol=1e-13)
    constant = np.broadcast_to([.3, -.2, .7], normals.shape)
    np.testing.assert_allclose(diagnostic.coordinate_sphere_flux(constant, normals, weights, radius), 0, atol=1e-12)
    print("passed: SR Poynting/energy, curved ADM tensor contractions, baryon current and sphere Jacobian")


if __name__ == "__main__":
    main()
