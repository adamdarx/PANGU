"""Coordinate flux densities for Cartesian ADM/GRMHD output.

These helpers do not interpolate an AMR mesh or assume a horizon is spherical.
Surface samples and quadrature weights must come from a separately qualified
sampler. In particular, a coordinate sphere's inflow is not an apparent-horizon
accretion rate. B uses PANGU's absorbed-4pi convention.
"""
import numpy as np


def flux_densities(rho, projected_velocity, btilde, spatial_metric, lapse, shift):
    """Return sqrt(-g) rho u^i and -sqrt(-g) T^i_t(EM).

    Vectors have trailing dimension 3, metrics trailing dimensions (3,3).
    projected_velocity = W*v^i; btilde = sqrt(gamma)*B^i. Input arrays may
    have arbitrary common batch dimensions. All returned vectors retain them.
    The temporal energy current is coordinate dependent in a live spacetime;
    it is not a statement of a conserved Killing energy.
    """
    rho, alpha = np.asarray(rho), np.asarray(lapse)
    U, bt, gamma, beta = map(np.asarray, (projected_velocity, btilde, spatial_metric, shift))
    if any(not np.isfinite(q).all() for q in (rho, alpha, U, bt, gamma, beta)):
        raise ValueError("nonfinite GRMHD diagnostic input")
    if np.any(rho <= 0) or np.any(alpha <= 0):
        raise ValueError("diagnostics require positive density and lapse")
    if not np.allclose(gamma, np.swapaxes(gamma, -1, -2), rtol=1e-13, atol=1e-14):
        raise ValueError("spatial metric is not symmetric")
    if np.any(np.linalg.eigvalsh(gamma) <= 0):
        raise ValueError("spatial metric is not positive definite")
    dot = lambda a, b: np.einsum("...i,...ij,...j->...", a, gamma, b)
    sqrt_gamma = np.sqrt(np.linalg.det(gamma))
    W = np.sqrt(1 + dot(U, U))
    v = U / W[..., None]
    B = bt / sqrt_gamma[..., None]
    Bv = dot(B, v)
    B2 = dot(B, B)
    b2 = B2 / W**2 + Bv**2
    u0 = W / alpha
    ui = U - u0[..., None] * beta
    b0 = W * Bv / alpha
    bi = B / W[..., None] + b0[..., None] * (alpha[..., None] * v - beta)
    # u_t=W*(-alpha+beta_i*v^i), b_t=(B_i*beta^i)/W+alpha*b0*u_t/W.
    ut = W * (-alpha + dot(beta, v))
    b_t = dot(B, beta) / W + alpha * b0 * ut / W
    weight = alpha * sqrt_gamma
    mass = (weight * rho)[..., None] * ui
    electromagnetic = -weight[..., None] * (b2[..., None] * ui * ut[..., None] - bi * b_t[..., None])
    return dict(mass_flux_density=mass, em_energy_flux_density=electromagnetic,
                lorentz_factor=W, magnetic_four_squared=b2,
                magnetization=b2 / rho, magnetic_field=B,
                baryon_density=sqrt_gamma * rho * W,
                electromagnetic_eulerian_energy=b2 * (W**2 - .5) - (alpha * b0)**2,
                fluid_four_velocity=np.concatenate((u0[..., None], ui), axis=-1),
                magnetic_four_vector=np.concatenate((b0[..., None], bi), axis=-1))


def coordinate_sphere_flux(vector_density, unit_normals, solid_angle_weights, radius):
    """Integrate an already densitized Cartesian vector over a coordinate sphere.

    Positive means outward transport. unit_normals are Euclidean coordinate
    normals; sqrt(gamma) is already included in vector_density. The explicit
    radius**2 is the Cartesian sphere's coordinate surface Jacobian.
    """
    values, normals, weights = map(np.asarray, (vector_density, unit_normals, solid_angle_weights))
    if values.shape != normals.shape or values.shape != weights.shape + (3,):
        raise ValueError("surface values, normals and weights have incompatible shapes")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("coordinate extraction radius must be positive")
    if any(not np.isfinite(q).all() for q in (values, normals, weights)) or np.any(weights <= 0):
        raise ValueError("invalid surface samples or quadrature weights")
    if not np.allclose(np.sum(normals**2, axis=-1), 1, atol=1e-12, rtol=1e-12):
        raise ValueError("coordinate surface normals must be unit vectors")
    if not np.isclose(np.sum(weights), 4 * np.pi, rtol=1e-12, atol=1e-12):
        raise ValueError("solid angle weights must cover a full sphere")
    return float(radius**2 * np.sum(np.sum(values * normals, axis=-1) * weights))
